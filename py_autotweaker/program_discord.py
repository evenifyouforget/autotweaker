"""
A blank Discord bot template using discord.py.
This bot responds to commands when mentioned (e.g., @BotName command).
It includes basic ping and hello commands, and functionality for managing
"design jobs" in memory. It now also supports attaching a JSON configuration
to a job, retrieving it, automatically loading the design structure,
and runs a continuous background task with garden maintenance logic.
Gardens marked as "frozen" will be skipped in the background loop.

Requires Python 3.8 or higher.
"""

# Import necessary modules from the discord.py library
import discord
from discord.ext import commands
import os
import asyncio # Import asyncio for running async functions
import sys # Import sys for exiting the script
import re # Import regex module for parsing design IDs from URLs
from dataclasses import dataclass, field # Import dataclass and field for default_factory
import json # Import json for parsing and serializing JSON
# Import Optional for type hints compatible with Python < 3.10
from typing import Optional
from pathlib import Path # For future disk storage management
# Imports for design retrieval (assuming these are in get_design.py in the same directory)
from get_design import retrieveDesign, designDomToStruct, FCDesignStruct
from save_design import save_design # Import the save_design function
from .auto_login import auto_login_get_user_id # Import auto_login_get_user_id for bot account login
# Assuming json_validate is in a file named json_validate.py in the same directory
from .job_struct import Creature, Garden, GardenStatus # Import GardenStatus as well for checkup return
from .json_validate import json_validate
from .performance import get_thread_count

# --- Constants ---
MAX_THREADS = get_thread_count() # Global constant for maximum threads available across all gardens
MAX_SNAPSHOT_K = 3 # Maximum number of best creatures to snapshot and upload
RETAIN_BEST_K = max(MAX_THREADS, MAX_SNAPSHOT_K) + 1 # Global constant for retaining best K creatures in evolution
MAX_GARDEN_SIZE = RETAIN_BEST_K * 3 # Global constant for the maximum garden size

# Global counter for background loop iterations, used for round-robin prioritization
_background_loop_iteration_counter: int = 0

# --- Design Job Management ---

@dataclass
class Job:
    """
    Represents a job associated with a design.
    Future fields could include: status, progress, assigned_user, etc.
    """
    config_json: Optional[dict] = None
    design_struct: Optional[FCDesignStruct] = None
    garden: Optional[Garden] = None
    # Added a list to store errors, using default_factory for mutable default arguments
    errors: list[str] = field(default_factory=list)
    # New field to track when this job last received threads for round-robin prioritization
    last_turn_allocated: int = 0 # Initialize to 0, jobs with lower numbers get priority

    def is_frozen(self) -> bool:
        """
        Stub function to determine if a job is "frozen" and should not
        undergo garden maintenance. Currently always returns False.
        """
        # This will eventually contain logic based on budget, time, etc.
        return False

# Global dictionary to store jobs, keyed by design ID.
# This data will be stored in memory and will reset if the bot restarts.
jobs: dict[int, Job] = {}

def extract_design_id(input_string: str) -> Optional[int]:
    """
    Extracts a design ID from a string, which can be either a raw integer
    or a URL containing 'designId=<number>'.

    Args:
        input_string (str): The string to parse (e.g., "12345" or "https://ft.jtai.dev/?designId=12707044).

    Returns:
        int | None: The extracted design ID as an integer, or None if not found.
    """
    # Try converting directly to an integer first
    try:
        return int(input_string)
    except ValueError:
        pass # Not a direct integer, proceed to regex

    # If not a direct integer, try to find it in a URL using regex
    # Matches 'designId=' followed by one or more digits
    match = re.search(r'designId=(\d+)', input_string)
    if match:
        try:
            return int(match.group(1)) # Extract the digits and convert to int
        except ValueError:
            return None # Should not happen if regex only captures digits, but good practice

    return None # No valid design ID found

def get_or_create_job(design_id: int) -> Job:
    """
    Retrieves an existing job for a given design ID, or creates a new one
    if it doesn't already exist.

    Args:
        design_id (int): The unique integer ID of the design.

    Returns:
        Job: The Job object associated with the design ID.
    """
    if design_id not in jobs:
        print(f"Initializing new job for design ID: {design_id}")
        jobs[design_id] = Job() # Create a new empty Job
    else:
        print(f"Retrieving existing job for design ID: {design_id}")
    return jobs[design_id]

# --- Discord Bot Setup ---

# Define the bot's intents. Intents specify which events your bot wants to receive from Discord.
# It's good practice to explicitly define your intents to avoid unnecessary data and for future-proofing.
# For a basic bot, you'll likely need default intents, plus message content if you're reading messages.
intents = discord.Intents.default()
intents.message_content = True  # Required to read message content for commands

# Initialize the bot with a command prefix and intents.
# The bot will ONLY respond to commands when mentioned (e.g., @BotName ping).
bot = commands.Bot(command_prefix=commands.when_mentioned, intents=intents)

# Event: on_ready
# This event fires when the bot has successfully connected to Discord.
@bot.event
async def on_ready():
    """
    Called when the bot is ready and has connected to Discord.
    Prints a confirmation message to the console.
    """
    print(f'Logged in as {bot.user.name} ({bot.user.id})')
    print('------')
    # Start the background loop when the bot is ready
    bot.loop.create_task(background_loop())

# Command: hello
# This is a simple command that responds with "Hello!" when a user types "@BotName hello".
@bot.command()
async def hello(ctx):
    """
    A simple command that sends a 'Hello!' message back to the channel.
    Usage: @BotName hello
    """
    await ctx.send('Hello!')

# Command: ping
# A common command to check the bot's latency.
@bot.command()
async def ping(ctx):
    """
    Responds with the bot's current latency (ping).
    Usage: @BotName ping
    """
    # ctx.bot.latency returns the average latency in seconds. Multiply by 1000 for milliseconds.
    latency_ms = round(bot.latency * 1000)
    await ctx.send(f'Pong! {latency_ms}ms')

# Command: status
# Checks the status of a design job, creating it if it doesn't exist.
@bot.command()
async def status(ctx, *, design_input: str):
    """
    Checks the status of a design job. If a job for the given design ID
    doesn't exist, it will be initialized.
    It also attempts to load the design struct if not already loaded.
    Supports raw ID or full URL input.

    Usage:
      @BotName status 12345
      @BotName status https://fantasticcontraption.com/original/?designId=12668445
    """
    design_id = extract_design_id(design_input)

    if design_id is None:
        await ctx.send(
            f"Sorry, I couldn't understand that design ID. "
            f"Please provide a number or a link like `https://example.com/?designId=12345`."
        )
        return

    # Get or create the job for this design ID
    job = get_or_create_job(design_id)
    # Don't clear errors immediately, but rather *after* they are reported.
    # job.errors.clear() # Clear previous errors before attempting new operations

    # Attempt to retrieve and parse design struct if not already loaded
    if job.design_struct is None:
        print(f"Attempting to load design struct for design ID: {design_id}")
        try:
            # retrieveDesign is a blocking call as noted, but we call it in an async context
            # Consider using bot.loop.run_in_executor for heavy blocking I/O in production
            design_dom = await asyncio.to_thread(retrieveDesign, design_id)

            if design_dom is None:
                job.errors.append(f"Failed to retrieve design DOM for ID {design_id}. It might not exist or there was a network issue.")
            else:
                try:
                    job.design_struct = designDomToStruct(design_dom)
                    print(f"Successfully loaded design struct for design ID: {design_id}")
                except Exception as e:
                    job.errors.append(f"Failed to parse design DOM for ID {design_id}: {e}")
        except Exception as e:
            job.errors.append(f"Error during design retrieval for ID {design_id}: {e}")
            print(f"Error retrieving design {design_id}: {e}")

    config_status = "No JSON config set."
    if job.config_json is not None:
        config_status = "JSON config is set."

    design_struct_status = "No design struct loaded."
    if job.design_struct is not None:
        design_struct_status = "Design struct is loaded."

    garden_status = "Garden not initialized."
    if job.garden is not None:
        garden_status = "Garden initialized."
    
    frozen_status = "Not frozen."
    if job.is_frozen():
        frozen_status = "Frozen (maintenance skipped)."


    # Report errors if any, with trimming
    error_report_lines = []
    if job.errors:
        error_report_lines.append("\n**Errors:**")
        # Calculate base length of the main response to ensure trimming accounts for it
        base_response_length = len(
            f"Job for design ID **{design_id}** is active. "
            f"Status: {config_status}. {design_struct_status}. {garden_status}. {frozen_status}"
        ) + len("\n**Errors:**\n") # Account for the error header and newline

        current_length = base_response_length

        for i, err in enumerate(job.errors):
            line = f"- {err}"
            # Check if adding this line would exceed the soft limit (1900 characters)
            # We also add 2 to current_length for a potential newline character.
            # And account for the length of " (more errors omitted)" if it gets added.
            if current_length + len(line) + (2 if i < len(job.errors) - 1 else 0) > 1900: # +2 for potential newline
                error_report_lines.append("... (more errors omitted)")
                break
            error_report_lines.append(line)
            current_length += len(line) + 1 # +1 for newline

    full_response = (
        f"Job for design ID **{design_id}** is active. "
        f"Status: {config_status}. {design_struct_status}. {garden_status}. {frozen_status}"
    )
    if error_report_lines:
        full_response += "\n".join(error_report_lines)

    await ctx.send(full_response)
    job.errors.clear() # Clear errors after reporting them

# Command: set_config
# Sets the JSON configuration for a design job.
@bot.command()
async def set_config(ctx, design_input: str, *, json_content: str = None):
    """
    Sets the JSON configuration for a design job.
    You can provide JSON directly in the message or as a .json attachment.

    Usage:
      @BotName set_config 12345 ```json\n{\"key\": \"value\"}\n```
      @BotName set_config https://example.com/?designId=12345 (and attach a .json file)
    """
    design_id = extract_design_id(design_input)

    if design_id is None:
        await ctx.send(
            f"Sorry, I couldn't understand that design ID. "
            f"Please provide a number or a link like `https://example.com/?designId=12345`."
        )
        return

    job = get_or_create_job(design_id)
    job.errors.clear() # Clear previous errors
    parsed_json = None

    # Check for JSON in attachment
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
        if attachment.content_type == 'application/json':
            try:
                json_bytes = await attachment.read()
                parsed_json = json.loads(json_bytes.decode('utf-8'))
                print(f"Parsed JSON from attachment for design ID {design_id}")
            except json.JSONDecodeError as e:
                job.errors.append(f"Failed to parse JSON from attachment: {e}")
                await ctx.send(f"Failed to parse JSON from attachment: {e}")
                return
            except Exception as e:
                job.errors.append(f"An error occurred reading attachment: {e}")
                await ctx.send(f"An error occurred reading attachment: {e}")
                return
        else:
            await ctx.send("Attached file is not a JSON file (`.json` content type).")
            return
    # Check for JSON in message content (if no attachment)
    elif json_content:
        # Try to extract JSON from a markdown code block if present
        json_match = re.search(r'```json\n(.*)```', json_content, re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
        else:
            json_string = json_content # Assume plain JSON if no markdown block

        try:
            parsed_json = json.loads(json_string)
            print(f"Parsed JSON from message content for design ID {design_id}")
        except json.JSONDecodeError as e:
            job.errors.append(f"Failed to parse JSON from message content: {e}")
            await ctx.send(f"Failed to parse JSON from message content: {e}")
            return
    else:
        await ctx.send(
            "Please provide JSON directly in the message (preferably in a `json` markdown block) "
            "or as a `.json` attachment."
        )
        return

    if parsed_json:
        if json_validate(parsed_json): # Validate the parsed JSON
            job.config_json = parsed_json
            await ctx.send(f"JSON config successfully set for design ID **{design_id}**.")
        else:
            job.errors.append(f"JSON config for design ID {design_id} is invalid according to `json_validate`.")
            await ctx.send(f"JSON config for design ID **{design_id}** is invalid according to `json_validate`.")
    else:
        await ctx.send(f"No valid JSON found to set for design ID **{design_id}**.")


# New Command: get_config
# Outputs or downloads the JSON config for a design job.
@bot.command()
async def get_config(ctx, *, design_input: str):
    """
    Retrieves and outputs the JSON configuration for a design job.
    If the JSON is too large, it will be sent as a file.

    Usage:
      @BotName get_config 12345
      @BotName get_config https://example.com/?designId=12345
    """
    design_id = extract_design_id(design_input)

    if design_id is None:
        await ctx.send(
            f"Sorry, I couldn't understand that design ID. "
            f"Please provide a number or a link like `https://example.com/?designId=12345`."
        )
        return

    job = get_or_create_job(design_id)
    job.errors.clear() # Clear previous errors

    if job.config_json is None:
        await ctx.send(f"No JSON config has been set for design ID **{design_id}** yet.")
        return

    json_output_string = json.dumps(job.config_json, indent=2)

    # Discord message character limit is 2000.
    # If the JSON is too long, send it as a file.
    if len(json_output_string) > 1900: # Give some buffer for markdown
        filename = f"design_{design_id}_config.json"
        with open(filename, 'w') as f:
            f.write(json_output_string)
        await ctx.send(
            f"Here is the JSON config for design ID **{design_id}**:",
            file=discord.File(filename)
        )
        os.remove(filename) # Clean up the temporary file
    else:
        await ctx.send(f"Here is the JSON config for design ID **{design_id}**:\n```json\n{json_output_string}\n```")

# New Command: start_job
# Tries to initialize the Garden for a design job.
@bot.command()
async def start_job(ctx, *, design_input: str):
    """
    Attempts to initialize the Garden (start the job) for a given design.
    Requires both the design structure and a JSON configuration to be present.

    Usage:
      @BotName start_job 12345
      @BotName start_job https://fantasticcontraption.com/?designId=12345
    """
    design_id = extract_design_id(design_input)

    if design_id is None:
        await ctx.send(
            f"Sorry, I couldn't understand that design ID. "
            f"Please provide a number or a link like `https://example.com/?designId=12345`."
        )
        return

    job = get_or_create_job(design_id)
    job.errors.clear() # Clear previous errors before attempting to start the job

    # Check prerequisites
    if job.design_struct is None:
        job.errors.append("Cannot start garden: Design structure not loaded. Use `@BotName status` to load it first.")
    if job.config_json is None:
        job.errors.append("Cannot start garden: JSON config not set. Use `@BotName set_config` to set it first.")

    if job.errors: # If there are existing errors or new ones from prerequisite checks
        error_report = "\n**Errors:**\n" + "\n".join([f"- {err}" for err in job.errors])
        await ctx.send(f"Could not start garden for design ID **{design_id}**. Please fix the listed issues.{error_report}")
        job.errors.clear()
        return

    if job.garden is not None:
        await ctx.send(f"Garden for design ID **{design_id}** is already running.")
        return

    try:
        # Initialize the garden
        # Assumes Creature takes an FCDesignStruct and Garden takes a list of Creatures, size, and config
        job.garden = Garden([Creature(job.design_struct)], MAX_GARDEN_SIZE, job.config_json)
        await ctx.send(f"Garden successfully initialized for design ID **{design_id}**!")
        print(f"Garden initialized for design ID {design_id}")
    except Exception as e:
        job.errors.append(f"Failed to initialize garden for ID {design_id}: {e}")
        await ctx.send(f"Failed to initialize garden for design ID **{design_id}**: {e}")
        print(f"Error initializing garden for design {design_id}: {e}")
    finally:
        job.errors.clear() # Clear errors after attempting initialization

# New Command: snapshot
@bot.command()
async def snapshot(ctx, design_input: str, k: Optional[int] = None):
    """
    Gets the best 'k' creatures from the garden, uploads them to the FC servers,
    and returns the links in a message.

    Usage:
      @BotName snapshot 12345 [k]
      @BotName snapshot https://fantasticcontraption.com/?designId=12345 [k]
      'k' is optional and defaults to MAX_SNAPSHOT_K.
    """
    design_id = extract_design_id(design_input)

    if design_id is None:
        await ctx.send(
            f"Sorry, I couldn't understand that design ID. "
            f"Please provide a number or a link like `https://example.com/?designId=12345`."
        )
        return

    job = get_or_create_job(design_id)
    job.errors.clear() # Clear previous errors

    if job.garden is None:
        job.errors.append("Garden not initialized for this design. Use `@BotName start_job` first.")
        await ctx.send(f"Cannot snapshot for design ID **{design_id}**. Garden not active. Please use `@BotName start_job`.")
        job.errors.clear()
        return

    upload_best_k = MAX_SNAPSHOT_K # Default value
    if k is not None:
        if k > MAX_SNAPSHOT_K:
            await ctx.send(f"Snapshot count cannot exceed {MAX_SNAPSHOT_K}. Using {MAX_SNAPSHOT_K} instead of {k}.")
            upload_best_k = MAX_SNAPSHOT_K
        elif k <= 0:
            await ctx.send("Snapshot count must be positive. No designs will be saved.")
            job.errors.append("Invalid snapshot count provided.")
            job.errors.clear()
            return
        else:
            upload_best_k = k

    if not job.garden.creatures:
        await ctx.send(f"No creatures available in the garden for design ID **{design_id}** to snapshot.")
        job.errors.clear()
        return

    # Attempt to get user ID for FC server login
    user_id = None
    try:
        user_id = await asyncio.to_thread(auto_login_get_user_id)
    except Exception as e:
        # DO NOT leak user_id in error messages
        job.errors.append("Failed to auto-login to FC servers for snapshot.")
        await ctx.send(f"Failed to snapshot for design ID **{design_id}**: Could not log in to FC servers. (Check bot credentials)")
        job.errors.clear()
        return

    if user_id is None: # Safeguard, though exception should catch most login issues
        job.errors.append("Could not obtain user ID for FC servers during snapshot.")
        await ctx.send(f"Failed to snapshot for design ID **{design_id}**: Could not obtain user ID. (Check bot credentials)")
        job.errors.clear()
        return

    saved_links_messages = []
    snapshot_count = 0
    await ctx.send(f"Attempting to snapshot top {upload_best_k} designs for ID **{design_id}**...")

    # Ensure we don't try to snapshot more creatures than are available
    creatures_to_snapshot = job.garden.creatures[:upload_best_k]

    for rank, creature in enumerate(creatures_to_snapshot):
        try:
            # Construct design name (max 15 chars)
            # Format: FC<design_id>-<rank+1> (e.g., FC12345678-1)
            design_name = f"FC{design_id}-{rank+1}"
            # Ensure name fits within 15 characters, truncate if necessary
            if len(design_name) > 15:
                design_name = design_name[:12] + "..." # Truncate and add ellipsis

            # Construct description (max 50 chars)
            score_status = f"score {creature.best_score}"
            emoji = ""
            if creature.best_score is not None and creature.best_score < 0: # Assuming negative score means solve
                score_status = f"SOLVED! Score: {creature.best_score}"
                emoji = " 🎉" # Celebration emoji for a solve

            description = f"Based on {design_id}, {score_status}{emoji}"
            # Ensure description fits within 50 characters, truncate if necessary
            if len(description) > 50:
                description = description[:47] + "..." # Truncate and add ellipsis

            # Call save_design which is a blocking operation, so run in a thread
            saved_design_id = await asyncio.to_thread(
                save_design, creature.design_struct, user_id, name=design_name, description=description
            )
            link = f"https://ft.jtai.dev/?designId={saved_design_id}"
            saved_links_messages.append(f"{rank+1}. {link} (Score: {creature.best_score}{emoji})")
            snapshot_count += 1
            print(f"{rank+1}. Saved to {link} achieving score of {creature.best_score}")

        except Exception as e:
            job.errors.append(f"Failed to save creature {rank+1} for design ID {design_id}: {e}")
            print(f"Error saving creature {rank+1} for design {design_id}: {e}")
            # Do not break, try to save other creatures

    if saved_links_messages:
        response_message = f"Successfully snapshotted {snapshot_count} designs for ID **{design_id}**:"
        response_message += "\n" + "\n".join(saved_links_messages)
        await ctx.send(response_message)
    else:
        if job.errors:
            # Errors would have been sent immediately, but this covers if no links were generated
            # but some errors happened.
            await ctx.send(f"Failed to snapshot any designs for ID **{design_id}**. Please check previous error messages or bot console.")
        else:
            await ctx.send(f"No designs were snapshotted for ID **{design_id}**. "
                           f"This might be because no creatures were available in the garden, "
                           f"or the specified `k` value ({upload_best_k}) was too high for the available creatures.")
    
    # Report any lingering errors from this command
    if job.errors:
        error_report_lines = ["\n**Snapshot Errors:**"]
        # Calculate base length dynamically, accounting for the main response already sent or planned
        # This is a bit tricky since the main response might have been sent already.
        # So, we'll aim for 1900 chars for this error message itself.
        current_length = len("\n**Snapshot Errors:**\n")
        
        for i, err in enumerate(job.errors):
            line = f"- {err}"
            # Check if adding this line would exceed the soft limit (1900 characters)
            if current_length + len(line) + (2 if i < len(job.errors) - 1 else 0) > 1900:
                error_report_lines.append("... (more errors omitted)")
                break
            error_report_lines.append(line)
            current_length += len(line) + 1
        await ctx.send("".join(error_report_lines))
    
    job.errors.clear() # Clear errors after reporting them


# --- Background Task ---
async def background_loop():
    """
    A continuous background task that runs while the bot is active.
    It performs garden maintenance logic for all active jobs,
    distributing thread resources based on congestion and a round-robin approach.
    """
    await bot.wait_until_ready() # Ensure the bot is fully ready before starting the loop

    global _background_loop_iteration_counter # Declare global to modify

    while not bot.is_closed():
        _background_loop_iteration_counter += 1
        print(f"Background loop running (Iteration {_background_loop_iteration_counter}) - performing garden maintenance...")
        
        # 1. Identify eligible jobs (have a garden and are not frozen)
        eligible_jobs_list = []
        for design_id, job in jobs.items():
            if job.garden is not None and not job.is_frozen():
                eligible_jobs_list.append((design_id, job))

        num_eligible_jobs = len(eligible_jobs_list)
        threads_to_allocate_per_job = {} # To store how many threads each eligible job gets

        if num_eligible_jobs == 0:
            print("No eligible gardens to maintain.")
        elif num_eligible_jobs == 1:
            design_id, job = eligible_jobs_list[0]
            threads_to_allocate_per_job[design_id] = MAX_THREADS
            job.last_turn_allocated = _background_loop_iteration_counter # Update turn for the single job
            print(f"Single garden {design_id} gets all {MAX_THREADS} threads.")
        else:
            # 2. Sort eligible jobs for round-robin distribution (lowest last_turn_allocated first)
            eligible_jobs_list.sort(key=lambda item: item[1].last_turn_allocated)

            base_threads_per_job = MAX_THREADS // num_eligible_jobs
            remainder_threads = MAX_THREADS % num_eligible_jobs

            for i, (design_id, job) in enumerate(eligible_jobs_list):
                allocated_threads = base_threads_per_job
                # Distribute remainder threads to jobs that have been waiting the longest
                if remainder_threads > 0 and i < remainder_threads:
                    allocated_threads += 1
                    remainder_threads -= 1
                
                threads_to_allocate_per_job[design_id] = allocated_threads
                if allocated_threads > 0: # Only update turn if the job actually received threads
                    job.last_turn_allocated = _background_loop_iteration_counter
                print(f"Allocating {allocated_threads} threads to garden {design_id}.")
        
        # 3. Perform maintenance for all eligible gardens based on allocated threads
        for design_id, job in list(jobs.items()): # Iterate over all jobs again (copy)
            if job.garden is not None and not job.is_frozen():
                try:
                    # Get the number of threads allocated to this specific job in this iteration
                    allocated_threads_for_this_job = threads_to_allocate_per_job.get(design_id, 0)
                    
                    # Perform garden checkup
                    garden_status = job.garden.checkup()
                    print(f"Garden {design_id} Checkup: Active Threads={garden_status.num_active_threads}, Generation={garden_status.generation}")

                    # Perform garden evolution. Note: As per instructions, evolve still uses global MAX_THREADS.
                    job.garden.evolve(MAX_THREADS, RETAIN_BEST_K)
                    print(f"Garden {design_id} Evolved.")

                    # Calculate how many NEW threads to start based on allocated threads and current active threads
                    # We ensure this is never negative, meaning we won't try to "stop" threads here.
                    new_threads_allowed = max(0, allocated_threads_for_this_job - garden_status.num_active_threads)

                    if new_threads_allowed > 0:
                        job.garden.start(new_threads_allowed)
                        print(f"Garden {design_id} Started {new_threads_allowed} new threads.")
                    else:
                        print(f"Garden {design_id} has enough active threads ({garden_status.num_active_threads}) or was allocated {allocated_threads_for_this_job} new threads (no new threads started).")
                        # This covers cases where it's over-allocated (more active threads than allocated)
                        # or allocated 0 new threads, and existing threads will eventually terminate.

                except Exception as e:
                    job.errors.append(f"Error during background garden maintenance for ID {design_id}: {e}")
                    print(f"Error maintaining garden {design_id}: {e}")
            elif job.garden is None:
                # This print statement informs about jobs without an initialized garden
                print(f"Design ID {design_id} has no active garden to maintain.")
            elif job.is_frozen():
                # This print statement informs about frozen gardens being skipped
                print(f"Garden for design ID {design_id} is FROZEN. Skipping maintenance.")

        # Avoid busy loop
        # Real benchmark - example job finished in about 5 seconds
        # We don't want to introduce too much artificial latency by waiting to update
        await asyncio.sleep(0.1)

# Define the main function to encapsulate the bot's execution logic
async def main():
    """
    Main function to initialize and run the Discord bot.
    Retrieves the bot token from the 'DISCORD_BOT_TOKEN' environment variable.
    """
    bot_token = os.environ.get('DISCORD_BOT_TOKEN') # Retrieve token directly here

    if bot_token: # Check if the token is provided (not None or empty string)
        try:
            # bot.start is used here instead of bot.run within an async function
            await bot.start(bot_token)
        except discord.LoginFailure:
            print("Error: Invalid bot token was provided. Please check your DISCORD_BOT_TOKEN.")
        except Exception as e:
            print(f"An unexpected error occurred while running the bot: {e}")
    else:
        # If bot_token is empty or None, print an error and exit.
        print("Error: Discord bot token is missing. Please set the 'DISCORD_BOT_TOKEN' environment variable.")
        sys.exit(1) # Exit the script with an error code

def program_discord(args):
    """
    External entry point for integrations
    """
    # This entry point for integrations now just calls main() without args
    asyncio.run(main())

# Entry point for the script.
# This ensures that main() is called only when the script is executed directly.
if __name__ == "__main__":
    # The main function now retrieves the token itself, so we just call it.
    asyncio.run(main())
