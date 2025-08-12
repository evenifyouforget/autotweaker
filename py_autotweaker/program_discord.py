"""
A Discord bot for managing Fantastic Contraption design evolution jobs.

This bot responds to commands when mentioned (e.g., @BotName command).
It enables users to:
- Start and "reheat" (fund) design evolution jobs.
- View the status of active and frozen jobs, including time budgets and contributors.
- Subscribe to notifications for job milestones (e.g., when a design solves or times out).
- Snapshot the best evolved designs by uploading them to the FC servers.

Jobs are managed in memory; data will reset if the bot restarts.
Thread resource allocation for active jobs is dynamically managed based on congestion
and a round-robin priority system.

Requires Python 3.8 or higher.
"""

# --- Standard Library Imports ---
import os
import asyncio
import sys
import re
import json
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# --- Third-Party Library Imports ---
import discord
from discord.ext import commands

# --- Local Module Imports (Assumed to be in the same package/directory) ---
# These modules are essential for the bot's functionality:
#   - get_design: Handles fetching design data from Fantastic Contraption servers.
#   - save_design: Manages uploading evolved designs back to FC servers.
#   - auto_login: Provides a bot account login to FC for server interactions.
#   - job_struct: Defines data structures like Creature, Garden, and GardenStatus.
#   - json_validate: Contains logic for validating job configuration JSON.
#   - performance: Provides system performance-related utilities, e.g., thread count.
from get_design import retrieveDesign, designDomToStruct, FCDesignStruct
from save_design import save_design
from .auto_login import auto_login_get_user_id
from .job_struct import Creature, Garden, GardenStatus
from .json_validate import json_validate
from .performance import get_thread_count # Nontrivial library function for available threads

# --- Global Constants & Configuration ---

# Resource allocation constants
MAX_THREADS_GLOBAL = get_thread_count() # Maximum compute threads available across all gardens collectively.

# Job evolution and snapshotting constants
DEFAULT_SNAPSHOT_K = 3 # Default number of best creatures to snapshot and upload for notifications/command
RETAIN_BEST_K_EVOLUTION = max(MAX_THREADS_GLOBAL, DEFAULT_SNAPSHOT_K) + 1 # Number of best creatures to retain during garden evolution.
                                                                           # Ensures enough creatures for both compute and snapshotting.
MAX_GARDEN_SIZE = RETAIN_BEST_K_EVOLUTION * 3 # Maximum number of creatures a single garden can hold,
                                               # providing sufficient buffer for new generations and diversity.

# Time budget (Reheating) constants
# This tuple defines the active duration granted to a job based on the number of unique funders.
# Index `n-1` corresponds to `n` funders. Values are in seconds.
# This table allows fine-grained, superlinear scaling for community-funded jobs.
FUNDING_DURATIONS_SECONDS = (
    3600,        # 1 hour for 1 funder
    4 * 3600,    # 4 hours for 2 funders
    12 * 3600,   # 12 hours for 3 funders
    36 * 3600,   # 36 hours for 4 funders
    72 * 3600    # 72 hours (3 days) for 5+ funders
)
MAX_JOB_ACTIVE_TIME_SECONDS = 3 * 24 * 3600  # Absolute maximum active time any single job can accumulate (3 days).
                                              # Acts as a hard cap regardless of funders or table values.

# User-specific limits
MAX_ACTIVE_JOBS_PER_USER = 3 # Maximum number of unfrozen jobs a single user can actively "reheat" at one time.

# Internal bot state management
_background_loop_iteration_counter: int = 0 # Increments with each background loop pass, used for round-robin thread allocation.


# --- Job Data Structure ---

@dataclass
class Job:
    """
    Represents a single Fantastic Contraption design evolution job managed by the bot.
    Each job maintains its configuration, the design structure being evolved,
    and a dedicated 'Garden' object for evolutionary computation.
    """
    config_json: Optional[dict] = None              # JSON configuration for the job's evolution parameters.
    design_struct: Optional[FCDesignStruct] = None  # Parsed design structure fetched from FC servers.
    garden: Optional[Garden] = None                 # The active Garden object where creatures are evolved.

    errors: List[str] = field(default_factory=list) # List to store recent operational errors specific to this job.

    # Thread allocation management for fairness and resource balancing.
    last_turn_allocated: int = 0                    # Stores the background loop iteration count when this job last received compute threads,
                                                    # used to prioritize jobs that have been waiting longest.

    # Time budgeting (Reheating) fields to manage job activity based on user engagement.
    expiration_time: Optional[float] = None         # Unix timestamp when the job's active time expires and it becomes frozen.
    funders: dict[int, str] = field(default_factory=dict) # Dictionary of Discord user_id: user_name pairs for users actively "reheating" this job.

    # Notification tracking fields to manage one-time alerts for job milestones.
    reporting_channel_id: Optional[int] = None      # Discord channel ID where the job was initially started or main interactions occur;
                                                    # used as the primary channel for sending notifications.
    solve_subscribers: dict[int, str] = field(default_factory=dict)     # user_id: user_name for 'on solve' notifications.
    timeout_subscribers: dict[int, str] = field(default_factory=dict)   # user_id: user_name for 'on timeout' notifications.
    has_solved_since_last_reheat: bool = False      # Flag: True if a 'solve' notification has been sent during the job's current active period.
    timeout_notified_for_current_period: bool = False # Flag: True if a 'timeout' notification has been sent during the job's current active period.


    def is_frozen(self) -> bool:
        """
        Determines if a job is currently "frozen" (inactive) and should not undergo
        further garden maintenance or resource allocation in the background loop.
        A job is considered frozen if its `expiration_time` is in the past, or
        if no expiration time has been set (implying it has not been actively funded).

        Returns:
            bool: True if the job is frozen, False otherwise.
        """
        if self.expiration_time is None:
            return True # A job with no funding is considered frozen by default.
        return time.time() > self.expiration_time


# Global dictionary to store all active and recently managed jobs, keyed by their design ID.
# This data is stored in memory and will reset if the bot process restarts.
jobs: dict[int, Job] = {}


# --- Helper Functions for Design ID Extraction ---

def extract_design_id(input_string: str) -> Optional[int]:
    """
    Extracts a Fantastic Contraption design ID from a given input string.
    The input can be either a direct integer ID or a URL containing 'designId=<number>'.

    Args:
        input_string (str): The string to parse (e.g., "12345" or "https://ft.jtai.dev/?designId=12707044").

    Returns:
        int | None: The extracted design ID as an integer, or None if no valid ID is found.
    """
    # Attempt to convert the input directly to an integer first.
    try:
        return int(input_string)
    except ValueError:
        pass # If not a direct integer, proceed to regex search for a URL.

    # If not a direct integer, try to find a 'designId' parameter in a URL using regex.
    match = re.search(r'designId=(\d+)', input_string)
    if match:
        try:
            return int(match.group(1)) # Extract the digits and convert to an integer.
        except ValueError:
            return None # This should ideally not happen if regex captures only digits, but acts as a safeguard.

    return None # Return None if no valid design ID could be extracted.


def get_or_create_job(design_id: int) -> Job:
    """
    Retrieves an existing `Job` object for a given Fantastic Contraption design ID.
    If no job exists for the specified ID, a new `Job` instance is created and returned.

    Args:
        design_id (int): The unique integer ID of the Fantastic Contraption design.

    Returns:
        Job: The `Job` object associated with the design ID, either existing or newly created.
    """
    if design_id not in jobs:
        print(f"Initializing new job entry for design ID: {design_id}")
        jobs[design_id] = Job() # Create a new, empty `Job` instance.
    else:
        print(f"Retrieving existing job for design ID: {design_id}")
    return jobs[design_id]


# --- Core Reheating (Time Budgeting) Logic ---

async def _reheat_job_logic(ctx: commands.Context, design_id: int, job: Job, user_id: int, user_name: str) -> None:
    """
    Handles the core logic for "reheating" (funding) a job, extending its active time.
    This function enforces per-user active job limits and calculates the job's new
    expiration time based on the number of unique funders using a predefined table.

    Args:
        ctx (commands.Context): The Discord context from which the command was invoked, used for sending messages.
        design_id (int): The ID of the Fantastic Contraption design associated with the job.
        job (Job): The `Job` object that is being reheated.
        user_id (int): The Discord user ID of the person initiating the reheat.
        user_name (str): The Discord display name of the person initiating the reheat.
    """
    # 1. Enforce per-user active job limit:
    # Count how many currently unfrozen jobs this specific user is actively funding.
    current_user_funded_jobs_count = 0
    for _, j_obj in list(jobs.items()): # Iterate over a copy to avoid modification issues during iteration.
        if not j_obj.is_frozen() and user_id in j_obj.funders:
            current_user_funded_jobs_count += 1

    # Check if the user has reached their maximum active job limit.
    # The current job is excluded from this count if the user is already funding it and it's active.
    # This prevents blocking a legitimate reheat of an already funded job.
    if not job.is_frozen() and user_id in job.funders:
        pass # User is already funding this active job, no new limit check needed for *this* job.
    elif current_user_funded_jobs_count >= MAX_ACTIVE_JOBS_PER_USER:
        await ctx.send(
            f"Sorry, **{user_name}**, you are currently actively reheating {current_user_funded_jobs_count} jobs. "
            f"You can only actively reheat up to **{MAX_ACTIVE_JOBS_PER_USER}** jobs simultaneously. "
            f"Please let some of your current jobs expire or use `@BotName status` to check their state before reheating more."
        )
        job.errors.append(f"User {user_name} (ID: {user_id}) exceeded per-user job limit during reheat attempt for design {design_id}.")
        return

    # 2. Add the current user as a funder to this job.
    job.funders[user_id] = user_name

    # 3. Calculate the new expiration time using the `FUNDING_DURATIONS_SECONDS` table.
    # The number of unique funders determines the base duration from the table.
    num_unique_funders = len(job.funders)
    
    # Retrieve the base duration from the table. If `num_unique_funders` exceeds
    # the table's defined tiers, the duration for the last tier is used.
    if num_unique_funders <= len(FUNDING_DURATIONS_SECONDS):
        calculated_duration_base = FUNDING_DURATIONS_SECONDS[num_unique_funders - 1]
    else:
        calculated_duration_base = FUNDING_DURATIONS_SECONDS[-1] # Use the duration of the highest defined tier.
    
    # Apply the absolute maximum active time to ensure no job runs excessively long.
    final_calculated_duration = min(calculated_duration_base, MAX_JOB_ACTIVE_TIME_SECONDS)

    # Determine the start time for extending the expiration:
    # If the job is currently active and not expired, extend from its existing expiration time.
    # If the job is already expired or has no prior expiration (first funding), extend from the current time.
    current_active_until = job.expiration_time if job.expiration_time is not None and job.expiration_time > time.time() else time.time()
    new_expiration_time = current_active_until + final_calculated_duration
    job.expiration_time = new_expiration_time

    # Reset notification flags for the new active period.
    # This ensures that 'solve' or 'timeout' alerts can be triggered again after a job is reheated.
    job.has_solved_since_last_reheat = False
    job.timeout_notified_for_current_period = False

    # 4. Construct a Discord timestamp for user-friendly relative time display.
    # The `<t:TIMESTAMP:R>` format dynamically displays "in 5 minutes," "2 hours ago," etc.
    discord_timestamp_relative = f"<t:{int(new_expiration_time)}:R>"
    
    # Format the list of funder names for display in the Discord message.
    funder_names_display = ", ".join(job.funders.values())

    # Send a confirmation message to the Discord channel.
    await ctx.send(
        f"Job for design ID **{design_id}** has been reheated! 🎉 "
        f"It will now remain active until {discord_timestamp_relative}. "
        f"Currently actively reheated by: {funder_names_display}."
    )
    print(f"Design ID {design_id} reheated by {user_name} (ID: {user_id}). New expiration: {new_expiration_time} ({num_unique_funders} funders)")
    job.errors.clear() # Clear general job-related errors upon a successful reheat operation.


# --- Snapshotting and Notification Helpers ---

async def _perform_snapshot_and_get_links(job: Job, k_to_snapshot: int) -> Tuple[List[str], List[str]]:
    """
    Executes the snapshot logic for a given job:
    1. Authenticates with the Fantastic Contraption servers using a bot account.
    2. Uploads the `k_to_snapshot` best designs from the job's `garden`.
    3. Formats the uploaded design names and descriptions to adhere to FC server character limits.

    Args:
        job (Job): The `Job` object containing the garden and design details required for snapshotting.
        k_to_snapshot (int): The number of top-performing creatures to attempt to upload.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - `List[str]`: URLs (strings) of successfully saved designs on the FC server.
            - `List[str]`: Error messages (strings) for any failures encountered during the snapshot process.
    """
    if not job.garden or not job.garden.creatures:
        return [], ["No garden or creatures available for snapshot."]

    # Attempt to log in to Fantastic Contraption servers using the bot's configured credentials.
    # `auto_login_get_user_id` is a blocking call, so it's executed in a separate thread to prevent blocking the event loop.
    fc_user_id = None
    try:
        fc_user_id = await asyncio.to_thread(auto_login_get_user_id)
    except Exception as e:
        # It's critical NOT to leak sensitive `user_id` or credential details in error messages returned to Discord.
        return [], ["Failed to auto-login to Fantastic Contraption servers for snapshot."]

    if fc_user_id is None: # Additional safeguard if `auto_login_get_user_id` returns None without an explicit exception.
        return [], ["Could not obtain bot user ID for Fantastic Contraption servers during snapshot."]

    saved_links = []
    snapshot_errors = []

    # Ensure we don't attempt to snapshot more creatures than are actually present in the garden.
    creatures_to_snapshot = job.garden.creatures[:k_to_snapshot]

    # Iterate through the best creatures and attempt to save each one to the FC server.
    for rank_idx, creature in enumerate(creatures_to_snapshot):
        try:
            # Construct a unique and descriptive design name, truncated to fit FC's 15-character limit.
            # Example format: 'FC<original_design_id>-<rank>' (e.g., 'FC12345678-1')
            design_name_base = f"FC{job.design_struct.design_id}-{rank_idx+1}"
            if len(design_name_base) > 15:
                design_name_base = design_name_base[:12] + "..." # Truncate and add ellipsis for clarity.

            # Prepare the design description, including score and a "SOLVED!" emoji if applicable.
            # Truncate to fit FC's 50-character limit.
            score_status_text = f"score {creature.best_score}"
            solve_emoji = ""
            if creature.best_score is not None and creature.best_score < 0: # Assuming negative score indicates a solved design.
                score_status_text = f"SOLVED! Score: {creature.best_score}"
                solve_emoji = " 🎉" # Add a celebration emoji for a solved design.

            description_base = f"Based on {job.design_struct.design_id}, {score_status_text}{solve_emoji}"
            if len(description_base) > 50:
                description_base = description_base[:47] + "..." # Truncate and add ellipsis.

            # Call `save_design`, which is a blocking network operation, run in a separate thread.
            saved_design_id = await asyncio.to_thread(
                save_design, creature.design_struct, fc_user_id, name=design_name_base, description=description_base
            )
            # Construct the full URL to the newly saved design on the FC website.
            link = f"https://ft.jtai.dev/?designId={saved_design_id}"
            saved_links.append(f"{rank_idx+1}. {link} (Score: {creature.best_score}{solve_emoji})")
            print(f"Snapshot: {rank_idx+1}. Saved to {link} achieving score of {creature.best_score}")

        except Exception as e:
            # Record specific errors for individual failed save operations.
            snapshot_errors.append(f"Failed to save creature {rank_idx+1}: {e}")
            print(f"Error saving creature {rank_idx+1} for design {job.design_struct.design_id}: {e}")
            # Continue processing other creatures even if one fails.
    
    return saved_links, snapshot_errors


async def _send_dm_notifications(bot_instance: commands.Bot, subscribers_dict: dict[int, str], message_content: str) -> None:
    """
    Sends notification messages to individual subscribers via Direct Message (DM).
    This is used as a fallback mechanism if sending to the primary channel fails.

    Args:
        bot_instance (commands.Bot): The bot's Discord client instance.
        subscribers_dict (dict[int, str]): A dictionary of `user_id: user_name` for the subscribers to be notified.
        message_content (str): The content of the notification message.
    """
    for user_id, user_name in subscribers_dict.items():
        try:
            # Fetch the Discord User object to send a Direct Message.
            user = await bot_instance.fetch_user(user_id)
            if user:
                await user.send(f"Hey {user_name}! {message_content}")
                print(f"Notification DM sent to {user_name} (ID: {user_id}).")
            else:
                print(f"Could not find user {user_name} (ID: {user_id}) to send DM notification.")
        except Exception as e:
            print(f"Failed to send DM notification to {user_name} (ID: {user_id}): {e}")


async def _send_notification(bot_instance: commands.Bot, job: Job, subscribers_dict: dict[int, str], event_type: str, design_id: int) -> None:
    """
    Sends a consolidated notification message to a group of subscribers for a specific event
    (e.g., design solve or job timeout).
    It first attempts to send the message to the job's primary reporting channel (pinging all subscribers),
    and falls back to sending individual DMs if the channel message fails.
    The message includes links to any successful snapshots.
    After successful (or attempted) notification, the relevant subscriber list is cleared
    to ensure one-time notification per event per active period.

    Args:
        bot_instance (commands.Bot): The bot's Discord client instance.
        job (Job): The `Job` object associated with the event that triggered the notification.
        subscribers_dict (dict[int, str]): The dictionary of `user_id: user_name` for subscribers
                                          to this specific event type (e.g., `job.solve_subscribers`).
        event_type (str): The type of event ("solve" or "timeout") for context in the message.
        design_id (int): The ID of the Fantastic Contraption design related to the event.
    """
    if not subscribers_dict:
        print(f"No subscribers for {event_type} event on design {design_id}. Skipping notification and snapshot.")
        return

    print(f"Triggering {event_type} notification for design {design_id} to {len(subscribers_dict)} subscribers.")

    # Perform a snapshot. This happens regardless of whether the notification reaches Discord channels/DMs successfully.
    snapshot_links, snapshot_errors = await _perform_snapshot_and_get_links(job, DEFAULT_SNAPSHOT_K)

    # Build the base notification message content.
    message_content_parts = []
    if event_type == "solve":
        message_content_parts.append(f"✨ DESIGN SOLVED! ✨ Design ID **{design_id}** has found a solution!")
    elif event_type == "timeout":
        message_content_parts.append(f"⏰ JOB TIMEOUT! ⏰ Design ID **{design_id}** has become frozen due to inactivity.")
    
    if snapshot_links:
        message_content_parts.append("\n**Latest Snapshots:**")
        message_content_parts.extend(snapshot_links)
    if snapshot_errors:
        message_content_parts.append("\n**Snapshot Errors:**")
        message_content_parts.extend(snapshot_errors)
    
    base_message_content = "\n".join(message_content_parts)

    # Attempt to send the message to the job's primary reporting channel first.
    if job.reporting_channel_id:
        channel = bot_instance.get_channel(job.reporting_channel_id)
        if channel and isinstance(channel, discord.TextChannel):
            # Construct a string of Discord mentions (pings) for all subscribers.
            all_subscriber_pings = " ".join([f"<@{uid}>" for uid in subscribers_dict.keys()])
            full_channel_message = f"{all_subscriber_pings}\n{base_message_content}"

            try:
                # Attempt to send the message to the channel.
                await channel.send(full_channel_message)
                print(f"Notification successfully sent to channel {channel.id} for design {design_id}.")
            except discord.HTTPException as e:
                # If sending to the channel fails (e.g., due to Discord's message length or ping limits),
                # log the error and fall back to sending individual DMs.
                print(f"Failed to send full notification to channel {channel.id} for design {design_id} (error: {e}). Falling back to DMs.")
                await _send_dm_notifications(bot_instance, subscribers_dict, base_message_content)
            except Exception as e:
                # Catch any other unexpected errors during channel send and fall back to DMs.
                print(f"Unexpected error sending to channel {channel.id} for design {design_id}: {e}. Falling back to DMs.")
                await _send_dm_notifications(bot_instance, subscribers_dict, base_message_content)
        else:
            # If the reporting channel ID is invalid or the bot cannot access it, fall back to DMs.
            print(f"Could not find or access reporting channel {job.reporting_channel_id} for design {design_id}. Falling back to DMs.")
            await _send_dm_notifications(bot_instance, subscribers_dict, base_message_content)
    else:
        # If no reporting channel ID is set for the job, directly send DMs to subscribers.
        print(f"No reporting channel set for design {design_id}. Sending DMs.")
        await _send_dm_notifications(bot_instance, subscribers_dict, base_message_content)

    # After sending notifications (or attempting to), clear the subscriber list for this event.
    # This ensures users are notified only once per event within the current active period.
    subscribers_dict.clear()


# --- Discord Bot Setup ---

# Define the bot's intents. Intents specify which events your bot wants to receive from Discord.
# It's good practice to explicitly define your intents for clarity and future-proofing.
intents = discord.Intents.default()
# The `message_content` intent is required for the bot to read the content of messages,
# which is essential for processing command arguments.
intents.message_content = True

# Initialize the bot with a command prefix and intents.
# The bot will respond to commands only when it is explicitly mentioned (e.g., `@BotName ping`).
bot = commands.Bot(command_prefix=commands.when_mentioned, intents=intents)


# --- Discord Event Handlers ---

@bot.event
async def on_ready():
    """
    Event handler that is called when the bot has successfully connected to Discord.
    This is the ideal place to perform setup tasks and start any continuous background operations.
    """
    print(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    print('------')
    # Start the continuous background loop as a task in the bot's event loop.
    bot.loop.create_task(background_loop())


# --- Discord Commands ---

@bot.command()
async def hello(ctx: commands.Context):
    """
    A simple command that makes the bot respond with a friendly greeting.
    Usage: @BotName hello
    """
    await ctx.send('Hello there!')

@bot.command()
async def ping(ctx: commands.Context):
    """
    Checks the bot's current latency (ping) to the Discord API.
    Usage: @BotName ping
    """
    # Calculate latency in milliseconds.
    latency_ms = round(bot.latency * 1000)
    await ctx.send(f'Pong! {latency_ms}ms')

@bot.command()
async def status(ctx: commands.Context, *, design_input: str):
    """
    Provides a detailed status report for a specific Fantastic Contraption design job.
    If a job for the given design ID does not yet exist, it will be initialized.
    This command also attempts to load the design structure if it hasn't been loaded already.

    Args:
        design_input (str): The design ID (e.g., "12345") or a full Fantastic Contraption URL
                            (e.g., "https://fantasticcontraption.com/original/?designId=12668445").

    Usage Examples:
      `@BotName status 12345`
      `@BotName status https://fantasticcontraption.com/original/?designId=12668445`
    """
    design_id = extract_design_id(design_input)

    if design_id is None:
        await ctx.send(
            f"Sorry, I couldn't understand that design ID. "
            f"Please provide a numeric ID or a valid Fantastic Contraption link like `https://example.com/?designId=12345`."
        )
        return

    job = get_or_create_job(design_id)
    job.errors.clear() # Clear any previous general errors for this job before generating a new status report.

    # Attempt to retrieve and parse the design structure if it's not already loaded for the job.
    if job.design_struct is None:
        print(f"Attempting to load design structure for design ID: {design_id}")
        try:
            # `retrieveDesign` is a blocking I/O operation; it's run in a separate thread to avoid blocking the bot.
            design_dom = await asyncio.to_thread(retrieveDesign, design_id)

            if design_dom is None:
                job.errors.append(f"Failed to retrieve design DOM for ID {design_id}. It might not exist or there was a network issue.")
            else:
                try:
                    job.design_struct = designDomToStruct(design_dom)
                    print(f"Successfully loaded design structure for design ID: {design_id}")
                except Exception as e:
                    job.errors.append(f"Failed to parse design DOM for ID {design_id}: {e}")
        except Exception as e:
            job.errors.append(f"Error during design retrieval for ID {design_id}: {e}")
            print(f"Error retrieving design {design_id}: {e}")

    # Compile various aspects of the job's status.
    config_status = "No JSON configuration set."
    if job.config_json is not None:
        config_status = "JSON configuration is set."

    design_struct_status = "No design structure loaded."
    if job.design_struct is not None:
        design_struct_status = "Design structure is loaded."

    garden_status = "Garden not initialized."
    if job.garden is not None:
        garden_status = "Garden initialized."
    
    # Status indicating whether the job is active or frozen, with a relative timestamp for expiration.
    active_time_status = "Not actively reheated (frozen by default)."
    if job.expiration_time is not None:
        if job.is_frozen():
            active_time_status = f"Frozen (expired {discord.utils.format_dt(discord.Object(int(job.expiration_time)), 'R')})."
        else:
            active_time_status = f"Active until {discord.utils.format_dt(discord.Object(int(job.expiration_time)), 'R')}."

    # Display the names of users currently "reheating" (funding) the job.
    funder_names_display = "None"
    if job.funders:
        funder_names_display = ", ".join(job.funders.values())

    # Status for 'on solve' notification subscriptions.
    solve_alert_status = "No solve subscribers."
    if job.solve_subscribers:
        solve_alert_status = f"{len(job.solve_subscribers)} solve subscribers."
    if job.has_solved_since_last_reheat:
        solve_alert_status += " (Solved this period)"

    # Status for 'on timeout' notification subscriptions.
    timeout_alert_status = "No timeout subscribers."
    if job.timeout_subscribers:
        timeout_alert_status = f"{len(job.timeout_subscribers)} timeout subscribers."
    if job.timeout_notified_for_current_period:
        timeout_alert_status += " (Notified of timeout this period)"


    # Assemble and potentially trim any accumulated error messages to fit within Discord's message limit (2000 characters).
    error_report_lines = []
    if job.errors:
        error_report_lines.append("\n**Recent Errors:**")
        # Estimate the length of the main status message to ensure errors don't push it over the limit.
        estimated_base_response_length = len(
            f"Job for design ID **{design_id}** is active. "
            f"Status: {config_status}. {design_struct_status}. {garden_status}. {active_time_status}. Reheated by: {funder_names_display}."
            f"Alerts: Solve ({solve_alert_status}), Timeout ({timeout_alert_status})"
        ) + len("\n**Recent Errors:**\n") # Account for the error header and newline.

        current_length = estimated_base_response_length

        for i, err in enumerate(job.errors):
            line = f"- {err}"
            # Truncate error messages if adding them would exceed a safe limit (1900 chars) for the total message.
            if current_length + len(line) + (2 if i < len(job.errors) - 1 else 0) > 1900:
                error_report_lines.append("... (more errors omitted)")
                break
            error_report_lines.append(line)
            current_length += len(line) + 1 # Account for the line length plus a newline character.

    # Construct the final, comprehensive response message.
    full_response = (
        f"Job for design ID **{design_id}** is active:\n"
        f"  - **Configuration:** {config_status}\n"
        f"  - **Design Structure:** {design_struct_status}\n"
        f"  - **Garden Status:** {garden_status}\n"
        f"  - **Active Time:** {active_time_status}\n"
        f"  - **Actively Reheated By:** {funder_names_display}\n"
        f"  - **Notifications:**\n"
        f"    - Solve Alerts: {solve_alert_status}\n"
        f"    - Timeout Alerts: {timeout_alert_status}"
    )
    if error_report_lines:
        full_response += "\n".join(error_report_lines)

    await ctx.send(full_response)
    job.errors.clear() # Clear general job-related errors after they have been reported.


@bot.command()
async def set_config(ctx: commands.Context, design_input: str, *, json_content: str = None):
    """
    Sets the JSON configuration for a specific design job.
    The JSON content can be provided directly within the message (preferably wrapped
    in a `json` markdown code block) or uploaded as a `.json` file attachment.

    Args:
        design_input (str): The design ID or URL for which to set the configuration.
        json_content (str, optional): The JSON content as a string, if provided directly.

    Usage Examples:
      `@BotName set_config 12345 ```json\n{\"key\": \"value\"}\n``` `
      `@BotName set_config https://example.com/?designId=12345` (with an attached `.json` file)
    """
    design_id = extract_design_id(design_input)

    if design_id is None:
        await ctx.send(
            f"Sorry, I couldn't understand that design ID. "
            f"Please provide a numeric ID or a valid Fantastic Contraption link like `https://example.com/?designId=12345`."
        )
        return

    job = get_or_create_job(design_id)
    job.errors.clear() # Clear previous errors before attempting to set the configuration.
    parsed_json = None

    # Prioritize checking for a JSON file attachment.
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
                job.errors.append(f"An unexpected error occurred while reading the attachment: {e}")
                await ctx.send(f"An unexpected error occurred while reading the attachment: {e}")
                return
        else:
            await ctx.send("The attached file is not a JSON file (`.json` content type). Please attach a valid JSON file.")
            return
    # If no attachment, check for JSON content directly in the message.
    elif json_content:
        # Attempt to extract JSON from a markdown code block (e.g., ````json\n...\n````).
        json_match = re.search(r'```json\n(.*)```', json_content, re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
        else:
            json_string = json_content # Assume plain JSON string if no markdown block is found.

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
        # Validate the parsed JSON using the external `json_validate` function.
        if json_validate(parsed_json):
            job.config_json = parsed_json
            await ctx.send(f"JSON configuration successfully set for design ID **{design_id}**.")
        else:
            job.errors.append(f"JSON configuration for design ID {design_id} is invalid according to validation rules.")
            await ctx.send(f"JSON configuration for design ID **{design_id}** is invalid according to validation rules. Please check the format.")
    else:
        await ctx.send(f"No valid JSON content was found to set for design ID **{design_id}**.")


@bot.command()
async def get_config(ctx: commands.Context, *, design_input: str):
    """
    Retrieves and outputs the JSON configuration that has been set for a design job.
    If the JSON content is too large to fit in a standard Discord message, it will be sent as a file attachment.

    Args:
        design_input (str): The design ID or URL for which to retrieve the configuration.

    Usage Examples:
      `@BotName get_config 12345`
      `@BotName get_config https://example.com/?designId=12345`
    """
    design_id = extract_design_id(design_input)

    if design_id is None:
        await ctx.send(
            f"Sorry, I couldn't understand that design ID. "
            f"Please provide a numeric ID or a valid Fantastic Contraption link like `https://example.com/?designId=12345`."
        )
        return

    job = get_or_create_job(design_id)
    job.errors.clear() # Clear previous errors before attempting to retrieve config.

    if job.config_json is None:
        await ctx.send(f"No JSON configuration has been set for design ID **{design_id}** yet.")
        return

    json_output_string = json.dumps(job.config_json, indent=2)

    # Discord's message character limit is 2000. Send as a file if the JSON string is too long.
    if len(json_output_string) > 1900: # Using a soft limit (1900 characters) for safety.
        filename = f"design_{design_id}_config.json"
        with open(filename, 'w') as f:
            f.write(json_output_string)
        await ctx.send(
            f"Here is the JSON configuration for design ID **{design_id}**:",
            file=discord.File(filename)
        )
        os.remove(filename) # Clean up the temporary file from the local filesystem.
    else:
        # Send the JSON directly in the message within a markdown code block for easy viewing.
        await ctx.send(f"Here is the JSON config for design ID **{design_id}**:\n```json\n{json_output_string}\n```")


@bot.command()
async def start_job(ctx: commands.Context, *, design_input: str):
    """
    Initializes a new Garden (effectively starting the evolutionary job) for a given design.
    This command requires both the design structure and a valid JSON configuration to be present.
    Upon successful initialization, the job is automatically "reheated" (funded) for the invoking user.

    Args:
        design_input (str): The design ID or URL of the Fantastic Contraption design to start the job for.

    Usage Examples:
      `@BotName start_job 12345`
      `@BotName start_job https://fantasticcontraption.com/?designId=12345`
    """
    design_id = extract_design_id(design_input)

    if design_id is None:
        await ctx.send(
            f"Sorry, I couldn't understand that design ID. "
            f"Please provide a numeric ID or a valid Fantastic Contraption link like `https://example.com/?designId=12345`."
        )
        return

    job = get_or_create_job(design_id)
    job.errors.clear() # Clear previous errors before attempting to start the job.

    # Store the channel ID where this command was invoked. This channel will be used
    # as the primary destination for future notifications related to this job.
    if isinstance(ctx.channel, discord.TextChannel):
        job.reporting_channel_id = ctx.channel.id

    # Check the necessary prerequisites for initializing the garden.
    if job.design_struct is None:
        job.errors.append("Cannot start garden: Design structure not loaded. Please use `@BotName status` to load it first.")
    if job.config_json is None:
        job.errors.append("Cannot start garden: JSON configuration not set. Please use `@BotName set_config` to set it first.")

    # If any prerequisites are missing, report the errors to the user and terminate the command.
    if job.errors:
        error_report = "\n**Errors:**\n" + "\n".join([f"- {err}" for err in job.errors])
        await ctx.send(f"Could not start garden for design ID **{design_id}**. Please fix the listed issues.{error_report}")
        job.errors.clear()
        return

    # If a garden is already initialized and running for this job, treat the command
    # as a request to "reheat" (re-fund) the existing job instead of starting a new one.
    if job.garden is not None:
        await ctx.send(f"Garden for design ID **{design_id}** is already running. Attempting to reheat instead.")
        user_id = ctx.author.id
        user_name = ctx.author.display_name
        await _reheat_job_logic(ctx, design_id, job, user_id, user_name)
        return

    try:
        # Initialize the Garden object. This typically involves creating initial creatures
        # based on the design structure and configuration.
        job.garden = Garden([Creature(job.design_struct)], MAX_GARDEN_SIZE, job.config_json)
        await ctx.send(f"Garden successfully initialized for design ID **{design_id}**!")
        print(f"Garden initialized for design ID {design_id}")

        # Automatically "reheat" (fund) the job for the user who initiated it,
        # granting it an initial period of active time.
        user_id = ctx.author.id
        user_name = ctx.author.display_name
        await _reheat_job_logic(ctx, design_id, job, user_id, user_name)

    except Exception as e:
        job.errors.append(f"Failed to initialize garden for ID {design_id}: {e}")
        await ctx.send(f"Failed to initialize garden for design ID **{design_id}**: {e}")
        print(f"Error initializing garden for design {design_id}: {e}")
    finally:
        job.errors.clear() # Ensure general job errors are cleared after the attempt to initialize.


@bot.command()
async def reheat(ctx: commands.Context, *, design_input: str):
    """
    "Reheats" (funds) an existing design job, extending its active time.
    Multiple users contributing to the same job will extend its time at a superlinear rate,
    up to a predefined maximum duration.
    Users are subject to a limit on how many jobs they can actively reheat simultaneously.

    Args:
        design_input (str): The design ID or URL of the job to reheat.

    Usage Examples:
      `@BotName reheat 12345`
      `@BotName reheat https://fantasticcontraption.com/?designId=12345`
    """
    design_id = extract_design_id(design_input)

    if design_id is None:
        await ctx.send(
            f"Sorry, I couldn't understand that design ID. "
            f"Please provide a numeric ID or a valid Fantastic Contraption link like `https://example.com/?designId=12345`."
        )
        return

    job = get_or_create_job(design_id)
    job.errors.clear() # Clear previous errors before processing the reheat request.

    # A garden must be initialized for a job to be reheated.
    if job.garden is None:
        job.errors.append("Cannot reheat job: Garden not initialized. Please use `@BotName start_job` first.")
        await ctx.send(f"Cannot reheat job for design ID **{design_id}**. Garden not active. Please use `@BotName start_job`.")
        job.errors.clear()
        return

    user_id = ctx.author.id
    user_name = ctx.author.display_name

    # Call the shared reheating logic to apply funding rules and update the job's expiration.
    await _reheat_job_logic(ctx, design_id, job, user_id, user_name)
    job.errors.clear() # Ensure errors from `_reheat_job_logic` are cleared after reporting.


@bot.command()
async def subscribe_solve(ctx: commands.Context, *, design_input: str):
    """
    Subscribes the invoking user to a notification that triggers when this design
    achieves its first "solve" (negative score) within its current active period.

    Args:
        design_input (str): The design ID or URL of the job to subscribe to.

    Usage Examples:
      `@BotName subscribe_solve 12345`
      `@BotName subscribe_solve https://fantasticcontraption.com/?designId=12345`
    """
    design_id = extract_design_id(design_input)
    if design_id is None:
        await ctx.send("Please provide a valid design ID or link.")
        return

    job = get_or_create_job(design_id)
    
    # Subscription is only valid for jobs with an active garden.
    if job.garden is None:
        await ctx.send(f"Cannot subscribe to solve alerts for design ID **{design_id}**: Garden not active. Please use `@BotName start_job` first.")
        return

    user_id = ctx.author.id
    user_name = ctx.author.display_name

    # Add the user to the solve subscribers list.
    job.solve_subscribers[user_id] = user_name
    await ctx.send(f"**{user_name}**, you are now subscribed to solve alerts for design ID **{design_id}**! I'll ping you here (or DM you if I can't) when it solves.")
    print(f"User {user_name} subscribed to solve alerts for design {design_id}")


@bot.command()
async def subscribe_timeout(ctx: commands.Context, *, design_input: str):
    """
    Subscribes the invoking user to a notification that triggers when this design's
    active time runs out and the job becomes frozen.

    Args:
        design_input (str): The design ID or URL of the job to subscribe to.

    Usage Examples:
      `@BotName subscribe_timeout 12345`
      `@BotName subscribe_timeout https://fantasticcontraption.com/?designId=12345`
    """
    design_id = extract_design_id(design_input)
    if design_id is None:
        await ctx.send("Please provide a valid design ID or link.")
        return

    job = get_or_create_job(design_id)

    # Subscription is only valid for jobs with an active garden.
    if job.garden is None:
        await ctx.send(f"Cannot subscribe to timeout alerts for design ID **{design_id}**: Garden not active. Please use `@BotName start_job` first.")
        return

    user_id = ctx.author.id
    user_name = ctx.author.display_name

    # Add the user to the timeout subscribers list.
    job.timeout_subscribers[user_id] = user_name
    await ctx.send(f"**{user_name}**, you are now subscribed to timeout alerts for design ID **{design_id}**! I'll ping you here (or DM you if I can't) when it times out.")
    print(f"User {user_name} subscribed to timeout alerts for design {design_id}")


@bot.command()
async def snapshot(ctx: commands.Context, design_input: str, k: Optional[int] = None):
    """
    Captures a "snapshot" of the current best designs from a job's garden.
    It uploads the top 'k' creatures to the Fantastic Contraption servers
    and returns their public links in a message.

    Args:
        design_input (str): The design ID or URL of the job to snapshot.
        k (Optional[int]): The number of best creatures to snapshot. Defaults to `DEFAULT_SNAPSHOT_K`.

    Usage Examples:
      `@BotName snapshot 12345` (snapshots default number of best designs)
      `@BotName snapshot 12345 5` (snapshots the top 5 best designs)
      `@BotName snapshot https://fantasticcontraption.com/?designId=12345 1`
    """
    design_id = extract_design_id(design_input)

    if design_id is None:
        await ctx.send(
            f"Sorry, I couldn't understand that design ID. "
            f"Please provide a numeric ID or a valid Fantastic Contraption link like `https://example.com/?designId=12345`."
        )
        return

    job = get_or_create_job(design_id)
    job.errors.clear() # Clear general job errors before executing the snapshot command.

    # Snapshotting requires an active garden.
    if job.garden is None:
        job.errors.append("Garden not initialized for this design. Please use `@BotName start_job` first.")
        await ctx.send(f"Cannot snapshot for design ID **{design_id}**. Garden not active. Please use `@BotName start_job`.")
        job.errors.clear()
        return

    # Determine the number of creatures to upload, respecting the `DEFAULT_SNAPSHOT_K` limit.
    upload_best_k = DEFAULT_SNAPSHOT_K
    if k is not None:
        if k > DEFAULT_SNAPSHOT_K:
            await ctx.send(f"Snapshot count cannot exceed {DEFAULT_SNAPSHOT_K}. Using {DEFAULT_SNAPSHOT_K} instead of {k}.")
            upload_best_k = DEFAULT_SNAPSHOT_K
        elif k <= 0:
            await ctx.send("Snapshot count must be positive. No designs will be saved.")
            job.errors.append("Invalid snapshot count provided.")
            job.errors.clear()
            return
        else:
            upload_best_k = k

    # Perform the snapshot operation and collect the resulting links and any errors.
    saved_links_messages, snapshot_errors = await _perform_snapshot_and_get_links(job, upload_best_k)

    # Report the results of the snapshot to the Discord channel.
    if saved_links_messages:
        response_message = f"Successfully snapshotted {len(saved_links_messages)} designs for ID **{design_id}**:"
        response_message += "\n" + "\n".join(saved_links_messages)
        await ctx.send(response_message)
    else:
        if snapshot_errors:
            await ctx.send(f"Failed to snapshot any designs for ID **{design_id}**. Please check previous error messages or the bot's console for details.")
        else:
            await ctx.send(f"No designs were snapshotted for ID **{design_id}**. "
                           f"This might be because no creatures were available in the garden, "
                           f"or the specified `k` value ({upload_best_k}) was too high for the available creatures.")
    
    # Report any lingering snapshot-specific errors that occurred.
    if snapshot_errors:
        error_report_lines = ["\n**Snapshot Errors:**"]
        current_length = len("\n**Snapshot Errors:**\n")
        
        for i, err in enumerate(snapshot_errors):
            line = f"- {err}"
            # Ensure error messages themselves don't exceed Discord's message limits.
            if current_length + len(line) + (2 if i < len(snapshot_errors) - 1 else 0) > 1900:
                error_report_lines.append("... (more errors omitted)")
                break
            error_report_lines.append(line)
            current_length += len(line) + 1
        await ctx.send("".join(error_report_lines))
    
    job.errors.clear() # Clear general job errors after reporting, distinct from `snapshot_errors`.


# --- Background Task for Job Maintenance ---

async def background_loop():
    """
    A continuous background task that runs while the bot is active.
    This loop performs essential garden maintenance for all active jobs,
    dynamically distributing compute thread resources based on system congestion
    and a fair round-robin approach. It also actively monitors jobs to trigger
    solve and timeout notifications.
    """
    await bot.wait_until_ready() # Ensure the bot is fully connected to Discord before starting the loop.

    global _background_loop_iteration_counter # Declare global to modify the counter.

    while not bot.is_closed():
        _background_loop_iteration_counter += 1
        print(f"Background loop running (Iteration {_background_loop_iteration_counter}) - performing garden maintenance and checking notifications...")
        
        # 1. Identify eligible jobs (those with an active garden and not currently frozen)
        # Simultaneously, check for and trigger timeout notifications for jobs that have become frozen.
        eligible_jobs_for_compute = []
        for design_id, job in list(jobs.items()): # Iterate over a copy to safely modify `jobs` dict if needed.
            # Check for Timeout Notification:
            # Trigger if the job is now frozen, had an expiration time, and hasn't been notified for this period.
            if job.garden is not None and job.is_frozen() and job.expiration_time is not None and not job.timeout_notified_for_current_period:
                if job.timeout_subscribers:
                    print(f"Triggering timeout notification for design ID: {design_id}")
                    # `_send_notification` handles sending to channel or DMs.
                    await _send_notification(bot, job, job.timeout_subscribers, "timeout", design_id)
                job.timeout_notified_for_current_period = True # Mark as notified to prevent duplicate alerts for this period.

            # Add the job to the list for compute resource allocation ONLY if it's not frozen.
            if job.garden is not None and not job.is_frozen():
                eligible_jobs_for_compute.append((design_id, job))

        num_eligible_jobs = len(eligible_jobs_for_compute)
        threads_to_allocate_per_job: dict[int, int] = {} # Map design_id to allocated thread count for this iteration.

        # Dynamic Thread Allocation Strategy:
        if num_eligible_jobs == 0:
            print("No eligible gardens to maintain.")
        elif num_eligible_jobs == 1:
            # If only one job is active, give it all available global threads.
            design_id, job = eligible_jobs_for_compute[0]
            threads_to_allocate_per_job[design_id] = MAX_THREADS_GLOBAL
            job.last_turn_allocated = _background_loop_iteration_counter # Update its turn timestamp.
            print(f"Single garden {design_id} gets all {MAX_THREADS_GLOBAL} threads.")
        else:
            # For multiple jobs, sort them by `last_turn_allocated` to implement round-robin fairness.
            eligible_jobs_for_compute.sort(key=lambda item: item[1].last_turn_allocated)

            # Distribute threads as evenly as possible, giving remainder to longest-waiting jobs.
            base_threads_per_job = MAX_THREADS_GLOBAL // num_eligible_jobs
            remainder_threads = MAX_THREADS_GLOBAL % num_eligible_jobs

            for i, (design_id, job) in enumerate(eligible_jobs_for_compute):
                allocated_threads = base_threads_per_job
                # Distribute the remainder threads one by one to the jobs that have waited longest.
                if remainder_threads > 0 and i < remainder_threads:
                    allocated_threads += 1
                    remainder_threads -= 1
                
                threads_to_allocate_per_job[design_id] = allocated_threads
                # Update the `last_turn_allocated` timestamp only for jobs that actually receive threads.
                if allocated_threads > 0:
                    job.last_turn_allocated = _background_loop_iteration_counter
                print(f"Allocating {allocated_threads} threads to garden {design_id}.")
        
        # 3. Perform maintenance for all eligible gardens based on their allocated threads.
        # Iterate over all jobs again (using a copy) to ensure all jobs, including those not receiving
        # new threads this turn, are considered for checks and logging.
        for design_id, job in list(jobs.items()):
            # Re-check `is_frozen()` here, as a job's status might have changed (e.g., just timed out)
            # between the initial list creation and this processing step.
            if job.garden is not None and not job.is_frozen():
                try:
                    # Get the number of threads allocated to this specific job for the current iteration.
                    allocated_threads_for_this_job = threads_to_allocate_per_job.get(design_id, 0)
                    
                    # Perform a `checkup` on the garden to get its current status.
                    garden_status = job.garden.checkup()
                    print(f"Garden {design_id} Checkup: Active Threads={garden_status.num_active_threads}, Generation={garden_status.generation}")

                    # Check for Solve Notification:
                    # Trigger if the best creature has a negative score (indicating a solve) AND
                    # a solve notification hasn't already been sent for the current active period.
                    if job.garden.creatures and job.garden.creatures[0].best_score is not None and \
                       job.garden.creatures[0].best_score < 0 and not job.has_solved_since_last_reheat:
                        if job.solve_subscribers:
                            print(f"Triggering solve notification for design ID: {design_id}")
                            await _send_notification(bot, job, job.solve_subscribers, "solve", design_id)
                        job.has_solved_since_last_reheat = True # Mark as notified to prevent duplicate alerts.

                    # Perform garden evolution. Note: `evolve` is called with `MAX_THREADS_GLOBAL`
                    # as per previous instructions, assuming its internal logic handles parallelism
                    # rather than directly using the allocated thread count here for *new* work.
                    job.garden.evolve(MAX_THREADS_GLOBAL, RETAIN_BEST_K_EVOLUTION)
                    print(f"Garden {design_id} Evolved.")

                    # Calculate how many *new* threads the garden is allowed to start this turn.
                    # This ensures we don't exceed its allocated share and handles cases where
                    # a job might temporarily have more active threads than its current allocation.
                    new_threads_allowed = max(0, allocated_threads_for_this_job - garden_status.num_active_threads)

                    if new_threads_allowed > 0:
                        job.garden.start(new_threads_allowed)
                        print(f"Garden {design_id} Started {new_threads_allowed} new threads.")
                    else:
                        print(f"Garden {design_id} has enough active threads ({garden_status.num_active_threads}) or "
                              f"was allocated {allocated_threads_for_this_job} new threads (no new threads started).")
                        # Existing threads will naturally terminate over time, bringing the job's
                        # active thread count down to its current allocation.

                except Exception as e:
                    job.errors.append(f"Error during background garden maintenance for ID {design_id}: {e}")
                    print(f"Error maintaining garden {design_id}: {e}")
            elif job.garden is None:
                # This log message indicates jobs that exist but haven't had their garden initialized yet.
                print(f"Design ID {design_id} has no active garden to maintain.")
            # Jobs that are `is_frozen()` are handled at the top of the loop for timeout notifications
            # and are then skipped for further compute, so no explicit `elif` for frozen here.

        # Introduce a small delay to avoid a busy-loop, allowing other asyncio tasks to run.
        # This interval can be tuned based on desired responsiveness vs. compute intensity.
        await asyncio.sleep(0.1) # Sleep for 100 milliseconds.


# --- Main Application Entry Points ---

async def main():
    """
    The main asynchronous function to initialize and run the Discord bot.
    Retrieves the Discord bot token securely from the 'DISCORD_BOT_TOKEN'
    environment variable and starts the bot client.
    """
    bot_token = os.environ.get('DISCORD_BOT_TOKEN') # Retrieve the token directly from the environment.

    if bot_token: # Check if the token was successfully retrieved.
        try:
            await bot.start(bot_token) # Start the Discord bot.
        except discord.LoginFailure:
            print("Error: Invalid bot token was provided. Please ensure your 'DISCORD_BOT_TOKEN' environment variable is correct.")
        except Exception as e:
            print(f"An unexpected error occurred while running the bot: {e}")
    else:
        # If the bot token is missing, print an error and exit the script.
        print("Error: Discord bot token is missing. Please set the 'DISCORD_BOT_TOKEN' environment variable.")
        sys.exit(1) # Exit with an error code.

def program_discord(args):
    """
    External entry point for integrations that need to run the Discord bot.
    This function currently serves as a wrapper to call the asynchronous `main` function.

    Args:
        args: Command-line arguments or other configuration passed from an external caller.
    """
    asyncio.run(main()) # Run the main asynchronous bot function.

# Standard Python entry point for direct script execution.
if __name__ == "__main__":
    # When the script is run directly, call the main asynchronous function.
    asyncio.run(main())
