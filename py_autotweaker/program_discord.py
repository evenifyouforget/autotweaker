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

Requires Python 3.9 or higher.
"""

# --- Standard Library Imports ---
import os
import asyncio
import sys
import re
import json
import time
import datetime # Import datetime for timestamp conversion
import subprocess # For running Git commands to get version info
import logging    # Import the logging module
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Any

# --- Third-Party Library Imports ---
import discord
from discord.ext import commands
# Assuming jsonschema is installed and ValidationError is directly importable from it.
from jsonschema import ValidationError 

# --- Local Module Imports (Assumed to be in the same package/directory) ---
# These modules are essential for the bot's functionality:
#   - get_design: Handles fetching design data from Fantastic Contraption servers.
#   - save_design: Manages uploading evolved designs back to FC servers.
#   - auto_login: Provides a bot account login to FC for server interactions.
#   - job_struct: Defines data structures like Creature, Garden, and GardenStatus.
#   - json_validate: Contains logic for validating job configuration JSON.
#   - performance: Provides system performance-related utilities, e.e.g., thread count.
from get_design import retrieveDesign, designDomToStruct, FCDesignStruct
from save_design import save_design
from .auto_login import auto_login_get_user_id
# Updated Garden import to reflect the user's provided structure
from .job_struct import Creature, Garden, GardenStatus
# This import is conceptual; in a real scenario, json_validate would need to be updated
# to return ValidationError details if it uses jsonschema.
from .json_validate import json_validate 
from .performance import get_thread_count # Nontrivial library function for available threads

# --- Global Constants & Configuration ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
_bot_start_time = time.time() # Timestamp when the bot started, used for uptime calculation.


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
    expiration_time: Optional[float] = None         # Unix timestamp when the job's active time expires and it becomes frozen (inactive).
                                                    # If None, the job has never been reheated and is considered frozen.
    funders: dict[int, str] = field(default_factory=dict) # Dictionary of Discord user_id: user_name pairs for users actively "reheating" this job.

    # Notification tracking fields to manage one-time alerts for job milestones.
    reporting_channel_id: Optional[int] = None      # Discord channel ID where the job was initially started or main interactions occur;
                                                    # used as the primary channel for sending notifications.
    solve_subscribers: dict[int, str] = field(default_factory=dict)     # user_id: user_name for 'on solve' notifications.
    timeout_subscribers: dict[int, str] = field(default_factory=dict)   # user_id: user_name for 'on timeout' notifications.
    has_solved_since_last_reheat: bool = False      # Flag: True if a 'solve' notification has been sent during the job's current active period (since last reheat).
                                                    # Resets to False upon a successful 'reheat' operation.
    timeout_notified_for_current_period: bool = False # Flag: True if a 'timeout' notification has been sent during the job's current active period (since last reheat).
                                                    # Resets to False upon a successful 'reheat' operation.


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
        input_string (str): The string to parse (e.g., "12345" or "[https://ft.jtai.dev/?designId=12707044](https://ft.jtai.dev/?designId=12707044)").

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


async def get_or_create_job(design_id: int) -> Job:
    """
    Retrieves an existing `Job` object for a given Fantastic Contraption design ID.
    If no job exists for the specified ID, a new `Job` instance is created and returned.
    This function also ensures that the `design_struct` is loaded for the job.

    Args:
        design_id (int): The unique integer ID of the Fantastic Contraption design.

    Returns:
        Job: The `Job` object associated with the design ID, either existing or newly created.
    """
    if design_id not in jobs:
        logger.info(f"Initializing new job entry for design ID: {design_id}")
        jobs[design_id] = Job() # Create a new, empty `Job` instance.
    else:
        logger.info(f"Retrieving existing job for design ID: {design_id}")
    
    job = jobs[design_id]
    
    # Proactively attempt to retrieve and parse the design structure if it's not already loaded.
    if job.design_struct is None:
        logger.info(f"Attempting to load design structure for design ID: {design_id}")
        try:
            # `retrieveDesign` is a blocking I/O operation; it's run in a separate thread to avoid blocking the bot.
            design_dom = await asyncio.to_thread(retrieveDesign, design_id)

            if design_dom is None:
                job.errors.append(f"Failed to retrieve design DOM for ID {design_id}. It might not exist or there was a network issue.")
                logger.warning(f"Failed to retrieve design DOM for ID {design_id}. It might not exist or there was a network issue.")
            else:
                try:
                    job.design_struct = designDomToStruct(design_dom)
                    logger.info(f"Successfully loaded design structure for design ID: {design_id}")
                except Exception as e:
                    job.errors.append(f"Failed to parse design DOM for ID {design_id}: {e}")
                    logger.error(f"Failed to parse design DOM for ID {design_id}: {e}")
        except Exception as e:
            job.errors.append(f"Error during design retrieval for ID {design_id}: {e}")
            logger.error(f"Error during design retrieval for ID {design_id}: {e}")

    return job


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
    # Count how many currently unfrozen jobs this specific user is actively reheating.
    current_user_reheating_jobs_count = 0
    for _, j_obj in list(jobs.items()): # Iterate over a copy to avoid modification issues during iteration.
        if not j_obj.is_frozen() and user_id in j_obj.funders:
            current_user_reheating_jobs_count += 1

    # Check if the user has reached their maximum active job limit.
    # The current job is excluded from this count if the user is already reheating it and it's active.
    # This prevents blocking a legitimate reheat of an an already reheated job.
    if not job.is_frozen() and user_id in job.funders:
         pass # User is already reheating this active job, no new limit check needed for *this* job.
    elif current_user_reheating_jobs_count >= MAX_ACTIVE_JOBS_PER_USER:
        await ctx.send(
            f"❌ Sorry, **{user_name}**, you are currently actively reheating {current_user_reheating_jobs_count} jobs. "
            f"You can only actively reheat up to **{MAX_ACTIVE_JOBS_PER_USER}** jobs simultaneously. "
            f"Please let some of your current jobs expire or use `@BotName status` to check their state before reheating more."
        )
        job.errors.append(f"User {user_name} (ID: {user_id}) exceeded per-user job limit during reheat attempt for design {design_id}.")
        logger.warning(f"User {user_name} (ID: {user_id}) exceeded per-user job limit during reheat attempt for design {design_id}.")
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
    # IMPORTANT FIX: Reheat now always refreshes the timer from the current time.
    new_expiration_time = time.time() + final_calculated_duration
    job.expiration_time = new_expiration_time

    # Reset notification flags for the new active period.
    # This ensures that 'solve' or 'timeout' alerts can be triggered again after a job is reheated.
    job.has_solved_since_last_reheat = False
    job.timeout_notified_for_current_period = False

    # 4. Construct a Discord timestamp for user-friendly relative time display.
    # Convert the float timestamp to a datetime object for discord.utils.format_dt.
    new_expiration_datetime = datetime.datetime.fromtimestamp(new_expiration_time)
    discord_timestamp_relative = discord.utils.format_dt(new_expiration_datetime, 'R')
    
    # Format the list of funder names for display in the Discord message.
    funder_names_display = ", ".join(job.funders.values())

    # Send a confirmation message to the Discord channel.
    await ctx.send(
        f"✅ Job for design ID **{design_id}** has been reheated! 🎉 "
        f"It will now remain active until {discord_timestamp_relative}. "
        f"Currently actively reheated by: {funder_names_display}."
    )
    logger.info(f"Design ID {design_id} reheated by {user_name} (ID: {user_id}). New expiration: {new_expiration_time} ({num_unique_funders} funders)")
    job.errors.clear() # Clear general job-related errors upon a successful reheat operation.


# --- Snapshotting and Notification Helpers ---

async def _perform_snapshot_and_get_links(design_id: int, job: Job, k_to_snapshot: int) -> Tuple[List[str], List[str]]:
    """
    Executes the snapshot logic for a given job:
    1. Authenticates with the Fantastic Contraption servers using a bot account.
    2. Uploads the `k_to_snapshot` best designs from the job's `garden`.
    3. Formats the uploaded design names and descriptions to adhere to FC server character limits.

    Args:
        design_id (int): The ID of the Fantastic Contraption design associated with the job.
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
        logger.error(f"Failed to auto-login to Fantastic Contraption servers for snapshot: {e}")
        return [], ["Failed to auto-login to Fantastic Contraption servers for snapshot."]

    if fc_user_id is None: # Additional safeguard if `auto_login_get_user_id` returns None without an explicit exception.
        logger.error("Could not obtain bot user ID for Fantastic Contraption servers during snapshot.")
        return [], ["Could not obtain bot user ID for Fantastic Contraption servers during snapshot."]

    saved_links = []
    snapshot_errors = []

    # Ensure we don't attempt to snapshot more creatures than are actually present in the garden.
    creatures_to_snapshot = job.garden.creatures[:k_to_snapshot]

    # Iterate through the best creatures and attempt to save each one to the FC server.
    for rank_idx, creature in enumerate(creatures_to_snapshot):
        try:
            # Corrected: Use job.design_struct.name (if available) or the design_id passed to the function
            design_name_base_identifier = job.design_struct.name if job.design_struct and job.design_struct.name else str(design_id)
            design_name_base = f"FC{design_name_base_identifier[:8]}-{rank_idx+1}" # Truncate for display in name

            # Prepare the design description, including score and a "SOLVED!" text if applicable.
            # Emojis are *not* allowed here as per user's request due to old XML parser issues.
            # Truncate to fit FC's 50-character limit.
            score_status_text = f"score {creature.best_score}"
            if creature.best_score is not None and creature.best_score < 0: # Assuming negative score indicates a solved design.
                score_status_text = f"SOLVED! Score: {creature.best_score}"

            # Corrected: Refer to the design by the design_id passed to the function
            description_base = f"Based on job {design_id}, {score_status_text}"
            if len(description_base) > 50:
                description_base = description_base[:47] + "..." # Truncate and add ellipsis.

            # Call `save_design`, which is a blocking network operation, run in a separate thread.
            saved_design_id = await asyncio.to_thread(
                save_design, creature.design_struct, fc_user_id, name=design_name_base, description=description_base
            )
            # Construct the full URL to the newly saved design on the FC website.
            # FIX: Simplified to direct URL output. Discord typically auto-embeds these.
            link = f"https://ft.jtai.dev/?designId={saved_design_id}"
            # Display emoji in Discord message, but not in the saved design's description.
            display_solve_emoji = " 🎉" if creature.best_score is not None and creature.best_score < 0 else ""
            saved_links.append(f"{rank_idx+1}. {link} (Score: {creature.best_score}{display_solve_emoji})")
            logger.info(f"Snapshot: {rank_idx+1}. Saved to {link} achieving score of {creature.best_score}")

        except Exception as e:
            # Record specific errors for individual failed save operations.
            snapshot_errors.append(f"Failed to save creature {rank_idx+1}: {e}")
            logger.error(f"Error saving creature {rank_idx+1} for design {design_id}: {e}")
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
                logger.info(f"Notification DM sent to {user_name} (ID: {user_id}).")
            else:
                logger.warning(f"Could not find user {user_name} (ID: {user_id}) to send DM notification.")
        except Exception as e:
            logger.error(f"Failed to send DM notification to {user_name} (ID: {user_id}): {e}")


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
        logger.info(f"No subscribers for {event_type} event on design {design_id}. Skipping notification and snapshot.")
        return

    logger.info(f"Triggering {event_type} notification for design {design_id} to {len(subscribers_dict)} subscribers.")

    # Perform a snapshot. This happens regardless of whether the notification reaches Discord channels/DMs successfully.
    # Corrected: Pass design_id to _perform_snapshot_and_get_links
    snapshot_links, snapshot_errors = await _perform_snapshot_and_get_links(design_id, job, DEFAULT_SNAPSHOT_K)

    # Build the base notification message content.
    message_content_parts = []
    if event_type == "solve":
        message_content_parts.append(f"✨ DESIGN SOLVED! ✨ Design ID **{design_id}** has found a solution!")
    elif event_type == "timeout":
        message_content_parts.append(f"⏰ JOB TIMEOUT! ⏰ Design ID **{design_id}** has become frozen due to inactivity.")
    
    if snapshot_links:
        # FIX: Message adjusted to indicate "top contenders" and clarify the snapshot.
        message_content_parts.append(f"\n**Top {DEFAULT_SNAPSHOT_K} Contender Snapshots:** (from a sorted garden)")
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
                logger.info(f"Notification successfully sent to channel {channel.id} for design {design_id}.")
            except discord.HTTPException as e:
                # If sending to the channel fails (e.g., due to Discord's message length or ping limits),
                # log the error and fall back to sending individual DMs.
                logger.error(f"Failed to send full notification to channel {channel.id} for design {design_id} (error: {e}). Falling back to DMs.")
                await _send_dm_notifications(bot_instance, subscribers_dict, base_message_content)
            except Exception as e:
                # Catch any other unexpected errors during channel send and fall back to DMs.
                logger.error(f"Unexpected error sending to channel {channel.id} for design {design_id}: {e}. Falling back to DMs.")
                await _send_dm_notifications(bot_instance, subscribers_dict, base_message_content)
        else:
            # If the reporting channel ID is invalid or the bot cannot access it, fall back to DMs.
            logger.warning(f"Could not find or access reporting channel {job.reporting_channel_id} for design {design_id}. Falling back to DMs.")
            await _send_dm_notifications(bot_instance, subscribers_dict, base_message_content)
    else:
        # If no reporting channel ID is set for the job, directly send DMs to subscribers.
        logger.info(f"No reporting channel set for design {design_id}. Sending DMs.")
        await _send_dm_notifications(bot_instance, subscribers_dict, base_message_content)

    # After sending notifications (or attempting to), clear the subscriber list for this event.
    # This ensures users are notified only once per event within the current active period.
    subscribers_dict.clear()


# --- Git Information Helper ---
async def get_git_info() -> dict:
    """
    Asynchronously retrieves Git information about the current repository.
    This includes the current branch, whether there are local changes,
    the number of commits at HEAD, and the current HEAD SHA.

    Returns:
        dict: A dictionary containing 'branch', 'dirty', 'commits_at_head', and 'head_sha'.
              Values will be 'N/A' for any piece of information that cannot be retrieved
              (e.e.g., Git not installed, not a Git repository).
    """
    info = {
        "branch": "N/A",
        "dirty": "N/A",
        "commits_at_head": "N/A",
        "head_sha": "N/A"
    }

    try:
        # Helper to run a shell command and capture output
        async def run_cmd(cmd: str) -> str:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                return stdout.decode().strip()
            else:
                # Log stderr but don't raise, allowing partial info
                logger.warning(f"Git command error for '{cmd}': {stderr.decode().strip()}")
                return "N/A"

        info["branch"] = await run_cmd("git rev-parse --abbrev-ref HEAD")
        
        # Check for dirty status: git status --porcelain returns output if there are uncommitted changes
        dirty_output = await run_cmd("git status --porcelain")
        info["dirty"] = "Yes" if dirty_output and dirty_output != "N/A" else "No" # Also handle "N/A" from run_cmd

        info["commits_at_head"] = await run_cmd("git rev-list --count HEAD")
        info["head_sha"] = await run_cmd("git rev-parse HEAD")

    except FileNotFoundError:
        logger.error("Git command not found. Please ensure Git is installed and in your PATH for version info.")
        # All info remains 'N/A' due to initial dictionary values
    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed during info retrieval: {e}")
        # All info remains 'N/A'
    except Exception as e:
        logger.error(f"An unexpected error occurred while getting Git info: {e}")

    return info


# --- Discord Bot Setup ---

# Define the bot's intents. Intents specifies which events your bot wants to receive from Discord.
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
    logger.info(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    logger.info('------')
    # Start the continuous background loop as a task in the bot's event loop.
    bot.loop.create_task(background_loop())

@bot.event
async def on_message(message: discord.Message):
    """
    Event handler that is called whenever a message is sent in any channel the bot can see.
    This is used to handle bare mentions of the bot to display the help message.
    """
    # Ignore messages from the bot itself to prevent infinite loops.
    if message.author == bot.user:
        return

    # Check if the message is a direct mention of the bot AND has no other content after the mention.
    # We use a regex to ensure only the mention itself (possibly with trailing whitespace) is present.
    mention_pattern = re.compile(rf"<@!?{bot.user.id}>\s*$")
    if mention_pattern.match(message.content):
        # Create a CommandContext from the message to use `send_help`.
        ctx = await bot.get_context(message)
        logger.info(f"Bot mentioned by {message.author} with no command. Sending help message.")
        await ctx.send("Hello! 👋 How can I help you evolve fantastic contraptions? Try one of these commands:")
        await ctx.send_help() # This will send the default help message
        return # Prevent further command processing for bare mentions

    # For all other messages, let the bot's command processor handle them.
    await bot.process_commands(message)

@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    """
    Global error handler for bot commands.
    This catches specific types of errors and provides user-friendly feedback.
    """
    if isinstance(error, commands.MissingRequiredArgument):
        # Handle cases where a required argument is missing for a command.
        logger.warning(f"MissingRequiredArgument error for command '{ctx.command.name}' by {ctx.author}: {error}")
        await ctx.send(
            f"❌ Oops! You're missing a required piece of information for that command. "
            f"It looks like you forgot to provide the **`{error.param.name}`** argument. "
            f"Try `@BotName help {ctx.command.name}` for more details on how to use it."
        )
    elif isinstance(error, commands.BadArgument):
        # Handle cases where an argument provided is of the wrong type or format.
        logger.warning(f"BadArgument error for command '{ctx.command.name}' by {ctx.author}: {error}")
        await ctx.send(
            f"❌ Hmm, I had trouble understanding one of your arguments: **`{error}`**. "
            f"Please double-check the format or type of information you're providing. "
            f"You can use `@BotName help {ctx.command.name}` for usage examples."
        )
    elif isinstance(error, commands.CommandNotFound):
        # Handle cases where the command doesn't exist.
        logger.info(f"CommandNotFound error for message '{ctx.message.content}' by {ctx.author}: {error}")
        await ctx.send(
            f"❓ I don't recognize that command! 🤔 Try `@BotName help` to see a list of commands I know."
        )
    elif isinstance(error, commands.CommandOnCooldown):
        # Handle commands that are on cooldown.
        logger.warning(f"CommandOnCooldown error for command '{ctx.command.name}' by {ctx.author}: {error}")
        await ctx.send(f"⏳ Whoa there, **{ctx.author.display_name}**! That command is on cooldown. Please try again in **{error.retry_after:.1f} seconds**.")
    elif isinstance(error, commands.NoPrivateMessage):
        # Handle commands that cannot be used in DMs.
        logger.warning(f"NoPrivateMessage error for command '{ctx.command.name}' by {ctx.author}: {error}")
        await ctx.send("🚫 This command can only be used in a server channel, not in a direct message.")
    else:
        # For any other unhandled errors, log them and provide a generic error message to the user.
        logger.exception(f"Unhandled command error in command '{ctx.command.name}' by {ctx.author}: {error}")
        await ctx.send(f"❌ An unexpected error occurred while trying to run that command: `{error}`. "
                       "My apologies! The developers have been notified.")


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
                            (e.g., "[https://ft.jtai.dev/?designId=12668445](https://ft.jtai.dev/?designId=12668445)").

    Usage Examples:
      `@BotName status 12345`
      `@BotName status https://fantasticcontraption.com/original/?designId=12668445`
    """
    design_id = extract_design_id(design_input)

    if design_id is None:
        await ctx.send(
            f"❌ Sorry, I couldn't understand that design ID. "
            f"Please provide a numeric ID or a valid Fantastic Contraption link like `https://example.com/?designId=12345`."
        )
        return

    job = await get_or_create_job(design_id) # Call get_or_create_job (now async)
    job.errors.clear() # Clear any previous general errors for this job before generating a new status report.

    # The design_struct loading is now handled by get_or_create_job.
    # We only need to check for its presence and any errors *after* the call.
    if job.design_struct is None:
        error_message = "\n**Errors during design retrieval/parsing:**\n" + "\n".join([f"- {err}" for err in job.errors])
        await ctx.send(f"❌ Could not retrieve design structure for design ID **{design_id}**. "
                       f"Please ensure the ID is correct and the design exists. {error_message}")
        job.errors.clear()
        return


    # Compile various aspects of the job's status.
    config_status = "No JSON configuration set. ⚙️"
    if job.config_json is not None:
        config_status = "JSON configuration is set. ✅"

    design_struct_status = "No design structure loaded."
    if job.design_struct is not None:
        design_struct_status = "Design structure is loaded. ✅"

    garden_status_text = "Not initialized. 🚫"
    if job.garden is not None:
        best_score_display = "N/A"
        if job.garden.creatures and job.garden.creatures[0].best_score is not None:
            best_score_display = f"{job.garden.creatures[0].best_score:.2f}" # Format to 2 decimal places
        garden_status_text = f"Initialized. 🌱 Creatures Processed: {job.garden.num_kills}, Best Score: {best_score_display}"
    
    # Status indicating whether the job is active or frozen, with a relative timestamp for expiration.
    active_time_status = "Not actively reheated (frozen by default). 🧊"
    if job.expiration_time is not None:
        # Convert float timestamp to datetime object for discord.utils.format_dt
        expiration_dt = datetime.datetime.fromtimestamp(job.expiration_time)
        if job.is_frozen():
            active_time_status = f"Frozen (expired {discord.utils.format_dt(expiration_dt, 'R')}). ❄️"
        else:
            active_time_status = f"Active until {discord.utils.format_dt(expiration_dt, 'R')}. 🔥"

    # Display the names of users currently "reheating" (funding) the job.
    funder_names_display = "None"
    if job.funders:
        funder_names_display = ", ".join(job.funders.values())

    # Status for 'on solve' notification subscriptions.
    solve_alert_status = "No solve subscribers. 🔔"
    if job.solve_subscribers:
        solve_alert_status = f"{len(job.solve_subscribers)} solve subscribers. 🔔"
    if job.has_solved_since_last_reheat:
        solve_alert_status += " (Solved this period) ✨"

    # Status for 'on timeout' notification subscriptions.
    timeout_alert_status = "No timeout subscribers. ⏰"
    if job.timeout_subscribers:
        timeout_alert_status = f"{len(job.timeout_subscribers)} timeout subscribers. ⏰"
    if job.timeout_notified_for_current_period:
        timeout_alert_status += " (Notified of timeout this period) 🚨"


    # Assemble and potentially trim any accumulated error messages to fit within Discord's message limit (2000 characters).
    error_report_lines = []
    if job.errors:
        error_report_lines.append("\n**Recent Errors:** ⚠️")
        # Estimate the length of the main status message to ensure errors don't push it over the limit.
        estimated_base_response_length = len(
            f"Job for design ID **{design_id}** details:\n"
            f"  - **Configuration:** {config_status}\n"
            f"  - **Design Structure:** {design_struct_status}\n"
            f"  - **Garden Status:** {garden_status_text}\n"
            f"  - **Active Time:** {active_time_status}\n"
            f"  - **Actively Reheated By:** {funder_names_display}\n"
            f"  - **Notifications:**\n"
            f"    - Solve Alerts: {solve_alert_status}\n"
            f"    - Timeout Alerts: {timeout_alert_status}"
        ) + len("\n**Recent Errors:** ⚠️\n") # Account for the error header and newline.

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
        f"Job for design ID **{design_id}** details:\n"
        f"  - **Configuration:** {config_status}\n"
        f"  - **Design Structure:** {design_struct_status}\n"
        f"  - **Garden Status:** {garden_status_text}\n"
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
    Once a job has started (its garden is initialized), its configuration cannot be changed.

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
            f"❌ Sorry, I couldn't understand that design ID. "
            f"Please provide a numeric ID or a valid Fantastic Contraption link like `https://example.com/?designId=12345`."
        )
        return

    job = await get_or_create_job(design_id) # Call get_or_create_job (now async)
    job.errors.clear() # Clear previous errors before attempting to set the configuration.
    
    # Check if the job's garden is already initialized. If so, prevent configuration changes.
    if job.garden is not None:
        await ctx.send(f"❌ The JSON configuration for design ID **{design_id}** cannot be changed because the job has already started. "
                       f"Configuration can only be set before a job begins.")
        logger.warning(f"Attempted to change JSON config for active job {design_id} by {ctx.author.display_name}. Blocked.")
        return

    parsed_json = None

    # Prioritize checking for a JSON file attachment.
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
        # Be more permissive with content types for JSON files.
        # Accept 'application/json' or 'text/plain' if the filename ends with '.json'.
        if attachment.content_type == 'application/json' or \
           (attachment.content_type == 'text/plain' and attachment.filename.lower().endswith('.json')):
            try:
                json_bytes = await attachment.read()
                parsed_json = json.loads(json_bytes.decode('utf-8'))
                logger.info(f"Parsed JSON from attachment for design ID {design_id}")
            except json.JSONDecodeError as e:
                job.errors.append(f"Failed to parse JSON from attachment: Invalid JSON syntax: {e}")
                logger.error(f"Failed to parse JSON from attachment: {e}")
                await ctx.send(
                    f"❌ Failed to parse JSON from attachment for design ID **{design_id}**: "
                    f"**Invalid JSON syntax.** Please ensure the file contains well-formed JSON. Error: {e}"
                )
                return
            except Exception as e:
                job.errors.append(f"An unexpected error occurred while reading the attachment: {e}")
                logger.error(f"An unexpected error occurred while reading the attachment: {e}")
                await ctx.send(f"❌ An unexpected error occurred while reading the attachment: {e}")
                return
        else:
            logger.warning(
                f"Attached file not recognized as JSON. Detected type: {attachment.content_type}, filename: {attachment.filename}."
            )
            await ctx.send(
                f"❌ The attached file is not a recognized JSON file type. "
                f"Expected `application/json` or a `.json` file with `text/plain` type. "
                f"Detected type: `{attachment.content_type}`, filename: `{attachment.filename}`."
            )
            return
    # If no attachment, check for JSON content directly in the message.
    elif json_content:
        # Attempt to extract JSON from a markdown code block (e.g., ````json\n...\n````).
        json_match = re.search(r'```json\n(.*)```', json_content, re.DOTALL)
        json_string_to_parse = ""

        if json_match:
            # If a markdown block is found, take its content and strip any surrounding whitespace.
            json_string_to_parse = json_match.group(1).strip()
        else:
            # If no markdown block, assume the entire content is JSON and strip whitespace.
            # This is key for allowing direct paste of JSON without a code block, but makes parsing more sensitive.
            json_string_to_parse = json_content.strip()

        try:
            parsed_json = json.loads(json_string_to_parse)
            logger.info(f"Parsed JSON from message content for design ID {design_id}")
        except json.JSONDecodeError as e:
            job.errors.append(f"Failed to parse JSON from message content: Invalid JSON syntax: {e}")
            logger.error(f"Failed to parse JSON from message content: {e}")
            await ctx.send(
                f"❌ Failed to parse JSON from message for design ID **{design_id}**: "
                f"**Invalid JSON syntax.** Please ensure it's valid JSON syntax. "
                f"Wrapping it in a ````json\n...\n``` ` block is highly recommended for best results. Error: {e}"
            )
            return
    else:
        await ctx.send(
            "❓ Please provide JSON directly in the message (preferably in a `json` markdown block) "
            "or as a `.json` attachment."
        )
        return

    if parsed_json:
        # Validate the parsed JSON using the external `json_validate` function.
        try:
            json_validate(parsed_json) # This function is expected to raise ValidationError on failure
            job.config_json = parsed_json
            logger.info(f"JSON configuration successfully set for design ID {design_id}.")
            await ctx.send(f"✅ JSON configuration successfully set for design ID **{design_id}**! ⚙️")
        except ValidationError as e: # Catching jsonschema.ValidationError specifically
            # Catch the specific validation error and extract its message
            error_message = str(e)
            job.errors.append(f"JSON configuration for design ID {design_id} is invalid: {error_message}")
            logger.warning(f"JSON configuration for design ID {design_id} is invalid: {e}")
            await ctx.send(
                f"❌ JSON configuration for design ID **{design_id}** is invalid. "
                f"It doesn't conform to the **backend validation schema**.\n"
                f"**Validation Error Details:** ```\n{error_message}\n```"
                f"Please review the expected JSON structure for this configuration."
            )
        except Exception as e:
            # Catch any other unexpected errors during validation
            job.errors.append(f"An unexpected error occurred during JSON validation for design ID {design_id}: {e}")
            logger.error(f"An unexpected error occurred during JSON validation for design ID {design_id}: {e}")
            await ctx.send(f"❌ An unexpected error occurred during JSON validation for design ID **{design_id}**: {e}")
    else:
        # This case should ideally not be reached if previous checks work.
        logger.error(f"No valid JSON content was found to set for design ID {design_id}. This indicates an unexpected state.")
        await ctx.send(f"❌ No valid JSON content was found to set for design ID **{design_id}**.")


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
            f"❌ Sorry, I couldn't understand that design ID. "
            f"Please provide a numeric ID or a valid Fantastic Contraption link like `https://example.com/?designId=12345`."
        )
        return

    job = await get_or_create_job(design_id) # Call get_or_create_job (now async)
    job.errors.clear() # Clear previous errors before attempting to retrieve config.

    if job.config_json is None:
        await ctx.send(f"ℹ️ No JSON configuration has been set for design ID **{design_id}** yet.")
        return

    json_output_string = json.dumps(job.config_json, indent=2)

    # Discord's message character limit is 2000. Send as a file if the JSON string is too long.
    if len(json_output_string) > 1900: # Using a soft limit (1900 characters) for safety.
        filename = f"design_{design_id}_config.json"
        with open(filename, 'w') as f:
            f.write(json_output_string)
        await ctx.send(
            f"⚙️ Here is the JSON configuration for design ID **{design_id}**:",
            file=discord.File(filename)
        )
        os.remove(filename); # Clean up the temporary file from the local filesystem.
        logger.info(f"JSON configuration for design ID {design_id} sent as file attachment.")
    else:
        # Send the JSON directly in the message within a markdown code block for easy viewing.
        await ctx.send(f"⚙️ Here is the JSON config for design ID **{design_id}**:\n```json\n{json_output_string}\n```")
        logger.info(f"JSON configuration for design ID {design_id} sent directly in message.")


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
            f"❌ Sorry, I couldn't understand that design ID. "
            f"Please provide a numeric ID or a valid Fantastic Contraption link like `https://example.com/?designId=12345`."
        )
        return

    job = await get_or_create_job(design_id) # Call get_or_create_job (now async)
    job.errors.clear() # Clear previous errors before attempting to start the job.

    # Store the channel ID where this command was invoked. This channel will be used
    # as the primary destination for future notifications related to this job.
    if isinstance(ctx.channel, discord.TextChannel):
        job.reporting_channel_id = ctx.channel.id

    # Check the necessary prerequisites for initializing the garden.
    if job.design_struct is None:
        job.errors.append("Cannot start garden: Design structure not loaded. This might be due to an invalid design ID or network issues during retrieval.")
        logger.warning(f"Cannot start garden for design {design_id}: Design structure not loaded.")
    if job.config_json is None:
        job.errors.append("Cannot start garden: JSON configuration not set. Please use `@BotName set_config` to set it first.")
        logger.warning(f"Cannot start garden for design {design_id}: JSON configuration not set.")

    # If any prerequisites are missing, report the errors to the user and terminate the command.
    if job.errors:
        error_report = "\n**Errors:**\n" + "\n".join([f"- {err}" for err in job.errors])
        await ctx.send(f"❌ Could not start garden for design ID **{design_id}**. Please fix the listed issues.{error_report}")
        job.errors.clear()
        return

    # If a garden is already initialized and running for this job, treat the command
    # as a request to "reheat" (re-fund) the existing job instead of starting a new one.
    if job.garden is not None:
        logger.info(f"Garden for design ID {design_id} is already running. Attempting to reheat instead.")
        await ctx.send(f"ℹ️ Garden for design ID **{design_id}** is already running. Attempting to reheat instead.")
        user_id = ctx.author.id
        user_name = ctx.author.display_name
        await _reheat_job_logic(ctx, design_id, job, user_id, user_name)
        return

    try:
        # Initialize the Garden object. This typically involves creating initial creatures
        # based on the design structure and configuration.
        job.garden = Garden([Creature(job.design_struct)], MAX_GARDEN_SIZE, job.config_json)
        logger.info(f"Garden successfully initialized for design ID {design_id}!")
        await ctx.send(f"✅ Garden successfully initialized for design ID **{design_id}**! 🎉")

        # Automatically "reheat" (fund) the job for the user who initiated it,
        # granting it an initial period of active time.
        user_id = ctx.author.id
        user_name = ctx.author.display_name
        await _reheat_job_logic(ctx, design_id, job, user_id, user_name)

    except Exception as e:
        job.errors.append(f"Failed to initialize garden for ID {design_id}: {e}")
        logger.error(f"Failed to initialize garden for design {design_id}: {e}")
        await ctx.send(f"❌ Failed to initialize garden for design ID **{design_id}**: {e}")
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
            f"❌ Sorry, I couldn't understand that design ID. "
            f"Please provide a numeric ID or a valid Fantastic Contraption link like `https://example.com/?designId=12345`."
        )
        return

    job = await get_or_create_job(design_id) # Call get_or_create_job (now async)
    job.errors.clear() # Clear previous errors before processing the reheat request.

    # A garden must be initialized for a job to be reheated.
    if job.garden is None:
        job.errors.append("Cannot reheat job: Garden not initialized. Please use `@BotName start_job` first.")
        logger.warning(f"Cannot reheat job for design {design_id}: Garden not initialized.")
        await ctx.send(f"❌ Cannot reheat job for design ID **{design_id}**. Garden not active. Please use `@BotName start_job`.")
        job.errors.clear()
        return

    user_id = ctx.author.id
    user_name = ctx.author.display_name

    # Call the shared reheating logic to apply funding rules and update the job's expiration.
    await _reheat_job_logic(ctx, design_id, job, user_id, user_name)
    job.errors.clear() # Ensure errors from `_reheat_job_logic` are cleared after reporting.


@bot.command()
async def subscribe(ctx: commands.Context, design_input: str, *notification_types: str):
    """
    Subscribes the invoking user to specific notifications for a design job.
    Supports multiple notification types at once.

    Args:
        design_input (str): The design ID or URL of the job to subscribe to.
        notification_types (str): One or more types of notifications to subscribe to.
                                  Currently supported: 'solve' and 'timeout'.

    Usage Examples:
      `@BotName subscribe 12345 solve`
      `@BotName subscribe https://fantasticcontraption.com/?designId=12345 timeout`
      `@BotName subscribe 12345 solve timeout`
    """
    design_id = extract_design_id(design_input)
    if design_id is None:
        await ctx.send("❌ Please provide a valid design ID or link.")
        return

    job = await get_or_create_job(design_id) # Call get_or_create_job (now async)
    
    # Subscription is only valid for jobs with an active garden.
    if job.garden is None:
        logger.warning(f"Cannot subscribe to alerts for design {design_id}: Garden not active.")
        await ctx.send(f"❌ Cannot subscribe to alerts for design ID **{design_id}**: Garden not active. Please use `@BotName start_job` first.")
        return

    user_id = ctx.author.id
    user_name = ctx.author.display_name
    
    successful_subscriptions = []
    invalid_types = []

    for notification_type in notification_types:
        notification_type_lower = notification_type.lower()

        if notification_type_lower == "solve":
            job.solve_subscribers[user_id] = user_name
            successful_subscriptions.append("solve")
            logger.info(f"User {user_name} subscribed to solve alerts for design {design_id}")
        elif notification_type_lower == "timeout":
            job.timeout_subscribers[user_id] = user_name
            successful_subscriptions.append("timeout")
            logger.info(f"User {user_name} subscribed to timeout alerts for design {design_id}")
        else:
            invalid_types.append(notification_type)
            logger.warning(f"Invalid notification type provided by {user_name}: {notification_type}")

    response_messages = []
    if successful_subscriptions:
        subscriptions_list = ", ".join([f"**{sub_type}** alerts" for sub_type in successful_subscriptions])
        response_messages.append(f"🔔 **{user_name}**, you are now subscribed to {subscriptions_list} for design ID **{design_id}**! I'll ping you here (or DM you if I can't) when these events occur.")
    
    if invalid_types:
        invalid_list = ", ".join([f"`{inv_type}`" for inv_type in invalid_types])
        response_messages.append(f"❌ I didn't recognize the following notification type(s): {invalid_list}. Supported types are `solve` and `timeout`.")

    if not response_messages:
        await ctx.send("ℹ️ Please specify at least one valid notification type (`solve` or `timeout`) to subscribe.")
    else:
        await ctx.send("\n".join(response_messages))


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
            f"❌ Sorry, I couldn't understand that design ID. "
            f"Please provide a numeric ID or a valid Fantastic Contraption link like `https://example.com/?designId=12345`."
        )
        return

    job = await get_or_create_job(design_id) # Call get_or_create_job (now async)
    job.errors.clear() # Clear general job errors before executing the snapshot command.

    # Snapshotting requires an active garden.
    if job.garden is None:
        job.errors.append("Garden not initialized for this design. Please use `@BotName start_job` first.")
        logger.warning(f"Cannot snapshot for design {design_id}: Garden not active.")
        await ctx.send(f"❌ Cannot snapshot for design ID **{design_id}**. Garden not active. Please use `@BotName start_job`.")
        job.errors.clear()
        return

    # Determine the number of creatures to upload, respecting the `DEFAULT_SNAPSHOT_K` limit.
    upload_best_k = DEFAULT_SNAPSHOT_K
    if k is not None:
        if k > DEFAULT_SNAPSHOT_K:
            logger.info(f"Snapshot count cannot exceed {DEFAULT_SNAPSHOT_K}. Using {DEFAULT_SNAPSHOT_K} instead of {k}.")
            await ctx.send(f"⚠️ Snapshot count cannot exceed {DEFAULT_SNAPSHOT_K}. Using {DEFAULT_SNAPSHOT_K} instead of {k}.")
            upload_best_k = DEFAULT_SNAPSHOT_K
        elif k <= 0:
            logger.warning("Snapshot count must be positive. No designs will be saved.")
            await ctx.send("❌ Snapshot count must be positive. No designs will be saved.")
            job.errors.append("Invalid snapshot count provided.")
            job.errors.clear()
            return
        else:
            upload_best_k = k

    # Perform the snapshot operation and collect the resulting links and any errors.
    # Corrected: Pass design_id to _perform_snapshot_and_get_links
    saved_links_messages, snapshot_errors = await _perform_snapshot_and_get_links(design_id, job, upload_best_k)

    # Report the results of the snapshot to the Discord channel.
    if saved_links_messages:
        # FIX: Message adjusted to indicate "top contenders" and clarify the snapshot.
        response_message = f"✅ Successfully snapshotted **top {len(saved_links_messages)} contender(s)** for ID **{design_id}** (from a sorted garden):"
        response_message += "\n" + "\n".join(saved_links_messages)
        await ctx.send(response_message)
        logger.info(f"Successfully snapshotted {len(saved_links_messages)} designs for ID {design_id}.")
    else:
        if snapshot_errors:
            logger.error(f"Failed to snapshot any designs for ID {design_id}. Errors: {snapshot_errors}")
            await ctx.send(f"❌ Failed to snapshot any designs for ID **{design_id}**. Please check previous error messages or the bot's console for details.")
        else:
            logger.info(f"No designs were snapshotted for ID {design_id}. No creatures available or k value too high.")
            await ctx.send(f"ℹ️ No designs were snapshotted for ID **{design_id}**. "
                           f"This might be because no creatures were available in the garden, "
                           f"or the specified `k` value ({upload_best_k}) was too high for the available creatures.")
    
    # Report any lingering snapshot-specific errors that occurred.
    if snapshot_errors:
        error_report_lines = ["\n**Snapshot Errors:** ⚠️"]
        current_length = len("\n**Snapshot Errors:** ⚠️\n")
        
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

@bot.command()
async def version(ctx: commands.Context):
    """
    Displays the bot's version information, including Git branch,
    dirty status, commit count, and HEAD SHA.
    Usage: @BotName version
    """
    git_info = await get_git_info()
    response = (
        f"**Bot Version Information:** ℹ️\n"
        f"  - **Branch:** `{git_info['branch']}`\n"
        f"  - **Local Changes (Dirty):** `{git_info['dirty']}`\n"
        f"  - **Commits at Head:** `{git_info['commits_at_head']}`\n"
        f"  - **HEAD SHA:** `{git_info['head_sha']}`"
    )
    logger.info("Version information displayed.")
    await ctx.send(response)

@bot.command()
async def stats(ctx: commands.Context):
    """
    Displays various operational statistics about the bot and its managed jobs.
    Usage: @BotName stats
    """
    total_jobs = len(jobs)
    active_jobs_count = 0
    frozen_jobs_count = 0
    
    all_funders_ever = set()
    current_active_funders = set()

    # Removed generation-related stats based on user feedback
    total_creatures_processed = 0 # To track total num_kills across all active gardens
    active_gardens_count = 0
    
    total_errors_logged = 0

    for design_id, job in jobs.items():
        if job.is_frozen():
            frozen_jobs_count += 1
        else:
            active_jobs_count += 1
        
        # Collect all funders, active or not
        for funder_id in job.funders.keys():
            all_funders_ever.add(funder_id)
        
        # Collect only currently active funders
        if not job.is_frozen():
            for funder_id in job.funders.keys():
                current_active_funders.add(funder_id)
        
        if job.garden is not None:
            if not job.is_frozen(): # Only count active gardens for processing stats
                active_gardens_count += 1
                total_creatures_processed += job.garden.num_kills # Accumulate num_kills for active gardens
        
        total_errors_logged += len(job.errors)

    avg_creatures_processed_per_garden = total_creatures_processed / active_gardens_count if active_gardens_count > 0 else 0

    uptime_seconds = time.time() - _bot_start_time
    # Format uptime nicely (days, hours, minutes, seconds)
    days, remainder = divmod(int(uptime_seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_string = []
    if days > 0: uptime_string.append(f"{days}d")
    if hours > 0: uptime_string.append(f"{hours}h")
    if minutes > 0: uptime_string.append(f"{minutes}m")
    # Only append seconds if no larger unit was added OR if seconds is the only non-zero unit
    if seconds > 0 or not uptime_string: uptime_string.append(f"{seconds}s")
    uptime_display = " ".join(uptime_string)

    response = (
        f"**Bot Operational Statistics:** 📊\n"
        f"  - **Uptime:** {uptime_display}\n"
        f"  - **Total Jobs Managed:** {total_jobs}\n"
        f"  - **Active Jobs:** {active_jobs_count} (🔥)\n"
        f"  - **Frozen Jobs:** {frozen_jobs_count} (❄️)\n"
        f"  - **Total Unique Funders (Ever):** {len(all_funders_ever)}\n"
        f"  - **Currently Active Funders:** {len(current_active_funders)}\n"
        f"  - **Active Gardens Currently Running:** {active_gardens_count} (🌱)\n"
        f"  - **Total Creatures Processed (Completed Runs):** {total_creatures_processed}\n"
        f"  - **Average Creatures Processed per Active Garden:** {avg_creatures_processed_per_garden:.2f}\n"
        f"  - **Total Errors Logged:** {total_errors_logged} (across all jobs) ⚠️\n"
    )
    logger.info("Bot statistics displayed.")
    await ctx.send(response)


# --- Background Task for Job Maintenance ---

async def background_loop():
    """
    A continuous background task that runs while the bot is active.
    This loop performs essential garden maintenance for all active jobs,
    dynamically distributing compute thread resources based on system congestion
    and a fair round-robin approach. It also actively monitors jobs to trigger
    solve and timeout notifications.
    """
    await bot.wait_until_ready(); # Ensure the bot is fully connected to Discord before starting the loop.

    global _background_loop_iteration_counter # Declare global to modify the counter.

    while not bot.is_closed():
        _background_loop_iteration_counter += 1
        logger.debug(f"Background loop running (Iteration {_background_loop_iteration_counter}) - performing garden maintenance and checking notifications...")
        
        # 1. Identify eligible jobs (those with an active garden and not currently frozen)
        # Simultaneously, check for and trigger timeout notifications for jobs that have become frozen.
        eligible_jobs_for_compute = []
        for design_id, job in list(jobs.items()): # Iterate over a copy to safely modify `jobs` dict if needed.
            # Check for Timeout Notification:
            # Trigger if the job is now frozen, had an expiration time, and hasn't been notified for this period.
            if job.garden is not None and job.is_frozen() and job.expiration_time is not None and not job.timeout_notified_for_current_period:
                if job.timeout_subscribers:
                    logger.info(f"Triggering timeout notification for design ID: {design_id}")
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
            logger.debug("No eligible gardens to maintain.")
        elif num_eligible_jobs == 1:
            # If only one job is active, give it all available global threads.
            design_id, job = eligible_jobs_for_compute[0]
            threads_to_allocate_per_job[design_id] = MAX_THREADS_GLOBAL
            job.last_turn_allocated = _background_loop_iteration_counter # Update its turn timestamp.
            logger.debug(f"Single garden {design_id} gets all {MAX_THREADS_GLOBAL} threads.")
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
                logger.debug(f"Allocating {allocated_threads} threads to garden {design_id}.")
        
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
                    logger.debug(f"Garden {design_id} Checkup: Active Threads={garden_status.num_active_threads}")

                    # Check for Solve Notification:
                    # Trigger if the best creature has a negative score (indicating a solve) AND
                    # a solve notification hasn't already been sent for the current active period.
                    if job.garden.creatures and job.garden.creatures[0].best_score is not None and \
                       job.garden.creatures[0].best_score < 0 and not job.has_solved_since_last_reheat:
                        if job.solve_subscribers:
                            logger.info(f"Triggering solve notification for design ID: {design_id}")
                            await _send_notification(bot, job, job.solve_subscribers, "solve", design_id)
                        job.has_solved_since_last_reheat = True # Mark as notified to prevent duplicate alerts.

                    # Perform garden evolution. Note: `evolve` is called with `MAX_THREADS_GLOBAL`
                    # as per previous instructions, assuming its internal logic handles parallelism
                    # rather than directly using the allocated thread count here for *new* work.
                    job.garden.evolve(MAX_THREADS_GLOBAL, RETAIN_BEST_K_EVOLUTION)
                    logger.debug(f"Garden {design_id} Evolved.")

                    # Calculate how many *new* threads the garden is allowed to start this turn.
                    # This ensures we don't exceed its allocated share and handles cases where
                    # a job might temporarily have more active threads than its current allocation.
                    new_threads_allowed = max(0, allocated_threads_for_this_job - garden_status.num_active_threads)

                    if new_threads_allowed > 0:
                        job.garden.start(new_threads_allowed)
                        logger.debug(f"Garden {design_id} Started {new_threads_allowed} new threads.")
                    else:
                        logger.debug(f"Garden {design_id} has enough active threads ({garden_status.num_active_threads}) or "
                                     f"was allocated {allocated_threads_for_this_job} new threads (no new threads started).")
                        # Existing threads will naturally terminate over time, bringing the job's
                        # active thread count down to its current allocation.

                except Exception as e:
                    job.errors.append(f"Error during background garden maintenance for ID {design_id}: {e}")
                    logger.error(f"Error maintaining garden {design_id}: {e}")
            elif job.garden is None:
                # This log message indicates jobs that exist but haven't had their garden initialized yet.
                logger.debug(f"Design ID {design_id} has no active garden to maintain.")
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
            logger.critical("Error: Invalid bot token was provided. Please ensure your 'DISCORD_BOT_TOKEN' environment variable is correct.")
        except Exception as e:
            logger.critical(f"An unexpected error occurred while running the bot: {e}")
    else:
        # If the bot token is missing, print an error and exit the script.
        logger.critical("Error: Discord bot token is missing. Please set the 'DISCORD_BOT_TOKEN' environment variable.")
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
