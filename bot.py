import logging
import os
import base64
import io
import aiosqlite
import asyncio
import requests
import json
import ast
from asyncio import Lock
from zoneinfo import ZoneInfo
from collections import defaultdict
from datetime import datetime, timedelta, time
from telegram import Update, constants
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
from telegram.request import HTTPXRequest
from telegram.error import NetworkError, TimedOut
from openai import AsyncOpenAI

# 1. Logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger("Simulacrum")
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Swallows network errors so the output stays clean."""
    if isinstance(context.error, (NetworkError, TimedOut)):
        logger.warning(f"Network glitch: {context.error}")
        return
    logger.error("Exception while handling an update:", exc_info=context.error)

# 2. Config
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OWNER_CHAT_ID = os.getenv("OWNER_CHAT_ID")  # For morning briefings
DB_FILE = "bot_memory.db"
MODEL_NAME = "gpt-5-mini"
NUM_HISTORY_MESSAGES = 10
IMAGE_QUALITY = "low"  # Options: low, medium, high
MULTIMODAL = True  # Enable/disable image input, output, and history
SCOPES = ['https://www.googleapis.com/auth/calendar']
MAX_TOOL_CALLS = 3
SHOW_LINKS = False  # Whether to return Google Calendar links to model
MY_TIMEZONE = 'America/New_York'
TOP_N_EVENTS = 7  # Max events to return in list_events

if not TELEGRAM_TOKEN or not OPENAI_API_KEY or not OWNER_CHAT_ID:
    raise ValueError("Missing keys! Did you run 'sudo micro /etc/systemd/system/tg-bot.service'?")

# Use AsyncOpenAI to keep the bot responsive
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Per-chat locks to ensure messages are processed in order
chat_locks = defaultdict(Lock)

# --- SYSTEM PROMPT ---

SYSTEM_PROMPT = f"""
You are a ChatGPT model running on a Raspberry Pi Zero 2. Conversations occur on Telegram via texting.

You have access to the user's Google Calendar and can create, delete, edit, and list events upon request. Use the provided tools to interact with the calendar as needed.

TOOLS GUIDE:
- To CREATE an event: Use 'create_event'. If you don't have enough information to make the event, ask the user for the missing details. If the user doesn't specify an end time, assume the event lasts 1 hour.
- To DELETE an event: Use 'delete_event'. You need the event ID to do this, which you can get from 'list_events'. NOTE: Deleting a recurring event instance will delete the entire series.
- To EDIT an event: Perform a "Delete and Remake". First do 'delete_event' on the old one, then 'create_event' with the new details.
- To LIST events: Use 'list_events' with a time range. To DELETE or EDIT an event, list_events around the expected time to find the information you need (event ID).

You can do {MAX_TOOL_CALLS} tool calls per user message. This means that to EDIT an event, you can first LIST events to find it, then DELETE it, and finally CREATE the new version.

Do not share the event IDs with the user directly. Instead, use them internally to manage events.
"""

if MULTIMODAL:
    SYSTEM_PROMPT += "\n\nYou also have the ability to send the user images."

# --- Google Calendar Setup ---

def is_connected():
    try:
        requests.head('http://www.google.com', timeout=3)
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False

def get_calendar_service():
    if not is_connected():
        logger.warning("[Google] No internet — skipping.")
        return None

    token_path = "token.json"
    creds_path = "credentials.json"
    creds = None

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        logger.error("[Google] Google client libraries not installed.")
        return None

    if not os.path.exists(creds_path):
        logger.error("[Google] No credentials.json found. Must upload credentials to the Pi from main computer.")
        return None

    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception:
            logger.error("[Google] Could not load credentials from token.json. May need to upload a new token to the Pi or credentials are invalid.")
            return None  # Can't refresh in Raspberry Pi environment (no browser)
    else:
        logger.error("[Google] No token.json found. Must authenticate on main computer and upload token.json.")
        return None
    
    try:  
        service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        logger.info("[Google] Authenticated successfully.")
        return service
    except Exception as e:
        logger.error(f"[Google] Auth failed: {e}")
        return None

# Get the calendar service at startup
GOOGLE_CALENDAR_SERVICE = get_calendar_service()

# --- Google Calendar Event Functions ---

def to_google_iso(date_str, time_str):
    """
    Combines 'MM/DD/YYYY' and 'HH:MM' (24hr) into Google's ISO format.
    Robustly strips quotes, brackets, and extra spaces.
    Example: ('[01/10/2026]', '"15:00"') -> '2026-01-10T15:00:00Z'
    """
    if not date_str or not time_str:
        return None
    try:
        # Peel off brackets, parens, quotes, and spaces
        garbage = " []()'\""
        d = date_str.strip(garbage)
        t = time_str.strip(garbage)
        # Parse the clean strings
        dt = datetime.strptime(f"{d} {t}", "%m/%d/%Y %H:%M")
        # Add time zone info
        dt_aware = dt.replace(tzinfo=ZoneInfo(MY_TIMEZONE))
        # Return the format Google expects
        return dt_aware.isoformat()
    except ValueError:
        logger.error(f"Date parse error: Raw('{date_str}', '{time_str}')")
        return None

def from_google_iso(iso_str):
    """
    Converts Google's ISO string to User's Local Time (MM/DD/YYYY, HH:MM).
    """
    if not iso_str:
        return None, None
    
    try:
        # Handle full datetime (Has 'T')
        if 'T' in iso_str:
            # 1. Parse the ISO string (handling Z for UTC if present)
            dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
            
            # 2. FIX: Convert to MY_TIMEZONE
            # This ensures 14:00 UTC becomes 09:00 EST automatically.
            dt = dt.astimezone(ZoneInfo(MY_TIMEZONE))
            
            return dt.strftime("%m/%d/%Y"), dt.strftime("%H:%M")
        
        # Handle all-day event (YYYY-MM-DD)
        else:
            dt = datetime.strptime(iso_str, "%Y-%m-%d")
            return dt.strftime("%m/%d/%Y"), "All Day"
            
    except ValueError:
        logger.error(f"Date parse error: {iso_str}")
        return iso_str, ""

# Google Calendar Color Map
COLOR_MAP = {
    "errands": "1",         # lavender
    "social": "2",          # sage
    "leisure": "3",         # grape
    "deadlines": "4",       # flamingo
    "appointments": "5",    # banana
    "goals": "6",           # tangerine
    "quiet": "7",           # peacock
    "reminders": "8",       # graphite
    "birthdays": "9",       # blueberry
    "health": "10",         # basil
    "work": "11"            # tomato
}
REVERSE_COLOR_MAP = {v: k for k, v in COLOR_MAP.items()}

def create_event(service, summary, description, start_date, start_time, end_date, end_time, recurrence=None, category=None):
    # Convert simple strings to Google ISO
    logger.info(f"Creating event: {summary} ({recurrence if recurrence else 'Single'})")
    start_iso = to_google_iso(start_date, start_time)
    end_iso = to_google_iso(end_date, end_time)

    if not start_iso or not end_iso:
        return {"error": "Invalid date format. Use MM/DD/YYYY and HH:MM."}

    event = {
        'summary': summary,
        'description': description,
        'start': {'dateTime': start_iso, 'timeZone': MY_TIMEZONE},
        'end': {'dateTime': end_iso, 'timeZone': MY_TIMEZONE},
    }
    # --- CATEGORY LOGIC ---
    if category and category.lower() in COLOR_MAP:
        event['colorId'] = COLOR_MAP[category.lower()]

    # Handle Recurrence Rules (RRULE)
    # The AI sends: "DAILY", "WEEKLY", "MONTHLY"
    # Google wants: ["RRULE:FREQ=WEEKLY"]
    if recurrence:
        rrule = f"RRULE:FREQ={recurrence.upper()}"
        event['recurrence'] = [rrule]

    try:
        # Execute the call
        event_result = service.events().insert(calendarId='primary', body=event).execute()

        cleaned_result = {
            "id": event_result.get("id"),
            "summary": summary,
            "description": description,
            "start": f"{start_date} {start_time}",
            "end": f"{end_date} {end_time}",
            "recurrence": recurrence,
            "category": category
        }
        if SHOW_LINKS:
            cleaned_result['link'] = event_result.get("htmlLink")
        return cleaned_result
    except Exception as e:
        logger.error(f"Failed to create event: {e}")
        return {"error": f"{e}"}

def delete_event(service, event_id):
    try:
        # Detect recurring instance ID (e.g., "12345_20260101T...")
        # Strip the suffix to delete the "Master Series" instead of just one day.
        if "_" in event_id:
            parts = event_id.split('_')
            if len(parts) > 1 and 'T' in parts[-1] and 'Z' in parts[-1]:
                event_id = parts[0]
                logger.info(f"Recurring instance detected. Deleting master series: {event_id}")

        service.events().delete(calendarId='primary', eventId=event_id).execute()
        logger.info(f"Event {event_id} deleted successfully.")
        
        # Return a dict to match the schema of create_event
        return {
            "id": event_id
        }
    except Exception as e:
        logger.error(f"Failed to delete event {event_id}: {e}")
        return [{"error": f"{e}"}]

def list_events(service, start_date, start_time, end_date, end_time, categories=None):
    # 1. Normalize 'categories' input to always be a list (or empty list)
    target_cats = categories if categories else []
            
    cat_log_str = ', '.join(target_cats) if target_cats else 'All'
    logger.info(f"Listing events from {start_date} {start_time} to {end_date} {end_time} (Prioritizing: {cat_log_str})")

    timeMin = to_google_iso(start_date, start_time)
    timeMax = to_google_iso(end_date, end_time)

    # 2. FETCH WIDE: Get many events so to have a deep pool to sort from
    max_results = 500
    
    try:
        events_result = service.events().list(
            calendarId='primary', timeMin=timeMin, timeMax=timeMax,
            maxResults=max_results, singleEvents=True, orderBy='startTime'
        ).execute()
        items = events_result.get('items', [])
        
        all_parsed_events = []
        
        # 3. PARSE EVERYTHING
        for item in items:
            # Determine Category
            color_id = item.get('colorId') 
            event_category = REVERSE_COLOR_MAP.get(color_id, "default") # 'default' instead of 'uncategorized' matches your new map

            # Parse Dates
            start_raw = item['start'].get('dateTime', item['start'].get('date'))
            end_raw = item['end'].get('dateTime', item['end'].get('date'))
            s_date, s_time = from_google_iso(start_raw)
            e_date, e_time = from_google_iso(end_raw)

            # Parse Recurrence
            recurrence_val = None
            rec_list = item.get('recurrence')
            if rec_list:
                try:
                    recurrence_val = rec_list[0].split('FREQ=')[1].split(';')[0]
                except IndexError:
                    pass

            # Build Object
            cleaned_result = {
                'id': item['id'], 
                'summary': item.get('summary', '*No title*'),
                'description': item.get('description', ''),
                'start': f"{s_date} {s_time}",
                'end': f"{e_date} {e_time}",
                'recurrence': recurrence_val,
                'category': event_category
            }
            if globals().get('SHOW_LINKS', False):
                cleaned_result['link'] = item.get('htmlLink')
                
            all_parsed_events.append(cleaned_result)

        # 4. SORT & SPLIT (The "Idiot Proof" Logic)
        if not target_cats:
            # No filter? Just return the top N by time
            return all_parsed_events[:TOP_N_EVENTS]
        
        # Split into "Priority" (Matches) and "Background" (Everything else)
        priority_events = []
        background_events = []
        
        for event in all_parsed_events:
            if event['category'] in target_cats:
                priority_events.append(event)
            else:
                background_events.append(event)
        
        # 5. MERGE: Put priority first, fill remaining slots with background events
        # Note: Both lists are already sorted by time because Google sent them that way
        combined_events = priority_events + background_events
        
        # 6. TRUNCATE: Only return the top N to save tokens
        return combined_events[:TOP_N_EVENTS]

    except Exception as e:
        logger.error(f"Failed to list events: {e}")
        return {"error": f"{e}"}

# --- OpenAI API Calendar Tools ---

create_event_tool = {
    "type": "function",
    "name": "create_event",
    "description": "Create a new event. Requires full date and time inputs.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Title of the event."},
            "description": {"type": "string", "description": "Description/Notes."},
            "start_date": {"type": "string", "description": "Date in MM/DD/YYYY format."},
            "start_time": {"type": "string", "description": "Time in HH:MM 24-hour format."},
            "end_date": {"type": "string", "description": "Date in MM/DD/YYYY format."},
            "end_time": {"type": "string", "description": "Time in HH:MM 24-hour format."},
            "recurrence": {"type": ["string", "null"], "enum": ["DAILY", "WEEKLY", "MONTHLY", "YEARLY", None], "description": "Frequency for recurring events. Leave null for single, one-time events."},
            "category": {"type": "string", "enum": list(COLOR_MAP.keys()), "description": "The category of the event."
            }
        },
        "required": ["summary", "description", "start_date", "start_time", "end_date", "end_time", "recurrence", "category"],
        "additionalProperties": False
    },
    "strict": True
}

delete_event_tool = {
    "type": "function",
    "name": "delete_event",
    "description": "Delete an event from the calendar using its ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "event_id": {"type": "string", "description": "The unique identifier of the event to delete, e.g. 's06qa9f1bidp2b4prkda0h275c'."}
        },
        "required": ["event_id"],
        "additionalProperties": False
    },
    "strict": True
}

list_events_tool = {
    "type": "function",
    "name": "list_events",
    "description": "List events within a time range.",
    "parameters": {
        "type": "object",
        "properties": {
            "start_date": {"type": "string", "description": "Search start date in MM/DD/YYYY format."},
            "start_time": {"type": "string", "description": "Search start time in HH:MM 24-hour format."},
            "end_date": {"type": "string", "description": "Search end date in MM/DD/YYYY format."},
            "end_time": {"type": "string", "description": "Search end time in HH:MM 24-hour format."},
            "categories": {
                "type": ["array", "null"],
                "items": {
                    "type": "string",
                    "enum": list(COLOR_MAP.keys())
                },
                "description": "Prioritize events from these categories. Leave null to list all events."
            }
        },
        "required": ["start_date", "start_time", "end_date", "end_time", "categories"],
        "additionalProperties": False
    },
    "strict": True
}

# Create the main tool list
TOOLS = [
    create_event_tool,
    delete_event_tool,
    list_events_tool
]

# Define tools (image generation only if MULTIMODAL is enabled)
if MULTIMODAL:
    TOOLS.append({
        "type": "image_generation",
        "quality": IMAGE_QUALITY,
        "moderation": "low",
        "size": "1024x1024"
    })

# --- Async Database Functions ---

async def init_db():
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                role TEXT,
                content TEXT,
                image_data TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Add image_data column if it doesn't exist (for existing databases)
        try:
            await db.execute('ALTER TABLE messages ADD COLUMN image_data TEXT')
        except aiosqlite.OperationalError:
            pass  # Column already exists
        await db.commit()

async def save_message(chat_id, role, content, image_data=None):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            "INSERT INTO messages (chat_id, role, content, image_data) VALUES (?, ?, ?, ?)",
            (chat_id, role, content, image_data)
        )
        await db.commit()

async def get_recent_history(chat_id, limit=30, include_images=True):
    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT role, content, image_data FROM messages 
            WHERE chat_id = ?
            ORDER BY id DESC 
            LIMIT ?
        """, (chat_id, limit))
        rows = await cursor.fetchall()
    
    messages = []
    rows = list(reversed(rows))
    total_messages = len(rows)

    for i, row in enumerate(rows):
        role = row["role"]
        content = row["content"]
        image_data = row["image_data"]
        # --- COMPRESSION LOGIC --- SAVES ON TOKENS
        # 1. Check for specific tag
        if role == "system" and "[Tool Result]" in content:
            # 2. Only compress if older than N turns
            if i < (total_messages - MAX_TOOL_CALLS):
                try:
                    # FIX: Split on the first colon (:), which separates metadata from data
                    # content looks like: "[Tool Result] list_events: [{...}, {...}]"
                    parts = content.split(":", 1)
                    # Extract Name: "[Tool Result] list_events" -> "list_events"
                    header = parts[0]
                    # Extract Data
                    raw_data = parts[1].strip()
                
                    data = ast.literal_eval(raw_data)
                    compressed_content = None

                    # Helper to shrink one event
                    def shrink(evt):
                        return {
                            'id': evt.get('id'),
                            'summary': evt.get('summary'),
                            'start': evt.get('start'),
                            'category': REVERSE_COLOR_MAP.get(evt.get('category'))
                        }

                    # Handle List (list_events)
                    if isinstance(data, list):
                        compressed_content = [shrink(e) for e in data if isinstance(e, dict)]
                    
                    # Handle Single Dict (create_event / delete_event)
                    elif isinstance(data, dict):
                        compressed_content = shrink(data)
                    
                    if compressed_content:
                        content = f"{header}: {str(compressed_content)}"
                        # logger.info(f"Compressed content: {content}")

                except Exception:
                    # If parsing fails, just leave the original content alone
                    pass

        if role == "user" and image_data and include_images:
            content_block = [
                {"type": "input_text", "text": content},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_data}",
                    "detail": "low"
                }
            ]
            messages.append({"role": role, "content": content_block})
        else:
            messages.append({"role": role, "content": content})
    
    return messages

async def clear_history(chat_id):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        await db.commit()

# --- Image Helper ---

async def encode_image_from_telegram(photo_file):
    """Downloads a photo from Telegram and converts it to base64 for OpenAI."""
    out = io.BytesIO()
    await photo_file.download_to_memory(out)
    out.seek(0)
    return base64.b64encode(out.read()).decode('utf-8')

async def send_action_periodically(context, chat_id, action):
    """Keeps the 'typing...' or 'uploading...' status alive to prevent timeouts."""
    try:
        while True:
            await context.bot.send_chat_action(chat_id=chat_id, action=action)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass

# --- Main Chat Logic ---

async def chat_logic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    
    # Serialize processing per chat to maintain message order
    async with chat_locks[chat_id]:
        # 1. Capture User Input
        user_text = ""
        
        # 1. Handle Voice Notes
        if update.message.voice:
            # Send a "typing" status so user knows it's listening
            await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
            
            try:
                # Download the voice file from Telegram
                voice_file = await update.message.voice.get_file()
                
                # Save to memory buffer (saves SD card writes)
                voice_buffer = io.BytesIO()
                await voice_file.download_to_memory(voice_buffer)
                voice_buffer.name = "voice.ogg" # Vital: Tells OpenAI this is an OGG file
                
                # Send to Whisper API
                transcript = await client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=voice_buffer
                )
                user_text = transcript.text
                
                # Optional: Reply with what it heard (so user can trust it)
                await update.message.reply_text(f"🎤 *Heard:* {user_text}", parse_mode=constants.ParseMode.MARKDOWN)
                
            except Exception as e:
                logger.error(f"Transcription failed: {e}")
                await update.message.reply_text("I couldn't hear that clearly.")
                return

        # Handle Text or Photo Captions (Fallback)
        elif update.message.caption:
            user_text = update.message.caption
        elif update.message.text:
            user_text = update.message.text
        
        # If there is still no text (and no photo), ignore.
        if not user_text and not update.message.photo:
            return

        # 2. Check for photo input (Indentation Fixed)
        base64_image = None
        if update.message.photo and MULTIMODAL:
            photo_obj = await update.message.photo[-1].get_file()
            base64_image = await encode_image_from_telegram(photo_obj)

        current_time = datetime.now().strftime("%m/%d/%Y %H:%M")  # Standard format for this script
        
        # Start typing status
        typing_task = asyncio.create_task(
            send_action_periodically(context, chat_id, constants.ChatAction.TYPING)
        )

        try:
            # 4. Save User Message
            db_content = f"[{current_time}] {user_text}" + (" [Attached Image]" if base64_image else "")
            await save_message(chat_id, "user", db_content, image_data=base64_image)

            # 5. Fetch History
            history = await get_recent_history(chat_id, limit=NUM_HISTORY_MESSAGES, include_images=MULTIMODAL)
            
            # 6. Construct Payload
            input_payload = [{"role": "system", "content": SYSTEM_PROMPT}] + history

            # 7. Agentic Loop (Max N turns)
            for _ in range(MAX_TOOL_CALLS):
                response = await client.responses.create(
                    model=MODEL_NAME,
                    input=input_payload,
                    tools=TOOLS if TOOLS else None,
                    text={"verbosity": "low"}
                )

                # Append EVERYTHING the model just said to history immediately.
                # This ensures 'reasoning' blocks stay attached to their 'function_calls'.
                input_payload.extend(response.output)

                # --- TOKEN LOGGING ---
                if hasattr(response, 'usage') and response.usage:
                    u = response.usage
                    logger.info(f"💰 Tokens: {u.total_tokens} (In: {u.input_tokens}, Out: {u.output_tokens})")

                tool_ran = False
                text_reply = ""
                generated_image = None
                
                # 8. Process Output Items
                for item in response.output:
                    
                    # CASE A: Text Response
                    if item.type == "message":
                        raw_content = item.content
                        final_text_block = ""
                        if isinstance(raw_content, list):
                            for part in raw_content:
                                if isinstance(part, str): final_text_block += part
                                elif hasattr(part, 'text'): final_text_block += part.text
                                elif isinstance(part, dict) and 'text' in part: final_text_block += part['text']
                        elif isinstance(raw_content, str):
                            final_text_block = raw_content

                        if final_text_block:
                            text_reply += final_text_block
                            # Send chunks immediately to keep it snappy
                            await update.message.reply_text(final_text_block)

                    # CASE B: Image Generation
                    elif item.type == "image_generation_call":
                        if hasattr(item, 'result') and item.result:
                            upload_task = asyncio.create_task(send_action_periodically(context, chat_id, constants.ChatAction.UPLOAD_PHOTO))
                            try:
                                image_bytes = base64.b64decode(item.result)
                                await update.message.reply_photo(photo=io.BytesIO(image_bytes))
                                generated_image = item.result
                            finally:
                                upload_task.cancel()

                    # CASE C: Tool Calls
                    elif item.type == "function_call":
                        tool_ran = True
                        logger.info(f"Tool call: {item.name}")

                        function_map = {
                            "create_event": create_event,
                            "delete_event": delete_event,
                            "list_events": list_events
                        }

                        message_map_success = {
                            "create_event": "Event created",
                            "delete_event": "Event deleted",
                            "list_events": "Events listed"
                        }

                        message_map_failure = {
                            "create_event": "Failed to create event",
                            "delete_event": "Failed to delete event",
                            "list_events": "Failed to list events"
                        }
                        
                        fn_name = item.name
                        fn_args = json.loads(item.arguments)
                        
                        if not GOOGLE_CALENDAR_SERVICE:
                            await update.message.reply_text("⚠️ Google Calendar unavailable.")
                            continue

                        if fn_name in function_map:
                            status_msg = await update.message.reply_text(f"🔄 Executing {fn_name}...")
                            try:
                                result = function_map[fn_name](GOOGLE_CALENDAR_SERVICE, **fn_args)
                                # If error, defer to exception block
                                if isinstance(result, dict) and result.get("error"):
                                    raise Exception(result["error"])
                                elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and result[0].get("error"):
                                    raise Exception(result[0]["error"])
                                
                                # --- Define the output item before appending ---
                                tool_output_item = {
                                    "type": "function_call_output",
                                    "call_id": item.call_id,
                                    "output": json.dumps(result) if result else "Success"
                                }

                                # Update UI
                                ui_text = f"✅ {message_map_success.get(fn_name)}."
                                if fn_name == "create_event" and result and SHOW_LINKS:
                                    ui_text += f"\n{result.get('htmlLink', '')}"
                                
                                await context.bot.edit_message_text(
                                    chat_id=chat_id, 
                                    message_id=status_msg.message_id, 
                                    text=ui_text
                                )

                                # Feed back to AI
                                input_payload.append(tool_output_item) # The result

                                logger.info(f"[Tool Result] {fn_name + ' completed'}: {str(result)}")
                                
                                await save_message(chat_id, "system", f"[Tool Result] {fn_name + ' completed'}: {str(result)}")  # Important... this is what the AI sees later

                            except Exception as e:
                                logger.error(f"[Tool Result] {fn_name + ' failed'}: {e}")
                                error_item = {
                                    "type": "function_call_output",
                                    "call_id": item.call_id,
                                    "output": f"Error: {str(e)}"
                                }
                                input_payload.append(error_item)
                                await context.bot.edit_message_text(
                                    chat_id=chat_id, 
                                    message_id=status_msg.message_id, 
                                    text=f"❌ {message_map_failure.get(fn_name)}: {e}"
                                )
                        else:
                            # Handle Hallucinated Tools
                            logger.warning(f"[Tool Result] AI tried to call unknown tool: {fn_name}")
                            error_item = {
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": "Error: Tool not found."
                            }
                            input_payload.append(error_item) # Append the error

                # Loop Logic: Check if need to run again (OUTSIDE the items loop)
                if not tool_ran:
                    break
            
            # 9. Save Assistant Memory
            if text_reply or generated_image:
                final_content = f"[{current_time}] {text_reply}" + (" [Generated Image]" if generated_image else "")
                await save_message(chat_id, "assistant", final_content)

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            await update.message.reply_text("Something went wrong.")
        finally:
            if not typing_task.cancelled(): typing_task.cancel()

# --- Commands ---

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await clear_history(update.effective_chat.id)
    await update.message.reply_text("Cleared memory.")
    logger.info("Cleared memory.")

# --- Morning Briefing ---

async def morning_briefing(context: ContextTypes.DEFAULT_TYPE):
    """Checks calendar for today and sends a summary if events exist."""
    if not GOOGLE_CALENDAR_SERVICE:
        return

    # 1. Get Today's Date Range
    now = datetime.now(ZoneInfo(MY_TIMEZONE))
    today_str = now.strftime("%m/%d/%Y")
    
    # 2. List Events for Today
    # Use existing function
    events = list_events(
        GOOGLE_CALENDAR_SERVICE, 
        today_str, "00:00", 
        today_str, "23:59"
    )

    if not events:
        # Optional: Send a "No events today" message? 
        # Usually better to stay silent if empty.
        return

    # 3. Generate Briefing with AI
    # Construct a specific one-off prompt for this
    system_instruction = "You are a helpful personal assistant. Summarize the following schedule for the user's morning briefing, and suggest ways they can accomplish their goals. Be concise and encouraging."
    user_content = f"Here is my schedule for today ({today_str}):\n{json.dumps(events)}"

    try:
        response = await client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            text={"verbosity": "medium"}
        )
        
        # Extract text (simplified for this one-off call)
        briefing_text = ""
        for item in response.output:
            if item.type == "message" and isinstance(item.content, str):
                briefing_text += item.content
            elif item.type == "message" and isinstance(item.content, list):
                for part in item.content:
                    if hasattr(part, 'text'): briefing_text += part.text

        # 4. Send to Telegram
        if briefing_text:
            await context.bot.send_message(chat_id=OWNER_CHAT_ID, text=f"☀️ **Morning Briefing**\n\n{briefing_text}", parse_mode=constants.ParseMode.MARKDOWN)
            
    except Exception as e:
        logger.error(f"Morning briefing failed: {e}")

# --- Main ---

if __name__ == '__main__':
    # Init DB Loop
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(init_db())

    # Start Bot
    t_request = HTTPXRequest(connection_pool_size=8, connect_timeout=20.0, read_timeout=20.0)
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).request(t_request).build()
    
    app.add_error_handler(error_handler)
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler((filters.TEXT | filters.PHOTO | filters.VOICE) & ~filters.COMMAND, chat_logic))

    app.job_queue.run_daily(morning_briefing, time=time(hour=6, minute=0, second=0, tzinfo=ZoneInfo(MY_TIMEZONE)))
    
    logger.info(f"Simulacrum Online. Model: {MODEL_NAME}")
    app.run_polling()