# Simulacrum
A personal AI assistant running on Raspberry Pi. Manages Google Calendar, transcribes voice notes, and remembers context via a secure Telegram interface. Powered by OpenAI.

## Features

* **Persistent Memory:** Stores conversation history locally using SQLite (`bot_memory.db`).
* **Calendar Agent:** Autonomous Google Calendar management (Create, Delete, List, Search) with full timezone and recurrence support.
* **Voice Interaction:** Transcribes Telegram voice notes instantly using OpenAI Whisper.
* **Morning Briefings:** Automatically sends a daily schedule summary at 6:00 AM.
* **Multi-Modal:** Supports image recognition and generation.

## Prerequisites

* Raspberry Pi Zero 2 W (or any Linux server)
* Python 3.9+
* Telegram Bot Token
* OpenAI API Key
* Google Calendar API Credentials (`credentials.json`)
* `ffmpeg` (for voice processing)

## Installation

1.  **System Dependencies**
    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```

2.  **Clone & Setup**
    ```bash
    git clone [https://github.com/henrydaum/simulacrum.git](https://github.com/henrydaum/simulacrum.git)
    cd simulacrum
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Google Auth**
    * Place your `credentials.json` from Google Cloud Console in the root folder.
    * Run the script locally once to generate `token.json` (requires browser login), then upload `token.json` to the Pi.

## Configuration

Create a `.env` file or set these variables in your systemd service:

```ini
TELEGRAM_BOT_TOKEN=your_token_here
OPENAI_API_KEY=sk-your_key_here
OWNER_CHAT_ID=123456789
