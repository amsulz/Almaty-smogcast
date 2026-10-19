"""Telegram bot server: answers everyone who writes to the bot and sends the forecast to all subscribers.

Runs on PythonAnywhere (free) as a Flask web app, so the bot answers even when my computer is off.
GitHub Actions still makes the forecast at 22:00 and posts it to /broadcast (see src/telegram_bot.py).

Settings in server/.env (not on GitHub):
TELEGRAM_TOKEN=from @BotFather
SERVER_SECRET=any long random string, only letters, digits, _ and - (the same one in GitHub Secrets)

Once, after the web app is running:  python bot_server.py --set-webhook https://USERNAME.pythonanywhere.com
"""
import hmac
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, abort, request

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

API = "https://api.telegram.org/bot{token}/{method}"
# chat ids stay on the server, they are other people's data and shouldn't be on GitHub
SUBSCRIBERS = HERE / "subscribers.json"
LATEST = HERE / "latest.json"
NO_FORECAST = "Прогноза пока нет, он придёт сегодня в 22:00."

app = Flask(__name__)


def load(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def telegram(method, **params):
    return requests.post(API.format(token=os.environ["TELEGRAM_TOKEN"], method=method),
                         data=params, timeout=30)


def send(chat_id, text):
    # False when it didn't go through, e.g. the person blocked the bot
    return telegram("sendMessage", chat_id=chat_id, text=text).ok


def secret_ok(value):
    return hmac.compare_digest(value or "", os.environ["SERVER_SECRET"])


@app.post("/telegram")
def on_message():
    # Telegram sends our secret back in this header, so nobody else can pretend to be Telegram
    if not secret_ok(request.headers.get("X-Telegram-Bot-Api-Secret-Token")):
        abort(403)
    message = request.get_json().get("message")
    if not message:  # edited messages and other updates
        return "ok"
    chat_id = message["chat"]["id"]
    text = message.get("text", "")
    subscribers = load(SUBSCRIBERS, [])
    latest = load(LATEST, {})

    if text.startswith("/stop"):
        if chat_id in subscribers:
            subscribers.remove(chat_id)
            save(SUBSCRIBERS, subscribers)
        send(chat_id, "Вы отписались. Чтобы снова получать прогноз, нажмите /start.")
    elif text.startswith("/start"):
        if chat_id not in subscribers:
            subscribers.append(chat_id)
            save(SUBSCRIBERS, subscribers)
        if "scale" in latest:
            send(chat_id, latest["scale"])
        send(chat_id, latest.get("text", NO_FORECAST))
    else:
        # any other message: just the newest forecast
        send(chat_id, latest.get("text", NO_FORECAST))
    return "ok"


@app.post("/broadcast")
def broadcast():
    # called by GitHub Actions every evening
    if not secret_ok(request.headers.get("X-Server-Secret")):
        abort(403)
    latest = request.get_json()
    save(LATEST, latest)
    subscribers = load(SUBSCRIBERS, [])
    sent = 0
    for chat_id in subscribers:
        sent += send(chat_id, latest["text"])
        time.sleep(0.05)  # Telegram allows about 30 messages per second
    return {"subscribers": len(subscribers), "sent": sent}


def set_webhook(site):
    # tells Telegram to send every new message to our server
    r = telegram("setWebhook", url=site.rstrip("/") + "/telegram",
                 secret_token=os.environ["SERVER_SECRET"])
    print(r.json())


if __name__ == "__main__":
    if "--set-webhook" in sys.argv:
        set_webhook(sys.argv[sys.argv.index("--set-webhook") + 1])
    else:
        print(__doc__)
