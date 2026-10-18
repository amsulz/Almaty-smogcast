"""Telegram bot: downloads fresh data and sends tomorrow's PM2.5 forecast.

Setup (once):
1. In Telegram, write to @BotFather -> /newbot -> copy the token
2. Put it in .env:  TELEGRAM_TOKEN=your_token
3. Write any message to your bot, then run:  python src/telegram_bot.py --chat-id
4. Put the number it prints in .env:  TELEGRAM_CHAT_ID=the_number

Every evening:  python src/telegram_bot.py
On GitHub it runs by itself every evening: .github/workflows/telegram_bot.yml
"""
import os
import sys

import requests
from dotenv import load_dotenv

import build_dataset
import download_airgradient
import download_weather
from forecast import load_data, load_model, predict_tomorrow

API = "https://api.telegram.org/bot{token}/{method}"
# US EPA PM2.5 scale (2024), 24h average. For the message I use the full official scale,
# my 3 groups in evaluate.py are too rough to tell people if they can go for a walk
LEVELS = [
    (9, "🟢", "чисто", "можно гулять всем"),
    (35.4, "🟡", "умеренно", "гулять можно, чувствительным людям лучше не подолгу"),
    (55.4, "🟠", "вредно для чувствительных", "пожилым, детям и людям с больными лёгкими лучше сократить прогулки"),
    (125.4, "🔴", "вредно", "всем лучше меньше быть на улице"),
    (225.4, "🟣", "очень вредно", "лучше остаться дома и закрыть окна"),
    (float("inf"), "🟤", "опасно", "не выходить без необходимости"),
]
SCALE = (
    "Шкала PM2.5 (мкг/м³ за сутки, EPA):\n"
    "🟢 0-9 чисто\n"
    "🟡 9-35 умеренно\n"
    "🟠 35-55 вредно для чувствительных\n"
    "🔴 55-125 вредно\n"
    "🟣 125-225 очень вредно\n"
    "🟤 225+ опасно"
)


def level(pm25):
    for upper, icon, name, advice in LEVELS:
        if pm25 <= upper:
            return icon, name, advice


def telegram(token, method, **params):
    r = requests.post(API.format(token=token, method=method), data=params, timeout=30)
    r.raise_for_status()
    return r.json()["result"]


def print_chat_ids(token):
    # the bot can't write first, so it finds you by the messages you sent to it
    updates = telegram(token, "getUpdates")
    if not updates:
        print("no messages yet, write anything to your bot in Telegram and run again")
    for u in updates:
        chat = u["message"]["chat"]
        print(f"chat id: {chat['id']}  ({chat.get('first_name') or chat.get('title')})")


def make_message():
    download_weather.main()
    download_airgradient.main()
    build_dataset.main()

    _, daily = load_data()
    today, tomorrow, pred = predict_tomorrow(load_model(), daily)
    icon, name, advice = level(pred)
    return (f"{SCALE}\n\n"
            f"Алматы, прогноз PM2.5 на {tomorrow:%d.%m}: {pred:.0f} мкг/м³\n"
            f"{icon} {name}: {advice}\n"
            f"Сегодня было {today.pm25_mean24h:.0f}. Норма ВОЗ за день: 15.")


def main():
    load_dotenv()
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise SystemExit("Set TELEGRAM_TOKEN in .env (see the top of this file)")
    if "--chat-id" in sys.argv:
        print_chat_ids(token)
        return

    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not chat_id:
        raise SystemExit("Set TELEGRAM_CHAT_ID in .env, run with --chat-id to find it")
    text = make_message()
    print(text)
    telegram(token, "sendMessage", chat_id=chat_id, text=text)
    print("sent")


if __name__ == "__main__":
    main()
