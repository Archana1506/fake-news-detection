import csv
import io
import logging
import os
import pickle
import sqlite3
from datetime import datetime

from flask import Flask, make_response, redirect, render_template, request, url_for

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Model ----------
with open(os.path.join(BASE_DIR, "news_model.pkl"), "rb") as f:
    model = pickle.load(f)

LABELS = {"FAKE": "Fake News", "0": "Fake News", "REAL": "Real News", "1": "Real News"}

# ---------- Database ----------
# On Render: set DATABASE_URL (Neon/Supabase Postgres). On your PC: uses a local SQLite file.
DATABASE_URL = os.environ.get("DATABASE_URL")
USE_PG = bool(DATABASE_URL)
if USE_PG:
 import psycopg2
PH = "%s" if USE_PG else "?"


def get_db():
    if USE_PG:
        return psycopg2.connect(DATABASE_URL)
    return sqlite3.connect(os.path.join(BASE_DIR, "history_local.db"))


def run(sql, params=(), fetch=False):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall() if fetch else None
        conn.commit()
        return rows
    finally:
        conn.close()


def init_db():
    pk = "SERIAL PRIMARY KEY" if USE_PG else "INTEGER PRIMARY KEY AUTOINCREMENT"
    run(f"""
        CREATE TABLE IF NOT EXISTS history (
            id {pk},
            news TEXT,
            prediction TEXT,
            is_fake INTEGER,
            confidence REAL,
            time TEXT
        )
    """)


def save_item(news, prediction, is_fake, confidence):
    run(
        f"INSERT INTO history (news, prediction, is_fake, confidence, time) "
        f"VALUES ({PH},{PH},{PH},{PH},{PH})",
        (news, prediction, int(is_fake), confidence,
         datetime.now().strftime("%d-%m-%Y %H:%M:%S")),
    )


def load_history(limit=500):
    rows = run(
        f"SELECT news, prediction, is_fake, confidence, time "
        f"FROM history ORDER BY id DESC LIMIT {int(limit)}",
        fetch=True,
    )
    return [
        {"news": r[0], "prediction": r[1], "is_fake": bool(r[2]),
         "confidence": r[3], "time": r[4]}
        for r in rows
    ]


def load_counts():
    row = run("SELECT COUNT(*), SUM(is_fake) FROM history", fetch=True)[0]
    total, fake = int(row[0] or 0), int(row[1] or 0)
    return total, fake, total - fake


init_db()

# ---------- Helpers ----------
def classify(text):
    """Return (label, confidence_percent_or_None)."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        idx = int(probs.argmax())
        raw = str(model.classes_[idx]).upper()
        confidence = round(float(probs[idx]) * 100, 2)
    else:
        raw = str(model.predict([text])[0]).upper()
        confidence = None
    return LABELS.get(raw, raw), confidence


def csv_safe(value):
    value = "" if value is None else str(value)
    return "'" + value if value[:1] in ("=", "+", "-", "@") else value


# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = confidence = None
    news_text = ""

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        body = request.form.get("news", "").strip()
        news_text = f"{title} {body}".strip()

        if not news_text:
            prediction = "Please enter some news."
        else:
            try:
                prediction, confidence = classify(news_text)
                save_item(news_text, prediction, prediction == "Fake News", confidence)
            except Exception:
                app.logger.exception("Prediction or save failed")
                prediction = "Sorry, something went wrong while analysing that text."

    total, fake, real = load_counts()
    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        news_text=news_text,
        history=load_history(10),
        total_count=total,
        fake_count=fake,
        real_count=real,
    )


@app.route("/history")
def history():
    return render_template("history.html", history=load_history())


@app.route("/clear_history", methods=["POST"])
def clear_history():
    run("DELETE FROM history")
    return redirect(url_for("history"))


@app.route("/download", methods=["GET", "POST"])
def download():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["News", "Prediction", "Confidence (%)", "Date"])
    for item in load_history():
        writer.writerow([
            csv_safe(item["news"]),
            item["prediction"],
            item["confidence"] if item["confidence"] is not None else "",
            item["time"],
        ])
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=news_report.csv"
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    return response


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
