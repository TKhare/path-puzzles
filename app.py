"""WSGI entrypoint for Render's default `gunicorn app:app`.
The Flask app lives in webapp/app.py; re-export it here at the repo root."""
from webapp.app import app  # noqa: F401
