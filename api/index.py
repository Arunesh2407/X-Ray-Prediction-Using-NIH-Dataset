"""Minimal Vercel-compatible Python entrypoint.

This repository's main UI is a Streamlit app in `app.py`, which is not
directly runnable as a long-lived service on Vercel. This function provides a
valid WSGI `app` callable so Vercel builds and deployments succeed.
"""

from __future__ import annotations

import json
from typing import Iterable


def app(environ: dict, start_response) -> Iterable[bytes]:
    """WSGI app used by Vercel's Python runtime."""
    payload = {
        "status": "ok",
        "message": (
            "Vercel Python entrypoint is working. "
            "This project's interactive UI is provided via Streamlit (app.py)."
        ),
        "path": environ.get("PATH_INFO", "/"),
    }

    body = json.dumps(payload).encode("utf-8")
    start_response(
        "200 OK",
        [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ],
    )
    return [body]
