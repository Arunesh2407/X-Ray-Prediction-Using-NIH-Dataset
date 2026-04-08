"""Root Python entrypoint for hosts that auto-detect `main.py`.

This forwards to the Vercel serverless callable in `api/index.py`.
"""

from __future__ import annotations

from api.index import app
