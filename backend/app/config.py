"""Application configuration and middleware setup."""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def setup_cors(app: FastAPI) -> None:
    """Configure CORS middleware for frontend-backend communication."""
    _default_origins = "http://localhost:3000,http://127.0.0.1:3000"
    allowed_origins = [
        o.strip()
        for o in os.environ.get("CORS_ALLOW_ORIGINS", _default_origins).split(",")
        if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
