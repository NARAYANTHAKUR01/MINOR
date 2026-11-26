#!/usr/bin/env python3
"""
Backend Server Launcher for YOLO11m Local API.

Runs FastAPI with Uvicorn on http://0.0.0.0:8000
Interactive OpenAPI documentation at http://localhost:8000/docs
Built-in Frontend UI Dashboard at http://localhost:8000/
"""

import argparse
import sys
import uvicorn

from backend.config import settings


def main():
    parser = argparse.ArgumentParser(description="Start the YOLO11m Local API Server")
    parser.add_argument("--host", type=str, default=settings.api_host, help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=settings.api_port, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print("=" * 65)
    print("  🚀 YOLO11m Local Backend Server Starting")
    print(f"  📡 API Endpoint:     http://localhost:{args.port}/api")
    print(f"  📖 Swagger Docs:     http://localhost:{args.port}/docs")
    print(f"  💻 Frontend UI:      http://localhost:{args.port}/")
    print("=" * 65)
    print("[INFO] Initializing PyTorch & Metal (MPS) runtime. Please wait a moment...")

    try:
        uvicorn.run(
            "backend.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n[INFO] Backend server stopped cleanly by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
