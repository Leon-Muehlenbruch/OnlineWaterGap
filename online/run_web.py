#!/usr/bin/env python3
"""
Start the ReWaterGAP web interface locally.

Usage:
    python run_web.py

Then open http://localhost:8000 in your browser.
"""

import subprocess
import sys


def main():
    # Check if uvicorn is installed
    try:
        import uvicorn
    except ImportError:
        print("Installing required packages (fastapi, uvicorn)...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "fastapi", "uvicorn[standard]", "python-multipart",
        ])
        import uvicorn

    print("\n" + "=" * 50)
    print("  ReWaterGAP Web Interface")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 50 + "\n")

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
