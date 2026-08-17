"""
Run Script - Blood Cell Detection & Classification Web Demo
"""
import os
import sys
import uvicorn
from pathlib import Path


def main():
    # Make sure app path is reachable
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)

    print("[SYSTEM] Starting HemoAI Medical Research Web Server...")
    # Run FastAPI app via Uvicorn
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
