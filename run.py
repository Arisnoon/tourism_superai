from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
ENV_FILE = ROOT_DIR / ".env"
BACKEND_ENTRY = ROOT_DIR / "agentic_api.py"


def _load_env_file(env_path: Path = ENV_FILE) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Agentic FastAPI backend.")
    parser.add_argument("--host", default="127.0.0.1", help="Backend bind host")
    parser.add_argument("--port", type=int, default=8000, help="Backend bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    return parser


def main() -> None:
    _load_env_file()
    args = _build_parser().parse_args()
    if not BACKEND_ENTRY.exists():
        raise SystemExit(f"Backend entry not found: {BACKEND_ENTRY}")
    try:
        uvicorn = importlib.import_module("uvicorn")
    except ImportError as exc:
        raise SystemExit("Missing dependency 'uvicorn'. Install with: pip install uvicorn") from exc

    print(f"[run.py] Starting backend at http://{args.host}:{args.port}")
    uvicorn.run("agentic_api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
