"""Wait for DB, synchronize vectors, then serve the Gradio app."""
import os
import subprocess
import sys
import time
from pathlib import Path
from vector_store import client

ROOT = Path(__file__).resolve().parents[1]


def main():
    for attempt in range(60):
        try:
            client().heartbeat()
            break
        except Exception:
            if attempt == 59:
                raise
            time.sleep(2)
    subprocess.run([sys.executable, str(ROOT/'src/04_chromadb_vector.py')], check=True, cwd=ROOT)
    subprocess.run(
    [sys.executable, str(ROOT / 'ui/app.py')],
    check=True,
    cwd=ROOT,
)


if __name__ == '__main__':
    main()
