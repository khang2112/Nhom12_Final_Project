"""Rebuild deterministic outputs; add --crawl to refresh the live source."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crawl', action='store_true')
    args = parser.parse_args()
    steps = (['00'] if args.crawl else []) + ['01', '02', '09', '10', '11', '12', '13', '14', '03', '04']
    for prefix in steps:
        script = next((ROOT/'src').glob(prefix+'_*.py'))
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q'], cwd=ROOT, check=True)
