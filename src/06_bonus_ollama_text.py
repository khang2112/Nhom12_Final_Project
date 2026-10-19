# REFERENCE DATA URL:
# https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv
# OLLAMA OFFICIAL API: https://docs.ollama.com/api/generate
"""Optional bonus: generate clearly synthetic, unstructured text with local Ollama."""

from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3.2"
BATCHES = 12


def main():
    """Request 12 batches and save every nonempty returned text line."""
    results = []

    for batch in range(BATCHES):
        prompt = (
            f"Write five original, fictional and clearly synthetic short policy-observation notes "
            f"for a historical education-data project. Batch ID: {batch}. "
            f"One line per note; NO real-country names, NO invented factual statistics. "
            "These are imagined scenarios, not measured research data."
        )
        response = requests.post(
            ENDPOINT,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.8},
            },
            timeout=180,
        )
        response.raise_for_status()
        raw = response.json()["response"]

        for line_no, line in enumerate(raw.splitlines(), start=1):
            if line.strip():
                results.append(
                    {
                        "batch": batch,
                        "line": line_no,
                        "generated_text": line.strip(),
                        "synthetic": True,
                        "generator": MODEL,
                    }
                )

    df = pd.DataFrame(results)
    assert len(df) > 0
    df.to_csv(ROOT / "data/synthetic/bonus_llm_unstructured_text.csv", index=False)
    print(f"Saved {len(df)} actual Ollama generated text lines")


if __name__ == "__main__":
    main()
