"""Dump the wikitext-2-raw-v1 test split to a plain text file.

llama-perplexity reads a flat text file, not a HF dataset, so this is a
one-time prep step shared by every GGUF checkpoint's perplexity run.
"""

from pathlib import Path

from datasets import load_dataset
from loguru import logger

OUT = Path(__file__).parent / "data" / "wikitext-2-raw-test.txt"


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    OUT.write_text(text)
    logger.info(f"wrote {OUT} ({len(text):,} chars)")
