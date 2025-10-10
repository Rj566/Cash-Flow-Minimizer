from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm

DRUG_LIST = ["Keytruda", "Opdivo", "Tagrisso", "Herceptin", "Ibrance"]
SIDE_EFFECTS_LIST = [
    "fatigue", "nausea", "rash", "hair loss",
    "joint pain", "appetite loss", "fever", "cough"
]


@dataclass
class ProcessingConfig:
    max_chars: int = 2000
    batch_size: int = 64
    concurrency: int = 8


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_drug_mentions(text: str) -> Optional[str]:
    text_lower = text.lower()
    drugs_found = [drug for drug in DRUG_LIST if drug.lower() in text_lower]
    return ", ".join(drugs_found) if drugs_found else None


def extract_side_effects(text: str) -> Optional[str]:
    text_lower = text.lower()
    effects_found = [effect for effect in SIDE_EFFECTS_LIST if effect.lower() in text_lower]
    return ", ".join(effects_found) if effects_found else None


def get_topic_cluster(text: str) -> str:
    side_effect_keywords = [
        "fatigue", "nausea", "rash", "hair loss", "joint pain", "appetite loss", "fever", "cough"
    ]
    positive_keywords = ["improved", "better", "helped", "stable"]
    med_change_keywords = ["switched", "changed", "alternative"]

    text_lower = text.lower()
    if any(k in text_lower for k in side_effect_keywords):
        return "Side Effects"
    if any(k in text_lower for k in positive_keywords):
        return "Positive Outcome"
    if any(k in text_lower for k in med_change_keywords):
        return "Medication Change"
    return "Other"


def _combine(entry: Dict[str, Any]) -> str:
    title = entry.get("title", "")
    text = entry.get("text", "")
    return f"{title} {text}".strip()


def process_documents(
    *,
    raw_documents: Iterable[Dict[str, Any]] | None = None,
    raw_json_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
    sentiment_chain=None,
    config: ProcessingConfig | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Process up to thousands of documents efficiently. Supports batching and concurrency.

    - raw_documents: iterable of dicts with keys including post_id, date, subreddit, title, text
    - raw_json_path: file path to JSON array of documents
    - output_csv_path: where to write CSV (optional)
    - sentiment_chain: LangChain chain returning a numeric score string; must expose parse_score
    - dry_run: if True, skip LLM calls and return 0.5 for all
    """
    if (raw_documents is None) == (raw_json_path is None):
        raise ValueError("Provide exactly one of raw_documents or raw_json_path")

    cfg = config or ProcessingConfig()

    if raw_json_path is not None:
        with open(raw_json_path, "r", encoding="utf-8") as f:
            raw_documents = json.load(f)

    assert raw_documents is not None

    rows: List[Dict[str, Any]] = []

    # Prepare batched inputs for LLM to minimize per-call overhead
    # We'll send one message per doc, but execute asyncronously in parallel using .abatch
    # Truncate inputs to keep prompts small
    inputs: List[Dict[str, str]] = []
    combined_texts: List[str] = []

    for entry in raw_documents:
        combined = _combine(entry)
        combined_texts.append(combined)
        truncated = combined[: cfg.max_chars]
        inputs.append({"text": truncated})

    # Compute static fields first (CPU-only, vectorized-ish)
    for i, entry in enumerate(raw_documents):
        combined = combined_texts[i]
        text_clean = clean_text(combined)
        drug_mentioned = extract_drug_mentions(combined)
        side_effects = extract_side_effects(combined)
        topic_cluster = get_topic_cluster(combined)

        rows.append({
            "post_id": entry.get("post_id"),
            "date": entry.get("date"),
            "subreddit": entry.get("subreddit"),
            "text_clean": text_clean,
            "drug_mentioned": drug_mentioned,
            "side_effects": side_effects,
            "topic_cluster": topic_cluster,
        })

    # Sentiment in batches with concurrency
    sentiments: List[float] = []
    if dry_run:
        sentiments = [0.5] * len(inputs)
    else:
        if sentiment_chain is None:
            raise ValueError("sentiment_chain is required when dry_run is False")

        # Execute in batches to respect rate limits and minimize latency
        # Using async abatch for parallel requests
        from math import ceil
        import asyncio

        async def run_batches():
            total = len(inputs)
            batch_sz = max(1, cfg.batch_size)
            num_batches = ceil(total / batch_sz)
            results: List[float] = [0.5] * total

            sem = asyncio.Semaphore(cfg.concurrency)

            async def run_single(idx: int, payload: Dict[str, str]):
                async with sem:
                    try:
                        text = await sentiment_chain.ainvoke(payload)
                        score = sentiment_chain.parse_score(text)
                    except Exception:
                        score = 0.5
                    results[idx] = score

            for b in tqdm(range(num_batches), desc="Sentiment batches"):
                start = b * batch_sz
                end = min((b + 1) * batch_sz, total)
                tasks = [asyncio.create_task(run_single(i, inputs[i])) for i in range(start, end)]
                await asyncio.gather(*tasks)
            return results

        sentiments = asyncio.run(run_batches())

    # Merge sentiments
    for i, score in enumerate(sentiments):
        rows[i]["sentiment_score"] = score

    df = pd.DataFrame(rows)
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
    return df
