from __future__ import annotations

import argparse
import os

from .llm import build_sentiment_chain
from .pipeline import process_documents, ProcessingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentibotics Agent CLI")
    parser.add_argument("input", help="Path to input JSON array file")
    parser.add_argument("--output", "-o", help="Output CSV path", default=None)
    parser.add_argument("--deployment", help="Azure OpenAI deployment name (overrides env)", default=None)
    parser.add_argument("--api-version", help="Azure OpenAI API version", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls (use 0.5)")

    args = parser.parse_args()

    cfg = ProcessingConfig(
        max_chars=args.max_chars,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
    )

    if args.dry_run:
        sentiment_chain = None
    else:
        sentiment_chain = build_sentiment_chain(
            deployment_name=args.deployment,
            api_version=args.api_version,
        )

    df = process_documents(
        raw_json_path=args.input,
        output_csv_path=args.output,
        sentiment_chain=sentiment_chain,
        config=cfg,
        dry_run=args.dry_run,
    )

    print(f"Processed {len(df)} rows")


if __name__ == "__main__":
    main()
