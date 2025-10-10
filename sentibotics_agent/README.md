### Sentibotics Agent

A minimal, fast LangChain-based pipeline to process oncology drug posts with batched, concurrent Azure OpenAI sentiment scoring.

#### Install

```bash
pip install -e ./sentibotics_agent
```

Set environment variables for Azure:

```bash
export AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.openai.azure.com"
export AZURE_OPENAI_API_KEY="<your-key>"
export AZURE_OPENAI_API_VERSION="2024-08-01-preview"
export AZURE_OPENAI_DEPLOYMENT="<your-deployment>"
```

#### Run

Dry run (no LLM calls):

```bash
sentibotics-agent path/to/input.json --dry-run -o output.csv
```

With Azure sentiment:

```bash
sentibotics-agent path/to/input.json -o output.csv --batch-size 64 --concurrency 8
```

Input JSON should be an array of documents like:

```json
{
  "post_id": "p0001",
  "date": "2025-08-28",
  "subreddit": "r/CancerSupport",
  "title": "Is appetite loss normal when on Keytruda?",
  "text": "After 3 weeks of Keytruda, I started experiencing appetite loss.",
  "author": "user_438",
  "upvotes": 55,
  "num_comments": 3
}
```

#### Notes
- Uses async concurrency and batching for low latency across 1,000+ docs.
- CPU-only fields (cleaning, keyword extraction, topic cluster) are computed without LLM.
- Sentiment is parsed into [0,1] with safe fallback of 0.5.
