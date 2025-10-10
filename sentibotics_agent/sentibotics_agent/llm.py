from __future__ import annotations

import os
import re
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI


SYSTEM_PROMPT = "You are an assistant that analyzes sentiment."
USER_PROMPT = (
    "Analyze the sentiment of the following text and provide only a numeric "
    "score between 0 (negative) and 1 (positive). If unsure, return 0.5.\n{text}"
)


def build_sentiment_chain(
    *,
    deployment_name: Optional[str] = None,
    api_version: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 5,
) -> AzureChatOpenAI:
    """
    Build a minimal sentiment analysis chain using AzureChatOpenAI.

    Environment variables used if args are None:
      - AZURE_OPENAI_ENDPOINT
      - AZURE_OPENAI_API_KEY
      - AZURE_OPENAI_API_VERSION
      - AZURE_OPENAI_DEPLOYMENT (aka model deployment)
    """
    deployment = deployment_name or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT (deployment name) is required")

    api_ver = api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    llm = AzureChatOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=deployment,
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version=api_ver,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=60,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])

    chain = prompt | llm | StrOutputParser()

    def parse_score(text: str) -> float:
        match = re.search(r"\d*\.?\d+", text)
        if not match:
            return 0.5
        score = float(match.group())
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score

    # Attach a parser method for reuse
    chain.parse_score = parse_score  # type: ignore[attr-defined]
    return chain
