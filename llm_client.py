import os
import json
import logging
from typing import Optional
from groq import Groq

logger = logging.getLogger("AgentLogger")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "Add-your-api-key-here")
GROQ_MODEL = "qwen/qwen3.6-27b"
# Initialize Groq Client with timeouts and retries
client = Groq(
    api_key=GROQ_API_KEY,
    timeout=15.0,
    max_retries=2
)


class GroqCallError(Exception):
    """Raised when Groq API fails or returns unusable output."""
    pass


def call_groq(system_prompt: str, user_prompt: str, json_schema: Optional[dict] = None) -> dict:
    """
    Calls Groq Cloud API using standard json_object mode.
    Injects JSON schema into system prompt for strict formatting.
    """
    try:
        effective_system_prompt = system_prompt
        if json_schema:
            effective_system_prompt += (
                f"\n\nYou MUST return a valid JSON object strictly matching this schema:\n"
                f"{json.dumps(json_schema, indent=2)}"
            )

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=5000
        )

        content = response.choices[0].message.content
        if not content:
            raise GroqCallError("Empty content returned from Groq model")

        return json.loads(content)

    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        raise GroqCallError(f"Error communicating with Groq Cloud: {e}") from e