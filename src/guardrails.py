"""
Guardrails module.
Implements Layer 1 (Rule-based) and Layer 2 (LLM-based) safety checks.
"""

import re
import logging
from typing import Tuple

from .config import MALICIOUS_KEYWORDS
from .llm import get_llm_generator


logger = logging.getLogger(__name__)


class Guardrails:
    """
    Implements multi-layer safety guardrails.
    """

    def __init__(self, malicious_keywords: list = None):

        self.malicious_keywords = malicious_keywords or MALICIOUS_KEYWORDS

        # Precompile regex patterns
        self.keyword_patterns = [
            re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
            for keyword in self.malicious_keywords
        ]

    def layer1_rule_based(self, query: str) -> Tuple[bool, str]:
        """
        Layer 1: Rule-based keyword filter.
        """

        for pattern in self.keyword_patterns:
            if pattern.search(query):
                logger.warning("Rule-based guardrail triggered")
                return False, "Query contains blocked keyword"

        return True, ""

    def layer2_llm_guardrail(self, query: str) -> Tuple[bool, str]:
        """
        Layer 2: LLM safety classification.
        """

        classification_prompt = f"""
You are a security classifier for a customer support chatbot.

Classify the query as SAFE or UNSAFE.

SAFE:
Questions about Tonton platform, subscriptions, login, streaming issues, FAQ.

UNSAFE:
Hacking, illegal activity, bypassing payments, harmful actions.

Return only one word:
SAFE or UNSAFE.

Query: {query}
Answer:
"""

        try:

            generator = get_llm_generator()

            response = generator.generate(
                classification_prompt,
                temperature=0.0,
                max_tokens=5
            )

            if not response:
                logger.warning("Empty response from guardrail LLM")
                return True, ""

            response_clean = response.strip().upper()

            logger.info("Guardrail classification: %s", response_clean)

            if response_clean.startswith("UNSAFE"):
                return False, "Query classified as unsafe by LLM"

            if response_clean.startswith("SAFE"):
                return True, ""

            # Unexpected output → allow query
            logger.warning("Unexpected guardrail output, allowing query")
            return True, ""

        except Exception as e:

            logger.error("LLM guardrail error: %s", e)

            # Fail open instead of blocking everything
            return True, ""

    def check(self, query: str) -> Tuple[bool, str]:
        """
        Run both guardrail layers.
        """

        # Layer 1
        is_safe, reason = self.layer1_rule_based(query)

        if not is_safe:
            return False, reason

        # Layer 2
        is_safe, reason = self.layer2_llm_guardrail(query)

        if not is_safe:
            return False, reason

        return True, ""


# ---------------------------------------------------
# Global singleton
# ---------------------------------------------------

_guardrails = None


def get_guardrails() -> Guardrails:

    global _guardrails

    if _guardrails is None:
        _guardrails = Guardrails()

    return _guardrails


def check_guardrails(query: str) -> Tuple[bool, str]:
    """
    Convenience function to run guardrails.
    """

    guardrails = get_guardrails()

    return guardrails.check(query)