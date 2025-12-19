from typing import List
from src.logging_system import get_logger
from src.exceptions import NLUException

logger = get_logger()


class PromptBuilder:
    def __init__(self, intents: List[dict]):
        """
        intents: list of dicts with keys ['intent', 'examples']
        """
        self.intents = intents
        self.intent_names = [i["intent"] for i in intents]

    def _build_intent_section(self) -> str:
        """
        Creates intent definitions block for grounding the LLM
        """
        intent_text = ""
        for intent in self.intents:
            examples = "\n".join(
                [f"- {ex}" for ex in intent.get("examples", [])[:3]]
            )
            intent_text += f"""
Intent: {intent['intent']}
Example Utterances:
{examples}
"""
        return intent_text.strip()

    def build_prompt(self, user_query: str) -> str:
        """
        Builds the final prompt for intent classification
        """
        try:
            logger.info("Building prompt for user query")

            intent_definitions = self._build_intent_section()

            prompt = f"""
You are an intelligent Natural Language Understanding (NLU) system.

Your task is to:
1. Understand the user query.
2. Classify it into ONE of the predefined intents.
3. Respond ONLY in valid JSON format.

### Available Intents:
{', '.join(self.intent_names)}

### Intent Definitions and Examples:
{intent_definitions}

### User Query:
"{user_query}"

### Output Format (STRICT):
Return ONLY a JSON object in the following format:

{{
  "intent": "<predicted_intent_name>",
  "confidence": <confidence_score_between_0_and_1>
}}

Rules:
- The intent MUST be one of the available intents.
- Confidence should reflect how sure you are.
- Do NOT add explanations or extra text.
- Output ONLY JSON.
"""

            logger.info("Prompt built successfully")
            return prompt.strip()

        except Exception as e:
            logger.error("Failed to build prompt")
            raise NLUException(e)
