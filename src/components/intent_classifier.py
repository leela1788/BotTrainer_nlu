import json
from src.logging_system import get_logger
from src.exceptions import NLUException

logger = get_logger()


class IntentClassifier:
    def __init__(self, prompt_builder, llm_inference):
        """
        prompt_builder: PromptBuilder instance
        llm_inference: LLMInference instance
        """
        self.prompt_builder = prompt_builder
        self.llm = llm_inference

    def _parse_response(self, response: str) -> dict:
        """
        Safely parse and validate LLM JSON output
        """
        try:
            response = response.strip()

            # Guard: empty response
            if not response:
                raise ValueError("Empty response from LLM")

            # Extract JSON part safely
            start = response.find("{")
            end = response.rfind("}") + 1

            if start == -1 or end == -1:
                raise ValueError("No JSON object found in response")

            json_text = response[start:end]

            parsed = json.loads(json_text)

            if "intent" not in parsed:
                raise ValueError("Missing 'intent' field")

            if "confidence" not in parsed:
                parsed["confidence"] = 0.0

            logger.info(
                f"Intent parsed: {parsed['intent']} | Confidence: {parsed['confidence']}"
            )

            return parsed

        except Exception as e:
            logger.error(f"Invalid LLM response: {response}")
            raise NLUException(e)

    def predict(self, user_query: str) -> dict:
        """
        Predict intent for a user query
        """
        try:
            logger.info(f"Running intent classification for: {user_query}")

            prompt = self.prompt_builder.build_prompt(user_query)
            llm_response = self.llm.generate(prompt)

            result = self._parse_response(llm_response)

            logger.info("Intent classification successful")
            return result

        except Exception as e:
            logger.error("Intent classification failed")
            raise NLUException(e)
