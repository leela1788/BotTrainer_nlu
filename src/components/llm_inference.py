import os
from groq import Groq
from dotenv import load_dotenv

from src.logging_system import get_logger
from src.exceptions import NLUException

load_dotenv()
logger = get_logger()


class LLMInference:
    def __init__(self, model, temperature, max_tokens):
        try:
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            self.model = model
            self.temperature = temperature
            self.max_tokens = max_tokens
            logger.info("Groq LLMInference initialized")
        except Exception as e:
            raise NLUException(e)

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise NLUException(e)
