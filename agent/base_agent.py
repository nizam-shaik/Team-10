from __future__ import annotations
from pathlib import Path
"""
Base Agent class for HLD generation agents
"""

# TODO: Import necessary modules for API communication, JSON parsing, and data modeling
# TODO: Implement BaseAgent class with abstract methods for system_prompt and process
# TODO: Implement call_llm() method to make API calls to Gemini with retry logic
# TODO: Implement parse_json_loose() method with multiple fallback strategies for JSON extraction
# TODO: Implement error handling and logging for API failures
# TODO: Add temperature and token parameters for LLM configuration
# TODO: Implement state normalization utilities (normalize_string, normalize_list, etc.)
# TODO: Handle both synchronous and potential asynchronous API calls
# TODO: Add API key management with environment variable loading
# TODO: Implement system prompt caching or dynamic loading strategy
# TODO: Add token counting and cost estimation features
# TODO: Implement rate limiting and backoff strategies for API calls
"""
Base Agent class for HLD generation agents
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ==========================================================
# Utility Models and Constants
# ==========================================================
class LLMResponse(BaseModel):
    """Standardized response model for LLM output."""
    text: str = Field(default="")
    parsed_json: Optional[Dict[str, Any]] = None
    tokens_used: int = 0
    cost_estimate_usd: float = 0.0


# ==========================================================
# BaseAgent Definition - Refactored with LangChain
# ==========================================================
class BaseAgent(ABC):
    """
    Abstract base agent for all HLD generation components - Using LangChain.
    
    Key Features:
      - Uses ChatGoogleGenerativeAI for unified LLM interface
      - Automatic retry logic with exponential backoff (built-in)
      - Structured output parsing with Pydantic models
      - Prompt template management
      - Chain composition with LCEL
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_output_tokens: int = 8192,
        max_retries: int = 5,
        request_timeout: int = 120,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        
        # Log agent initialization
        logger = logging.getLogger(self.__class__.__name__)
        logger.info("=" * 80)
        logger.info(f"{self.__class__.__name__} INITIALIZED (LangChain)")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Max Output Tokens: {max_output_tokens}")
        logger.info(f"Max Retries: {max_retries}")
        logger.info("=" * 80)
        
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not found in environment variables")
        
        # ✅ Initialize LangChain's ChatGoogleGenerativeAI
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_output_tokens,
            max_retries=max_retries,  # Built-in retry logic
            request_timeout=request_timeout,
            google_api_key=api_key,
            convert_system_message_to_human=True,  # For Gemini compatibility
        )
        
        # ✅ Initialize output parsers
        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser()
        
        # ✅ Create prompt template (will be customized by subclasses)
        self.prompt_template = self._create_prompt_template()
        
        # ✅ Create default chain (can be overridden)
        self.chain = self._create_default_chain()

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create a basic prompt template. Subclasses can override."""
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("{system_prompt}"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
    
    def _create_default_chain(self):
        """Create default LCEL chain. Subclasses can override for custom chains."""
        return (
            self.prompt_template
            | self.llm
            | self.str_parser
        )

    # ======================================================
    # Abstract Methods
    # ======================================================
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Base system prompt defining the role and context of the agent."""
        ...

    @abstractmethod
    def process(self, state: Any) -> Dict[str, Any]:
        """Main method each derived agent implements to transform the workflow state."""
        ...

    # ======================================================
    # LLM Invocation - Now using LangChain
    # ======================================================
    def call_llm(self, prompt: str, parse_json: bool = True) -> LLMResponse:
        """
        Make a call to Gemini using LangChain's ChatGoogleGenerativeAI.
        Automatic retry, rate limiting, and error handling are built-in.
        
        Args:
            prompt: The user prompt/input text
            parse_json: Whether to attempt JSON parsing on the response
            
        Returns:
            LLMResponse with text and optional parsed JSON
        """
        logger = logging.getLogger(self.__class__.__name__)
        
        try:
            logger.info(f"Calling LLM via LangChain (model: {self.model_name})")
            
            # ✅ Use LangChain chain for invocation (automatic retry built-in)
            if parse_json:
                # Try to parse as JSON
                json_chain = (
                    self.prompt_template
                    | self.llm
                    | self.json_parser
                )
                
                try:
                    result = json_chain.invoke({
                        "system_prompt": self.system_prompt,
                        "input": prompt
                    })
                    
                    # Successfully parsed as JSON
                    logger.info(f"✓ LLM call successful with JSON parsing")
                    return LLMResponse(
                        text=json.dumps(result, indent=2),
                        parsed_json=result,
                        tokens_used=0,  # LangChain doesn't expose token count directly
                        cost_estimate_usd=0.0
                    )
                except Exception as json_error:
                    logger.warning(f"JSON parsing failed: {json_error}, falling back to string output")
                    # Fall back to string parsing
                    parse_json = False
            
            if not parse_json:
                # Use string parser
                result = self.chain.invoke({
                    "system_prompt": self.system_prompt,
                    "input": prompt
                })
                
                logger.info(f"✓ LLM call successful (text output)")
                
                # Try to extract JSON from text
                parsed = self.parse_json_loose(result)
                
                return LLMResponse(
                    text=result,
                    parsed_json=parsed,
                    tokens_used=0,
                    cost_estimate_usd=0.0
                )
                
        except Exception as e:
            logger.error(f"LLM call failed after retries: {e}")
            raise RuntimeError(f"LLM invocation failed: {e}")

    # ======================================================
    # JSON Parsing Utilities (Kept for backward compatibility)
    # ======================================================
    @staticmethod
    def parse_json_loose(text: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to parse JSON robustly from a model response.
        Tries multiple fallback strategies to extract JSON objects.
        """
        if not text:
            return None

        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback 1: extract text between braces
        if "{" in text and "}" in text:
            candidate = text[text.find("{"): text.rfind("}") + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Fallback 2: manually fix common JSON issues
        cleaned = text.replace("'", '"').replace("\n", " ")
        try:
            return json.loads(cleaned)
        except Exception:
            logging.warning("[BaseAgent] Could not extract valid JSON from model response.")
            return None

    # ======================================================
    # Normalization Utilities
    # ======================================================
    @staticmethod
    def normalize_string(value: Optional[str]) -> str:
        if not value:
            return ""
        return " ".join(value.strip().split())

    @staticmethod
    def normalize_list(values: Optional[List[Any]]) -> List[Any]:
        if not values:
            return []
        if isinstance(values, str):
            return [values.strip()]
        return [v for v in values if v is not None and str(v).strip()]

    @staticmethod
    def normalize_dict(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not data:
            return {}
        return {str(k).strip(): v for k, v in data.items() if k is not None}

    # ======================================================
    # Logging & Caching Utilities
    # ======================================================
    def log_cost(self, response: LLMResponse):
        logging.info(
            f"[LLM] Tokens: {response.tokens_used}, Cost: ${response.cost_estimate_usd:.6f}"
        )

    def cached_prompt_path(self) -> Path:
        """Optional: path for storing system prompts for inspection."""
        base = Path("cache/prompts")
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{self.__class__.__name__}.txt"

    def save_system_prompt(self):
        """Save the system prompt to a cache file (optional for debugging)."""
        path = self.cached_prompt_path()
        path.write_text(self.system_prompt, encoding="utf-8")
        logging.info(f"Saved system prompt: {path}")
