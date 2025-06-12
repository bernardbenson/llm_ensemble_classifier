# llm_agents.py
import json
from typing import List, Dict, Any, Tuple
from openai import AsyncOpenAI
from loguru import logger

import config

def get_system_prompt(prompt_key: str = "default") -> str:
    """
    Constructs and returns the system prompt for the LLM agents based on a key from config.
    Falls back to the 'default' prompt if the key is not found.
    """
    prompt = config.PROMPT_CONFIG.get(prompt_key)
    if not prompt:
        logger.warning(f"Prompt key '{prompt_key}' not found in PROMPT_CONFIG. Falling back to 'default' prompt.")
        return config.PROMPT_CONFIG["default"]
    
    return prompt

def _parse_model_string(model_string: str) -> Tuple[str, str]:
    """
    Parses 'model_name:prompt_key' format, returning (model_name, prompt_key).
    It robustly handles model names that may contain colons, like Ollama's 'llama3:8b'.
    """
    # It assumes the prompt key is the part after the *last* colon,
    # and only if that part exists as a key in the PROMPT_CONFIG.
    if ":" in model_string:
        parts = model_string.rsplit(":", 1)
        potential_model = parts[0]
        potential_key = parts[1]
        if potential_key in config.PROMPT_CONFIG:
            return potential_model, potential_key

    # If no valid prompt key is found after a colon, or no colon exists,
    # the whole string is the model name and the prompt is 'default'.
    return model_string, "default"

def initialize_agents(openai_models: List[str], local_models: List[str]) -> List[Dict[str, Any]]:
    """
    Initializes and returns a list of configured LLM agents.
    Each agent is a dictionary containing an AsyncOpenAI client, model details, and a prompt key.
    It supports specifying prompts via the 'model_name:prompt_key' syntax.
    """
    agents = []
    
    def add_agent(client: AsyncOpenAI, full_model_str: str, provider_name: str) -> None:
        """Helper to parse model string, create a unique agent name, and add to the list."""
        model_name, prompt_key = _parse_model_string(full_model_str)
        
        # Create a unique agent name to distinguish between the same model with different prompts.
        # e.g., 'openai_gpt-4o-mini_creative'
        sanitized_model_name = model_name.replace('-', '_').replace('.', '_').replace(':', '_').replace('/', '_')
        agent_name = f"{provider_name}_{sanitized_model_name}_{prompt_key}"

        agents.append({
            "client": client,
            "model_name": model_name,
            "prompt_key": prompt_key,
            "name": agent_name,
        })
        logger.info(f"Successfully initialized {provider_name} agent for model: '{model_name}' with prompt: '{prompt_key}'")

    # 1. Initialize OpenAI Agents
    if openai_models and config.OPENAI_API_KEY:
        openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        for model_string in openai_models:
            add_agent(openai_client, model_string, "openai")
    elif openai_models:
        logger.warning("OpenAI models specified, but OPENAI_API_KEY environment variable not found. Skipping.")

    # 2. Initialize Ollama/Local Agents
    if local_models:
        local_client = AsyncOpenAI(
            base_url=config.OLLAMA_BASE_URL,
            api_key="ollama" # Required by the library, but not used by Ollama
        )
        for model_string in local_models:
            add_agent(local_client, model_string, "ollama")

    if not agents:
        logger.error("No LLM agents could be initialized. Please check your configurations and command-line arguments.")
    
    return agents
