# main.py
import sys
import asyncio
from loguru import logger

from config import parse_arguments
from llm_agents import initialize_agents
from orchestrator import run_labeling_process

async def main():
    """
    The main entry point for the LLM labeling application.
    """
    # 1. Get user settings from command line
    args = parse_arguments()

    if not args.openai and not args.local:
        logger.error("You must specify at least one model to use with --openai or --local.")
        sys.exit(1)

    # 2. Initialize the required LLM agents
    logger.info("--- Initializing LLM Agents ---")
    configured_agents = initialize_agents(
        openai_models=args.openai,
        local_models=args.local
    )

    # 3. Run the main processing logic
    if configured_agents:
        logger.info(f"\n--- Starting Labeling Process (Voting Scheme: {args.voting_scheme}) ---")
        # The output path is no longer passed here
        await run_labeling_process(
            input_path=args.input,
            agents=configured_agents,
            voting_scheme=args.voting_scheme
        )
    else:
        logger.error("\nHalting: Process cannot start as no agents were successfully initialized.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
