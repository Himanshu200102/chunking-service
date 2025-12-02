#!/usr/bin/env python3
"""Quick smoke test for the local Llama ADK integration."""

import argparse
import asyncio
import os

from google.adk import Agent

import app.agents  # noqa: F401 - registers custom LLMs


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a prompt against the local ADK Llama model")
    parser.add_argument("prompt", help="User prompt to send to the local agent")
    parser.add_argument(
        "--model",
        default="local/llama-8b-gguf",
        help="Model identifier registered with ADK",
    )
    args = parser.parse_args()

    if "LLAMA_MODEL_PATH" not in os.environ:
        raise SystemExit("LLAMA_MODEL_PATH must be set before running this script.")

    agent = Agent(
        name="local_helper",
        model=args.model,
        instruction="You are a helpful assistant that answers concisely.",
    )

    result = await agent.run(args.prompt)
    print(f"[{args.model}] -> {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
