"""Example: Basic prompt execution with automatic retry."""
import asyncio
from promptguard import PromptChain


async def main():
    """Execute a simple prompt with fallback models."""
    
    # Create chain with multiple models
    # Will try models in order until one succeeds
    chain = PromptChain(
        models=[
            "anthropic/claude-3-5-sonnet",
            "openai/gpt-4o",
            "groq/llama-70b"
        ],
        strategy="cascade",
        max_retries=3,
        retry_delay="exponential",  # 1s, 2s, 4s, 8s...
        timeout=30.0
    )
    
    # Execute prompt
    result = await chain.execute(
        "What are the top 3 benefits of machine learning?"
    )
    
    # Access results
    print(f"Success!")
    print(f"Response: {result.response}")
    print(f"Model used: {result.metadata.model_used}")
    print(f"Attempts: {result.metadata.attempts}")
    print(f"Execution time: {result.metadata.execution_time_ms:.0f}ms")
    print(f"Cost: ${result.metadata.estimated_cost:.4f}")
    print(f"Tokens: {result.metadata.tokens_used}")


if __name__ == "__main__":
    asyncio.run(main())
