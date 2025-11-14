"""Example: Streaming responses for real-time output."""
import asyncio
from promptguard import PromptChain


async def main():
    """Stream long-form content."""
    
    chain = PromptChain(
        models=["anthropic/claude-3-5-sonnet"]
    )
    
    prompt = "Write a detailed blog post about the future of AI in healthcare (500+ words)"
    
    print("🔄 Streaming response...\n")
    
    accumulated_text = ""
    async for chunk in chain.stream(prompt):
        # Print each chunk as it arrives (real-time)
        print(chunk.delta, end="", flush=True)
        accumulated_text = chunk.accumulated_text
        
        if chunk.finished:
            print("\n Streaming complete!")
            print(f"Total length: {len(accumulated_text)} characters")


if __name__ == "__main__":
    asyncio.run(main())
