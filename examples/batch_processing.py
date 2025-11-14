"""Example: Batch processing with concurrency control."""
import asyncio
from promptguard import PromptChain


async def main():
    """Process multiple prompts efficiently."""
    
    chain = PromptChain(
        models=["anthropic/claude-3-5-sonnet"],
        max_retries=2
    )
    
    # Documents to analyze
    documents = [
        "Document about AI safety",
        "Document about machine learning ethics",
        "Document about responsible AI",
        "Document about AI governance",
        "Document about AI regulation",
    ]
    
    # Create prompts
    prompts = [f"Summarize this: {doc}" for doc in documents]
    
    # Execute in parallel with max 2 concurrent requests
    print(f"Processing {len(prompts)} documents...")
    results = await chain.batch_execute(
        prompts,
        max_concurrent=2,
        show_progress=True
    )
    
    # Aggregate results
    total_cost = 0
    total_tokens = 0
    
    for i, result in enumerate(results):
        if result.success:
            print(f"\n📄 Document {i+1}:")
            print(f"   Summary: {result.response[:100]}...")
            total_cost += result.metadata.estimated_cost
            total_tokens += result.metadata.tokens_used.get('total', 0)
        else:
            print(f"\n Document {i+1} failed: {result.error}")
    
    print(f"\n Batch Results:")
    print(f"   Total cost: ${total_cost:.4f}")
    print(f"   Total tokens: {total_tokens}")
    print(f"   Success rate: {sum(1 for r in results if r.success)}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
