"""Example: Response caching for performance."""
import asyncio
import time
from promptguard import PromptChain, CacheBackend


async def main():
    """Execute with caching enabled."""
    
    # Create chain with memory cache
    chain = PromptChain(
        models=["anthropic/claude-3-5-sonnet"],
        cache=CacheBackend.memory(),
        cache_ttl=3600  # 1 hour
    )
    
    prompt = "Define machine learning in one sentence."
    cache_key = "ml_definition_v1"
    
    # First execution - hits API
    print("First execution (cache miss)...")
    start = time.time()
    result1 = await chain.execute(prompt, cache_key=cache_key)
    time1 = time.time() - start
    
    print(f"  Response: {result1.response}")
    print(f"  Cached: {result1.metadata.cached}")
    print(f"  Time: {time1:.2f}s")
    print(f"  Cost: ${result1.metadata.estimated_cost:.4f}")
    
    # Second execution - from cache (instant)
    print("\nSecond execution (cache hit)...")
    start = time.time()
    result2 = await chain.execute(prompt, cache_key=cache_key)
    time2 = time.time() - start
    
    print(f"  Response: {result2.response}")
    print(f"  Cached: {result2.metadata.cached}")
    print(f"  Time: {time2:.4f}s (instant!)")
    print(f"  Cost: ${result2.metadata.estimated_cost:.4f} (free!)")
    
    print(f"\n⚡ Performance improvement: {time1/time2:.0f}x faster")
    print(f"💰 Cost saved: ${result1.metadata.estimated_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
