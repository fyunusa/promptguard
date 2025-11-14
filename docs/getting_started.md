# Getting Started with PromptGuard

## Installation

```bash
pip install promptguard[all]
```

## Basic Usage

### Simple Execution

```python
import asyncio
from promptguard import PromptChain

async def main():
    chain = PromptChain(models=["anthropic/claude-3-5-sonnet"])
    result = await chain.execute("What is machine learning?")
    print(result.response)

asyncio.run(main())
```

### With Type-Safe Responses

```python
from pydantic import BaseModel, Field
from promptguard import PromptChain

class Answer(BaseModel):
    content: str = Field(description="The answer")
    confidence: float = Field(description="Confidence score 0-1")

async def main():
    chain = PromptChain(
        models=["anthropic/claude-3-5-sonnet"],
        response_schema=Answer
    )
    result = await chain.execute("What is AI?")
    print(f"Answer: {result.response.content}")
    print(f"Confidence: {result.response.confidence}")

asyncio.run(main())
```

### With Caching

```python
from promptguard import PromptChain, CacheBackend

async def main():
    chain = PromptChain(
        models=["anthropic/claude-3-5-sonnet"],
        cache=CacheBackend.memory(),
        cache_ttl=3600
    )
    
    # First call - hits API
    result1 = await chain.execute("Question 1", cache_key="q1")
    print(f"Cached: {result1.metadata.cached}")  # False
    
    # Second call - from cache
    result2 = await chain.execute("Question 1", cache_key="q1")
    print(f"Cached: {result2.metadata.cached}")  # True

asyncio.run(main())
```

### With Multiple Models (Fallback)

```python
from promptguard import PromptChain

async def main():
    chain = PromptChain(
        models=[
            "anthropic/claude-3-5-sonnet",
            "openai/gpt-4o",
            "groq/llama-70b"
        ],
        strategy="cascade",
        max_retries=3
    )
    
    result = await chain.execute("Complex question...")
    print(f"Used model: {result.metadata.model_used}")
    print(f"Attempts: {result.metadata.attempts}")

asyncio.run(main())
```

### With Validation

```python
from promptguard import PromptChain, validators

async def main():
    chain = PromptChain(
        models=["anthropic/claude-3-5-sonnet"],
        validators=[
            validators.length_range(min_chars=50, max_chars=500),
            validators.contains_keywords(["important", "key"]),
            validators.has_citations()
        ],
        validation_mode="strict"  # Retry if validation fails
    )
    
    result = await chain.execute("Summarize: ...")
    print(result.validation_results)

asyncio.run(main())
```

### Batch Processing

```python
from promptguard import PromptChain

async def main():
    chain = PromptChain(models=["anthropic/claude-3-5-sonnet"])
    
    prompts = [
        "Question 1",
        "Question 2",
        "Question 3",
    ]
    
    results = await chain.batch_execute(
        prompts,
        max_concurrent=2,
        show_progress=True
    )
    
    for result in results:
        if result.success:
            print(result.response)

asyncio.run(main())
```

### Streaming

```python
from promptguard import PromptChain

async def main():
    chain = PromptChain(models=["anthropic/claude-3-5-sonnet"])
    
    async for chunk in chain.stream("Write an essay..."):
        print(chunk.delta, end="", flush=True)

asyncio.run(main())
```

## Cost Tracking

```python
async def main():
    chain = PromptChain(models=["anthropic/claude-3-5-sonnet"])
    result = await chain.execute("Expensive query...")
    
    metadata = result.metadata
    print(f"Tokens: {metadata.tokens_used}")
    print(f"Cost: ${metadata.estimated_cost:.4f}")
    print(f"Time: {metadata.execution_time_ms:.0f}ms")

asyncio.run(main())
```

## Next Steps

- Read the [API Reference](./api_reference.md)
- Check out [Examples](../examples/)
- Learn about [Advanced Features](./advanced.md)
