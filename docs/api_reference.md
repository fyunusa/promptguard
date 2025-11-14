# PromptGuard API Reference

## Core Classes

### PromptChain

Main orchestrator for LLM execution.

```python
class PromptChain:
    def __init__(
        self,
        models: List[str],
        strategy: Literal["cascade", "fastest", "cheapest", "parallel"] = "cascade",
        max_retries: int = 3,
        retry_delay: Union[float, Literal["exponential", "fibonacci"]] = "exponential",
        timeout: float = 30.0,
        response_schema: Optional[Type[BaseModel]] = None,
        validation_mode: Literal["strict", "lenient"] = "strict",
        validators: Optional[List[Validator]] = None,
        cache: Optional[CacheBackend] = None,
        cache_ttl: int = 3600,
        log_level: LogLevel = LogLevel.INFO,
        on_success: Optional[Callable] = None,
        on_retry: Optional[Callable] = None,
        on_failure: Optional[Callable] = None,
        **provider_kwargs
    )
```

#### Methods

**`execute()`** - Execute prompt with retries and validation

```python
async def execute(
    prompt: Union[str, List[dict]],
    context: Optional[dict] = None,
    temperature: float = 0.01,
    max_tokens: int = 4096,
    cache_key: Optional[str] = None,
    metadata: Optional[dict] = None,
    **kwargs
) -> PromptResult
```

Returns: `PromptResult` object with response and metadata

**`stream()`** - Stream response chunks

```python
async def stream(
    prompt: Union[str, List[dict]],
    context: Optional[dict] = None,
    temperature: float = 0.01,
    max_tokens: int = 4096,
    **kwargs
) -> AsyncIterator[StreamChunk]
```

Yields: `StreamChunk` objects

**`batch_execute()`** - Process multiple prompts

```python
async def batch_execute(
    prompts: List[Union[str, List[dict]]],
    max_concurrent: int = 5,
    show_progress: bool = False,
    fail_fast: bool = False,
    **kwargs
) -> List[PromptResult]
```

Returns: List of `PromptResult` objects

---

### PromptResult

Response object containing the result and metadata.

```python
class PromptResult:
    success: bool
    response: Union[str, BaseModel, None]
    raw_response: Any
    metadata: ExecutionMetadata
    error: Optional[Exception]
    validation_results: Optional[dict]
    
    def to_dict(self) -> dict
    def to_json(self) -> str
```

---

### ExecutionMetadata

Execution information and statistics.

```python
@dataclass
class ExecutionMetadata:
    model_used: str
    provider: str
    attempts: int
    execution_time_ms: float
    tokens_used: Dict[str, int]
    estimated_cost: float
    cached: bool
    cache_key: Optional[str]
    retry_history: List[dict]
    timestamp: datetime
    temperature: Optional[float]
    max_tokens: Optional[int]
```

---

## Cache Backends

### CacheBackend

Base cache interface.

```python
# In-memory cache
cache = CacheBackend.memory()

# Redis cache
cache = CacheBackend.redis("redis://localhost:6379")

# Disk-based cache
cache = CacheBackend.disk(".promptguard_cache")
```

---

## Validators

### Built-in Validators

**`length_range()`** - Validate response length

```python
validators.length_range(min_chars=100, max_chars=5000)
```

**`contains_keywords()`** - Ensure required keywords

```python
validators.contains_keywords(
    keywords=["risk", "mitigation"],
    require_all=True
)
```

**`has_citations()`** - Check for citations

```python
validators.has_citations(required=True)
```

**`no_hallucination()`** - Detect hallucinations

```python
validators.no_hallucination(
    source_docs=documents,
    threshold=0.8
)
```

**`sentiment_check()`** - Validate sentiment

```python
validators.sentiment_check(allowed=["neutral", "positive"])
```

---

## Retry Strategies

### ExponentialBackoff

```python
from promptguard import ExponentialBackoff

strategy = ExponentialBackoff(
    base=1.0,
    max_delay=60.0,
    max_retries=3
)
```

### FibonacciBackoff

```python
from promptguard import FibonacciBackoff

strategy = FibonacciBackoff(max_retries=3)
```

### ConstantDelay

```python
from promptguard import ConstantDelay

strategy = ConstantDelay(delay=2.0, max_retries=3)
```

---

## Exceptions

All exceptions inherit from `PromptGuardError`.

**`PromptExecutionError`** - Execution failed after retries

```python
try:
    result = await chain.execute(prompt)
except PromptExecutionError as e:
    print(f"Attempts: {e.attempts}")
    print(f"Last error: {e.last_error}")
```

**`ValidationError`** - Response validation failed

```python
except ValidationError as e:
    print(f"Validation results: {e.validation_results}")
```

**`RateLimitError`** - Rate limit exceeded

```python
except RateLimitError as e:
    print(f"Retry after: {e.retry_after} seconds")
```

**`ModelNotFoundError`** - Model not available

**`TimeoutError`** - Request timed out

**`ProviderError`** - Provider-specific error

---

## Environment Variables

```bash
# Required for respective providers
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk-...
COHERE_API_KEY=...
GOOGLE_API_KEY=...

# Optional
REDIS_URL=redis://localhost:6379
```

---

## Model Identifiers

Supported models by provider:

**Anthropic:**
- `anthropic/claude-3-5-sonnet`
- `anthropic/claude-3-opus`
- `anthropic/claude-3-sonnet`
- `anthropic/claude-3-haiku`

**OpenAI:**
- `openai/gpt-4o`
- `openai/gpt-4-turbo`
- `openai/gpt-4`
- `openai/gpt-3.5-turbo`

**Groq:**
- `groq/llama-3-70b`
- `groq/mixtral-8x7b`

**Google:**
- `google/gemini-1.5-pro`
- `google/gemini-1.5-flash`

**Cohere:**
- `cohere/command-r-plus`
- `cohere/command-r`
