# PromptGuard Architecture

## System Overview

PromptGuard is a production-grade framework for reliable LLM orchestration. It provides:

1. **Provider Abstraction** - Unified interface across multiple LLM providers
2. **Reliability** - Automatic retry logic, fallback chains, and error handling
3. **Validation** - Both schema and semantic response validation
4. **Observability** - Detailed logging, metrics, and tracing
5. **Performance** - Caching, streaming, and batch processing

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PromptChain (Main API)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Executor     │  │  Validators  │  │  Caching         │  │
│  │  - Retry      │  │  - Schema    │  │  - Memory        │  │
│  │  - Fallback   │  │  - Semantic  │  │  - Redis         │  │
│  │  - Strategy   │  │  - Custom    │  │  - Disk          │  │
│  └───────────────┘  └──────────────┘  └──────────────────┘  │
│         │                   │                    │           │
└─────────┼───────────────────┼────────────────────┼──────────┘
          │                   │                    │
┌─────────┴───────────────────┴────────────────────┴──────────┐
│                    Provider Layer                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌───────┐ │
│  │Anthropic│ │ OpenAI  │ │  Groq    │ │ Cohere  │ │Google │ │
│  │Provider │ │Provider │ │Provider  │ │Provider │ │Provider│ │
│  └─────────┘ └─────────┘ └──────────┘ └─────────┘ └───────┘ │
│       │          │            │            │          │      │
└───────┼──────────┼────────────┼────────────┼──────────┼───────┘
        │          │            │            │          │
    ┌───┴──┬───────┴──┬─────────┴──┬────────┴──┬───────┴────┐
    │      │          │            │           │            │
  Claude  GPT-4    Llama-70b   Command-R    Gemini      Custom
```

## Data Flow

### Execution Flow

```
User Request
    │
    ↓
┌─────────────────────┐
│  Format Messages    │
│  (string → list)    │
└─────────────────────┘
    │
    ↓
┌─────────────────────┐
│  Check Cache        │◄─── If cached, return immediately
└─────────────────────┘
    │
    ↓
┌─────────────────────────────────────┐
│  Execute with Cascade Strategy      │
│  (try models in order)              │
└─────────────────────────────────────┘
    │
    ├──→ Model 1 ──┐
    │    ├─ Attempt 1  ──→ Failed
    │    ├─ Attempt 2  ──→ Failed
    │    └─ Attempt 3  ──→ Success! ✓
    │
    ├──→ [Model 2, 3 not tried]
    │
    ↓
┌──────────────────────────────────┐
│  Validate Response               │
│  ├─ Schema validation            │
│  └─ Semantic validators          │
└──────────────────────────────────┘
    │
    ↓
┌──────────────────────────────────┐
│  Cache Result                    │
└──────────────────────────────────┘
    │
    ↓
┌──────────────────────────────────┐
│  Return PromptResult             │
│  ├─ Response                     │
│  ├─ Metadata                     │
│  └─ Validation Results           │
└──────────────────────────────────┘
```

## Core Components

### 1. PromptChain (core/chain.py)

Main orchestrator. Coordinates:
- Message formatting
- Provider selection
- Execution strategy
- Validation pipeline
- Caching

### 2. Executor (core/executor.py)

Handles execution logic:
- Retry strategies with backoff
- Model fallback chains
- Timeout handling
- Error aggregation

### 3. Providers (providers/)

Abstract interface with implementations:
- Base provider interface
- Provider-specific implementations
- Unified response format
- Token counting
- Cost estimation

### 4. Validators (validation/)

Response validation:
- Schema validation (Pydantic)
- Semantic validators (keywords, citations, etc.)
- Custom validator framework

### 5. Caching (caching/)

Response caching backends:
- In-memory (fast, no persistence)
- Redis (fast, distributed)
- Disk (persistent, single-machine)

### 6. Retry (retry/)

Retry strategies:
- Exponential backoff
- Fibonacci backoff
- Linear backoff
- Custom strategies

## Design Patterns

### 1. Provider Pattern

Each provider implements the same interface but connects to different APIs.

```python
class BaseProvider(ABC):
    async def execute(messages, model, **kwargs) -> dict
    async def stream(messages, model, **kwargs) -> AsyncIterator[str]
    def count_tokens(text, model) -> int
    def estimate_cost(input_tokens, output_tokens, model) -> float
```

### 2. Strategy Pattern

Different execution strategies (cascade, fastest, cheapest, parallel).

### 3. Decorator Pattern

Validators wrap responses and enforce constraints.

### 4. Factory Pattern

Provider factory creates appropriate provider based on model identifier.

## Extensibility

### Adding a Custom Provider

```python
from promptguard.providers.base import BaseProvider

class CustomProvider(BaseProvider):
    async def execute(self, messages, model, **kwargs):
        # Implementation
        pass
    
    # Implement other abstract methods...

# Register it
from promptguard.providers import register_provider
register_provider("custom", CustomProvider)

# Use it
chain = PromptChain(models=["custom/model-name"])
```

### Adding a Custom Validator

```python
from promptguard.validation.semantic import Validator

class CustomValidator(Validator):
    async def validate(self, response, context=None):
        # Custom validation logic
        pass

# Use it
chain = PromptChain(
    models=["anthropic/claude-3-5-sonnet"],
    validators=[CustomValidator()]
)
```

## Performance Considerations

1. **Caching** - Reduces latency for repeated queries by up to 100x
2. **Streaming** - Enables real-time response display
3. **Batch Processing** - Concurrency control prevents rate limiting
4. **Async** - Non-blocking I/O for multiple concurrent requests
5. **Token Counting** - Preemptively validates token limits

## Error Handling Strategy

1. **Provider Errors** - Caught and retried with backoff
2. **Rate Limits** - Special handling with longer delays
3. **Timeouts** - Graceful fallback to next model
4. **Validation Errors** - Can either retry (strict) or warn (lenient)
5. **Cache Errors** - Silently fail and fall back to API

## Testing Strategy

- **Unit Tests** - Test individual components in isolation
- **Integration Tests** - Test provider integrations (with mocks)
- **End-to-End Tests** - Test full workflows
- **Performance Tests** - Benchmark critical paths
- **Stress Tests** - Test under rate limits and high load
