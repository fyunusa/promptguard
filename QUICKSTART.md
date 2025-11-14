# Quick Start - PromptGuard

Get started with PromptGuard in 5 minutes.

## 1. Install

```bash
# Navigate to project directory
cd /Users/fyunusa/Documents/promptguard

# Install in development mode with all features
pip install -e ".[all]"

# Install development tools
pip install -r requirements-dev.txt
pre-commit install
```

## 2. Set API Keys

```bash
# Add to your ~/.zshrc or environment
export ANTHROPIC_API_KEY=sk-ant-YOUR_KEY
export OPENAI_API_KEY=sk-YOUR_KEY
export GROQ_API_KEY=gsk-YOUR_KEY
```

## 3. Run Examples

```bash
# Try basic execution with fallbacks
python examples/basic_execution.py

# Try type-safe responses
python examples/type_safe_responses.py

# Try response validation
python examples/validation.py

# Try caching
python examples/caching.py

# Try batch processing
python examples/batch_processing.py

# Try streaming
python examples/streaming.py
```

## 4. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
make test-cov

# Run specific test
pytest tests/unit/test_core.py::TestPromptChain -v
```

## 5. Development Tasks

```bash
# Format code
make format

# Lint code
make lint

# Type check
make type

# All checks
make all
```

## 6. Explore the Code

Start with these files:

1. **`promptguard/core/chain.py`** - Main orchestrator
2. **`promptguard/providers/base.py`** - Provider interface
3. **`promptguard/validation/semantic.py`** - Validators
4. **`examples/`** - Usage examples

## 7. Read Documentation

- `README.md` - Overview
- `docs/getting_started.md` - Detailed guide
- `docs/api_reference.md` - API docs
- `docs/architecture.md` - Architecture details
- `CONTRIBUTING.md` - How to contribute

## 8. Common Tasks

### Execute a Simple Prompt

```python
import asyncio
from promptguard import PromptChain

async def main():
    chain = PromptChain(models=["anthropic/claude-3-5-sonnet"])
    result = await chain.execute("What is AI?")
    print(result.response)

asyncio.run(main())
```

### With Multiple Models (Fallback)

```python
chain = PromptChain(
    models=[
        "anthropic/claude-3-5-sonnet",
        "openai/gpt-4o",
        "groq/llama-70b"
    ]
)
```

### With Caching

```python
from promptguard import CacheBackend

chain = PromptChain(
    models=["anthropic/claude-3-5-sonnet"],
    cache=CacheBackend.memory()
)

result1 = await chain.execute("Q1", cache_key="q1")
result2 = await chain.execute("Q1", cache_key="q1")  # From cache!
```

### With Type-Safe Responses

```python
from pydantic import BaseModel
from promptguard import PromptChain

class Answer(BaseModel):
    content: str
    confidence: float

chain = PromptChain(
    models=["anthropic/claude-3-5-sonnet"],
    response_schema=Answer
)

result = await chain.execute("What is ML?")
print(result.response.confidence)  # Type-safe!
```

## 🎯 Next Steps

1. Run the examples
2. Read the documentation
3. Try the API with your own prompts
4. Customize with validators
5. Integrate into your application

## 📚 Resources

- **Project Completion**: `PROJECT_COMPLETION.md`
- **Getting Started**: `docs/getting_started.md`
- **API Reference**: `docs/api_reference.md`
- **Architecture**: `docs/architecture.md`
- **Examples**: `examples/` directory

## ❓ Need Help?

- Check examples in `examples/` directory
- Read API docs in `docs/api_reference.md`
- Review architecture in `docs/architecture.md`
- Check test files for usage patterns in `tests/`

## 🚀 You're Ready!

PromptGuard is now set up and ready to use. Start with an example or write your own code using the API reference.
