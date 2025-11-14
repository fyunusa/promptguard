"""Pytest configuration for PromptGuard tests."""
import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_prompt():
    """Provide a sample prompt for testing."""
    return "What is artificial intelligence?"


@pytest.fixture
def mock_response():
    """Provide a sample response for testing."""
    return {
        "content": "Artificial intelligence is the simulation of human intelligence.",
        "model": "claude-3-5-sonnet",
        "usage": {
            "input": 10,
            "output": 20,
            "total": 30
        }
    }


@pytest.fixture
def mock_messages():
    """Provide sample messages for testing."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
