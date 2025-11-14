"""Unit tests for PromptGuard core functionality."""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from promptguard.core.chain import PromptChain
from promptguard.core.response import PromptResult, ExecutionMetadata
from promptguard.caching.base import CacheBackend
from promptguard.caching.memory import MemoryCacheBackend
from promptguard.retry.strategies import ExponentialBackoff, ConstantDelay
from promptguard.exceptions import PromptExecutionError


class TestPromptChain:
    """Tests for PromptChain orchestrator."""
    
    @pytest.fixture
    def chain(self):
        """Create test PromptChain."""
        with patch('promptguard.providers.get_provider'):
            return PromptChain(models=["anthropic/claude-3-5-sonnet"])
    
    def test_init(self):
        """Test PromptChain initialization."""
        with patch('promptguard.providers.get_provider'):
            chain = PromptChain(
                models=["anthropic/claude-3-5-sonnet"],
                strategy="cascade",
                max_retries=3,
                timeout=30.0
            )
            assert chain.models == ["anthropic/claude-3-5-sonnet"]
            assert chain.strategy == "cascade"
            assert chain.max_retries == 3
            assert chain.timeout == 30.0
    
    def test_format_messages_string(self, chain):
        """Test message formatting from string."""
        prompt = "What is AI?"
        messages = chain._format_messages(prompt)
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is AI?"
    
    def test_format_messages_list(self, chain):
        """Test message formatting from list."""
        messages_list = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        messages = chain._format_messages(messages_list)
        
        assert messages == messages_list
    
    def test_format_messages_with_context(self, chain):
        """Test message formatting with context."""
        prompt = "What is {topic}?"
        context = {"topic": "machine learning"}
        messages = chain._format_messages(prompt, context)
        
        assert messages[0]["content"] == "What is machine learning?"


class TestRetryStrategies:
    """Tests for retry strategies."""
    
    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        strategy = ExponentialBackoff(base=1.0, max_delay=60.0)
        
        assert strategy.get_delay(0, Exception()) == 1.0
        assert strategy.get_delay(1, Exception()) == 2.0
        assert strategy.get_delay(2, Exception()) == 4.0
        assert strategy.get_delay(10, Exception()) == 60.0  # Capped at max_delay
    
    def test_constant_delay(self):
        """Test constant delay calculation."""
        strategy = ConstantDelay(delay=2.0)
        
        assert strategy.get_delay(0, Exception()) == 2.0
        assert strategy.get_delay(1, Exception()) == 2.0
        assert strategy.get_delay(2, Exception()) == 2.0
    
    def test_should_retry(self):
        """Test retry decision logic."""
        strategy = ExponentialBackoff(max_retries=3)
        
        assert strategy.should_retry(0, Exception()) is True
        assert strategy.should_retry(1, Exception()) is True
        assert strategy.should_retry(2, Exception()) is True
        assert strategy.should_retry(3, Exception()) is False


class TestCache:
    """Tests for caching system."""
    
    @pytest.mark.asyncio
    async def test_memory_cache(self):
        """Test in-memory cache backend."""
        cache = MemoryCacheBackend()
        
        # Test set and get
        await cache.set("key1", {"data": "value"}, ttl=3600)
        result = await cache.get("key1")
        assert result == {"data": "value"}
        
        # Test non-existent key
        result = await cache.get("non_existent")
        assert result is None
        
        # Test delete
        await cache.delete("key1")
        result = await cache.get("key1")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_memory_cache_expiry(self):
        """Test cache expiry."""
        cache = MemoryCacheBackend()
        
        # Set with 1ms TTL
        await cache.set("key1", {"data": "value"}, ttl=0.001)
        
        # Wait for expiry
        await asyncio.sleep(0.01)
        
        result = await cache.get("key1")
        assert result is None


class TestPromptResult:
    """Tests for PromptResult."""
    
    def test_result_creation(self):
        """Test PromptResult creation."""
        metadata = ExecutionMetadata(
            model_used="claude-3-5-sonnet",
            provider="anthropic",
            attempts=1,
            execution_time_ms=1234.5,
            tokens_used={"input": 100, "output": 50},
            estimated_cost=0.023,
            cached=False,
        )
        
        result = PromptResult(
            success=True,
            response="This is a response",
            metadata=metadata
        )
        
        assert result.success is True
        assert result.response == "This is a response"
        assert result.metadata.model_used == "claude-3-5-sonnet"
    
    def test_result_to_dict(self):
        """Test PromptResult serialization."""
        metadata = ExecutionMetadata(
            model_used="claude-3-5-sonnet",
            provider="anthropic",
            attempts=1,
            execution_time_ms=1234.5,
            tokens_used={"input": 100, "output": 50},
            estimated_cost=0.023,
            cached=False,
        )
        
        result = PromptResult(
            success=True,
            response="Test response",
            metadata=metadata
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['success'] is True
        assert result_dict['response'] == "Test response"
        assert result_dict['metadata']['model_used'] == "claude-3-5-sonnet"
