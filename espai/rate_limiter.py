"""Rate limiter with exponential backoff."""

import asyncio
import time
from typing import Any, Callable


class RateLimiter:
    """Rate limiter with exponential backoff."""
    
    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 32.0,
        max_retries: int = 5
    ):
        """Initialize the rate limiter.
        
        Args:
            base_delay: Base delay between requests in seconds
            max_delay: Maximum delay between requests in seconds
            max_retries: Maximum number of retries
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.last_request_time = 0.0
    
    async def execute(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Execute the function with rate limiting and retries."""
        retry_count = 0
        current_delay = self.base_delay
        
        while True:
            try:
                # Wait for minimum delay since last request
                now = time.time()
                time_since_last = now - self.last_request_time
                if time_since_last < self.base_delay:
                    await asyncio.sleep(self.base_delay - time_since_last)
                
                # Execute function
                result = await func(*args, **kwargs)
                self.last_request_time = time.time()
                return result
                
            except Exception as e:
                retry_count += 1
                
                if retry_count > self.max_retries:
                    raise
                
                if "429" in str(e):  # Too Many Requests
                    # Use exponential backoff
                    delay = min(current_delay * (2 ** (retry_count - 1)), self.max_delay)
                    await asyncio.sleep(delay)
                    current_delay = delay
                else:
                    raise
