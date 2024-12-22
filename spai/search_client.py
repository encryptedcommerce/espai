"""Google Search API client for performing searches."""

import os
from typing import List, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .rate_limiter import RateLimiter, with_exponential_backoff


class GoogleSearchClient:
    """Client for performing Google searches."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cse_id: Optional[str] = None,
        max_results: int = 10
    ):
        """Initialize the Google Search client."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.cse_id = cse_id or os.getenv("GOOGLE_CSE_ID")
        
        if not self.api_key:
            raise ValueError("Google API key not provided")
        if not self.cse_id:
            raise ValueError("Google Custom Search Engine ID not provided")
            
        self.max_results = max_results
        self.service = build("customsearch", "v1", developerKey=self.api_key)
        self.rate_limiter = RateLimiter(min_delay=1.0)  # 1 request per second
    
    async def _execute_search(self, query: str, start_index: int) -> dict:
        """Execute a single search request with retries."""
        await self.rate_limiter.wait()
        
        return await with_exponential_backoff(
            self.service.cse().list(
                q=query,
                cx=self.cse_id,
                start=start_index
            ).execute,
            max_retries=6,
            base_delay=2.0,
            max_delay=64.0
        )
    
    async def search(self, query: str) -> List[dict]:
        """Perform a Google search and return results."""
        try:
            results = []
            start_index = 1
            
            while len(results) < self.max_results:
                response = await self._execute_search(query, start_index)
                
                if "items" not in response:
                    break
                    
                results.extend(response["items"])
                if len(response["items"]) < 10:  # Less than a full page
                    break
                    
                start_index += 10
                
            return results[:self.max_results]
            
        except HttpError as e:
            print(f"Error performing Google search: {e}")
            return []
