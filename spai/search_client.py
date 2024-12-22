"""Google Search API client for performing searches."""

import os
from typing import List, Dict, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .rate_limiter import RateLimiter


class GoogleSearchClient:
    """Client for Google Custom Search API."""
    
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
        self.rate_limiter = RateLimiter(
            base_delay=0.5,  # 0.5 second between requests
            max_delay=32.0,  # Max 32 second delay
            max_retries=6    # Up to 6 retries
        )
    
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
    
    async def search(self, query: str) -> List[Dict[str, str]]:
        """Search using Google Custom Search API."""
        async def _search():
            service = build(
                "customsearch", "v1",
                developerKey=self.api_key,
                static_discovery=False
            )
            
            try:
                result = service.cse().list(
                    q=query,
                    cx=self.cse_id,
                    num=self.max_results
                ).execute()
                
                return result.get("items", [])
            finally:
                service.close()
        
        try:
            return await self.rate_limiter.execute(_search)
        except Exception as e:
            print(f"Error searching: {str(e)}")
            return []
