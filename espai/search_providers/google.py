"""Google Custom Search API provider."""

import os
from typing import List

import aiohttp
from aiohttp import ClientSession

from . import SearchProvider, SearchResult


class GoogleSearchProvider(SearchProvider):
    """Google Custom Search API provider."""
    
    def __init__(self):
        """Initialize the Google Custom Search API client."""
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        if not self.cse_id:
            raise ValueError("GOOGLE_CSE_ID environment variable not set")
        
        self.session = None
    
    async def _ensure_session(self):
        """Ensure we have an aiohttp session."""
        if self.session is None:
            self.session = ClientSession()
        return self.session
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[SearchResult]:
        """Execute a Google Custom Search query.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        results = []
        
        # Google CSE returns max 10 results per request
        num_results = min(max_results, 10)
        
        try:
            session = await self._ensure_session()
            
            # Build the Google CSE API URL
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": num_results,
                "alt": "json"
            }
            
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    print(f"Error from Google API: {response.status}")
                    return results
                
                data = await response.json()
                
                if "items" in data:
                    for item in data["items"]:
                        result = SearchResult(
                            title=item.get("title", ""),
                            url=item.get("link", ""),
                            snippet=item.get("snippet", ""),
                            domain=item.get("displayLink", "")
                        )
                        results.append(result)
                        
                        if len(results) >= max_results:
                            break
                            
        except Exception as e:
            print(f"Error searching Google: {str(e)}")
            
        return results
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
