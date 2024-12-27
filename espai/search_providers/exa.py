"""Exa.ai search API provider."""

import os
from datetime import datetime
from typing import List

import httpx

from . import SearchProvider, SearchResult


class ExaSearchProvider(SearchProvider):
    """Exa.ai search API provider."""
    
    def __init__(self):
        """Initialize the Exa search client."""
        self.api_key = os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY environment variable not set")
            
        self.client = None
    
    async def _ensure_client(self):
        """Ensure we have an active client."""
        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url="https://api.exa.ai",
                headers={"x-api-key": self.api_key}
            )
        return self.client
    
    async def search(
        self,
        query: str,
        max_results: int = 100,
    ) -> List[SearchResult]:
        """Execute an Exa search query.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        results = []
        
        # Exa supports up to 100 results per request
        size = min(max_results, 100)
        
        try:
            client = await self._ensure_client()
            
            response = await client.post(
                "/search",
                json={
                    "query": query,
                    "num_results": size,
                    "include_domains": [],  # No domain restrictions
                    "exclude_domains": [],
                    "use_autoprompt": True,  # Let Exa optimize the query
                    "type": "neural",       # Neural search
                }
            )
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("results", []):
                # Convert timestamp to ISO format if present
                published = None
                if item.get("published_date"):
                    try:
                        timestamp = int(item["published_date"])
                        published = datetime.fromtimestamp(timestamp).isoformat()
                    except (ValueError, TypeError):
                        pass
                
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("text", ""),  # Exa returns full text
                    domain=item.get("domain", ""),
                    published_date=published
                )
                results.append(result)
                
                if len(results) >= max_results:
                    break
            
        except Exception as e:
            print(f"Error searching Exa: {str(e)}")
            
        return results
    
    async def close(self):
        """Close the client."""
        if self.client:
            await self.client.aclose()
            self.client = None
