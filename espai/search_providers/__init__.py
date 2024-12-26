"""Base classes for search providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SearchResult:
    """A single search result."""
    
    title: str
    url: str
    snippet: str
    domain: Optional[str] = None
    published_date: Optional[str] = None


class SearchProvider(ABC):
    """Base class for search providers."""
    
    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[SearchResult]:
        """Execute a search query.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        pass
