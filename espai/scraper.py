"""Web scraping utilities for extracting data from search result pages."""

import asyncio
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup


class Scraper:
    """Web scraper for extracting data from search result pages."""
    
    def __init__(self, timeout: int = 10):
        """Initialize the scraper.
        
        Args:
            timeout: Default timeout in seconds for requests
        """
        self.timeout = timeout
        self.session = None
    
    async def __aenter__(self):
        """Create aiohttp session when entering context."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session when exiting context."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def scrape_page(self, url: str, timeout: Optional[int] = None) -> Optional[str]:
        """Scrape a webpage and return its text content.
        
        Args:
            url: URL to scrape
            timeout: Optional timeout override
            
        Returns:
            Text content of the page, or None if scraping failed
        """
        timeout = timeout or self.timeout
        
        try:
            # Create session if needed
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(
                url,
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                }
            ) as response:
                if response.status != 200:
                    return None
                    
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                
                # Remove unwanted elements
                for element in soup(["script", "style", "meta", "link", "noscript", "iframe"]):
                    element.decompose()
                
                # Replace common block elements with newlines
                for tag in soup.find_all(["p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6"]):
                    tag.insert_after(soup.new_string("\n"))
                
                # Replace list items with bullet points
                for li in soup.find_all("li"):
                    li.insert_before(soup.new_string("• "))
                
                # Get text and clean it up
                text = soup.get_text()
                
                # Clean up whitespace while preserving structure
                lines = []
                for line in text.splitlines():
                    line = line.strip()
                    if line:  # Skip empty lines
                        # Preserve certain punctuation by ensuring spaces around it
                        for char in [",", ".", ":", ";", "•"]:
                            line = line.replace(char + " ", char)  # Remove extra space after punctuation
                        lines.append(line)
                
                # Join with newlines and remove excessive whitespace
                text = "\n".join(lines)
                text = " ".join(text.split())  # Normalize whitespace within lines
                
                # Remove duplicate newlines while preserving paragraph structure
                while "\n\n\n" in text:
                    text = text.replace("\n\n\n", "\n\n")
                
                # Truncate if too long (Gemini has a context limit)
                if len(text) > 10000:
                    # Try to truncate at a sentence boundary
                    truncate_point = text[:10000].rfind(".")
                    if truncate_point > 0:
                        text = text[:truncate_point + 1]
                    else:
                        text = text[:10000] + "..."
                
                return text
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return None
        
        finally:
            # Don't close the session here - let it be reused
            pass
