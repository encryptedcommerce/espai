"""Web scraping utilities."""

import asyncio
from typing import Optional

import httpx
from bs4 import BeautifulSoup


class Scraper:
    """Simple web scraper."""
    
    def __init__(self):
        """Initialize the scraper."""
        self.client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=10.0
        )
        
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
        
    async def scrape_page(self, url: str) -> Optional[str]:
        """Scrape text content from a webpage.
        
        Args:
            url: URL to scrape
            
        Returns:
            Extracted text content or None if failed
        """
        try:
            # Get the page content
            response = await self.client.get(url)
            
            # Try to detect the encoding
            if response.encoding is None:
                # Try to detect from content
                if response.content.startswith(b'\xef\xbb\xbf'):  # UTF-8 BOM
                    response.encoding = 'utf-8-sig'
                else:
                    # Try common encodings
                    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
                    content = None
                    for encoding in encodings:
                        try:
                            content = response.content.decode(encoding)
                            response.encoding = encoding
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if content is None:
                        # If all encodings fail, use replace error handler
                        content = response.content.decode('utf-8', errors='replace')
                        response.encoding = 'utf-8'
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "meta", "link"]):
                script.decompose()
                
            # Extract text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up text
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = '\n'.join(lines)
            
            # Truncate if too long (keep first 1000 chars)
            if len(text) > 1000:
                text = text[:1000] + "..."
                
            return text
            
        except Exception as e:
            print(f"\033[31mError scraping {url}: {str(e)}\033[0m\n")
            return None
            
        finally:
            # Ensure we don't keep too many connections open
            await asyncio.sleep(0.1)
