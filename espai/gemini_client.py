"""Client for interacting with Gemini AI."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai

from .rate_limiter import RateLimiter


class GeminiClient:
    """Client for interacting with Gemini AI."""
    
    MODEL_NAME = "gemini-2.0-flash-exp"
    
    def __init__(self, temperature: float = 0.7, verbose: bool = False):
        """Initialize the Gemini client.
        
        Args:
            temperature: Temperature for generation (0.0 to 1.0)
            verbose: Whether to print debug information
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided")
        
        self.verbose = verbose
        self.temperature = temperature
        self.rate_limiter = RateLimiter(
            base_delay=0.5,  # 0.5 second between requests
            max_delay=32.0,  # Max 32 second delay
            max_retries=6    # Up to 6 retries
        )
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.MODEL_NAME)  # Use MODEL_NAME class constant
    
    async def _generate_with_retry(self, prompt: str) -> str:
        """Generate text with retries and rate limiting."""
        async def _generate():
            response = await self.model.generate_content_async(
                prompt,
                generation_config={"temperature": self.temperature}
            )
            return response.text
            
        try:
            result = await self.rate_limiter.execute(_generate)
            return result
        except Exception as e:
            if self.verbose:
                print(f"\n[Error] Generation failed: {str(e)}")
            raise
    
    async def _generate_and_parse_json(self, prompt: str) -> Any:
        """Generate content and parse it as JSON."""
        try:
            text = await self._generate_with_retry(prompt)
            text = self._clean_json_text(text)
            
            # Check for truncation
            if text.count('[') != text.count(']'):
                if self.verbose:
                    print("Warning: Response appears to be truncated")
                # Try to fix truncated array by finding last complete item
                last_comma = text.rfind(',')
                if last_comma != -1:
                    text = text[:last_comma] + ']'
            
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Gemini response as JSON. Response: {text}")
    
    def _clean_json_text(self, text: str) -> str:
        """Clean up text to extract JSON content."""
        # Find the first { or [ and last } or ]
        start = min(
            (text.find('{'), text.find('[')),
            key=lambda x: float('inf') if x == -1 else x
        )
        end = max(text.rfind('}'), text.rfind(']')) + 1
        
        if start == -1 or end == 0:
            raise ValueError("No JSON content found in response")
            
        return text[start:end]
    
    async def parse_query(
        self,
        query: str
    ) -> Tuple[str, List[str], Optional[str]]:
        """Parse a natural language query into structured components.
        
        Args:
            query: Natural language query string
            
        Returns:
            Tuple of (entity_type, attributes, search_space)
        """
        prompt = f"""Parse this search query into structured components:
"{query}"

Return a JSON object with these fields:
- entity: The type of entity being searched for (e.g. "restaurants", "doctors")
- attributes: List of attributes to extract (e.g. ["name", "address", "phone"])
- search_space: The scope or area to search within (e.g. "New York City", "California zip codes"), or null if not specified

Example response:
{{
    "entity": "restaurants",
    "attributes": ["name", "address", "phone"],
    "search_space": "Manhattan"
}}"""
        
        result = await self._generate_and_parse_json(prompt)
        
        return (
            result["entity"],
            result["attributes"],
            result.get("search_space")
        )
    
    async def enumerate_search_space(self, search_space: str) -> List[str]:
        """Enumerate items in a search space.
        
        Args:
            search_space: Description of the search space
            
        Returns:
            List of items in the search space
        """
        prompt = f"""Enumerate all items in this search space:
"{search_space}"

Return a JSON array of strings. For geographic areas, use official names.
For example, for "New England states" return:
[
    "Maine",
    "New Hampshire",
    "Vermont",
    "Massachusetts",
    "Rhode Island",
    "Connecticut"
]"""
        
        return await self._generate_and_parse_json(prompt)
    
    async def parse_search_result(
        self,
        text: str,
        entity_type: str,
        attributes: List[str]
    ) -> Optional[Dict[str, str]]:
        """Extract structured data from a search result.
        
        Args:
            text: Text to extract from
            entity_type: Type of entity to extract
            attributes: List of attributes to extract
            
        Returns:
            Dictionary of extracted attributes, or None if no entity found
        """
        # Build list of attributes with examples
        attr_examples = {
            "name": "e.g. 'ACME Fitness Center', 'Gold's Gym'",
            "address": "e.g. '123 Main St, Springfield, IL 62701', '456 Oak Ave, Chicago, IL 60601'"
        }
        
        attr_list = "\n".join(
            f"- {attr}: {attr_examples.get(attr, 'any relevant text')}"
            for attr in attributes
        )
        
        prompt = f"""Extract information about a {entity_type} from this text:
{text}

Return a JSON object with these attributes:
{attr_list}

If an attribute is not found, omit it from the response. If no relevant entity is found, return null.
Do not make up or guess at information - only return what is explicitly stated in the text.

Example response:
{{
    "name": "ACME Fitness Center",
    "address": "123 Main St, Springfield, IL 62701"
}}"""
        
        try:
            result = await self._generate_and_parse_json(prompt)
            return result if result and any(result.values()) else None
        except ValueError:
            return None
