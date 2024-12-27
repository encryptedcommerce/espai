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
        """Parse a natural language query into structured components."""
        prompt = f"""Parse this query into structured components:
{query}

Output format:
entity_type: <type of entity being searched for>
attributes: <list of attributes to extract, comma separated>
search_space: <optional geographic or domain-specific scope>

Rules:
1. entity_type should be a singular noun (e.g. "company" not "companies")
2. attributes should be normalized:
   - "name" and "names" -> "name"
   - "website" and "websites" -> "website"
   - "phone" and "phones" -> "phone"
   - "email" and "emails" -> "email"
3. search_space should capture any geographic or domain constraints
4. if no search_space is specified, output "none"

Example 1:
Query: "find tech companies in california with their websites and phone numbers"
Output:
entity_type: company
attributes: website,phone
search_space: california

Example 2:
Query: "list universities with email addresses"
Output:
entity_type: university
attributes: email
search_space: none"""

        try:
            response = self.model.generate_content(prompt)
            if self.verbose:
                print(f"\033[34mGemini Response:\n{response.text}\033[0m\n")
            
            # Parse response
            lines = response.text.strip().split('\n')
            entity_type = None
            attributes = []
            search_space = None
            
            for line in lines:
                if ':' not in line:
                    continue
                    
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip().lower()
                
                if key == 'entity_type':
                    entity_type = value
                elif key == 'attributes':
                    # Split and clean attributes
                    attrs = [a.strip() for a in value.split(',')]
                    # Normalize attribute names
                    normalized = []
                    for attr in attrs:
                        if attr in ['name', 'names']:
                            continue  # Skip 'name' since it's always included
                        elif attr in ['website', 'websites']:
                            normalized.append('website')
                        elif attr in ['phone', 'phones']:
                            normalized.append('phone')
                        elif attr in ['email', 'emails']:
                            normalized.append('email')
                        elif attr in ['director', 'directors']:
                            normalized.append('director')
                        else:
                            normalized.append(attr)
                    attributes = normalized
                elif key == 'search_space' and value != 'none':
                    search_space = value
            
            if not entity_type:
                raise ValueError("No entity type found in response")
            
            return entity_type, attributes, search_space
            
        except Exception as e:
            if self.verbose:
                print(f"\033[31mError parsing query: {str(e)}\033[0m\n")
            raise
    
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
        """Parse a search result to extract requested attributes.
        
        Args:
            text: Text to parse (title + snippet)
            entity_type: Type of entity to extract
            attributes: List of attributes to extract
            
        Returns:
            Dictionary of extracted attributes or None if no match
        """
        prompt = f"""Extract information about a {entity_type} from this text:

{text}

Rules:
1. If the text contains a {entity_type}, extract these attributes: {', '.join(attributes)}
2. If you can't find a {entity_type}, return "none"
3. If a URL is provided, use it to help identify the entity name
4. The name should be the official/formal name of the {entity_type}
5. Remove any extra text like "Home", "Welcome to", "Official Site of" from names
6. If you're not confident about the extraction, return "none"

Example 1:
Text: "Welcome to Example Corp - Home Page | Leading Tech Solutions"
URL: example-corp.com
Output: {{"name": "Example Corp"}}

Example 2:
Text: "Front Page"
URL: adrenalinesportsacademy.com
Output: {{"name": "Adrenaline Sports Academy"}}

Example 3:
Text: "Page not found"
URL: example.com/404
Output: none

Format the output as a JSON object with the extracted attributes as keys.
If no {entity_type} is found, output "none"."""

        try:
            result = await self._generate_and_parse_json(prompt)
            if not result or not isinstance(result, dict):
                return None
            return result if any(result.values()) else None
        except (ValueError, AttributeError):
            if self.verbose:
                print(f"\nFailed to parse result from text: {text[:100]}...")
            return None
