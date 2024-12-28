"""Client for interacting with Gemini AI."""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai

from .rate_limiter import RateLimiter


class GeminiClient:
    """Client for interacting with Gemini AI."""
    
    def __init__(self, verbose: bool = False, temperature: float = 0.7):
        """Initialize the Gemini client.
        
        Args:
            verbose: Whether to show verbose output
            temperature: Temperature for generation (0.0 to 1.0)
        """
        self.verbose = verbose
        
        # Configure Gemini with API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=genai.types.GenerationConfig(
                temperature=temperature
            )
        )
        
    async def _generate_with_retry(self, prompt: str) -> str:
        """Generate text with retries and rate limiting."""
        async def _generate():
            response = await self.model.generate_content_async(
                prompt
            )
            return response
            
        try:
            result = await RateLimiter(
                base_delay=0.5,  # 0.5 second between requests
                max_delay=32.0,  # Max 32 second delay
                max_retries=6    # Up to 6 retries
            ).execute(_generate)
            return result
        except Exception as e:
            if self.verbose:
                print(f"\033[38;5;209mError generating response: {str(e)}\033[0m\n")  # Light orange color
            raise
    
    async def _generate_and_parse_json(self, prompt: str) -> Optional[Dict]:
        """Generate a response and parse it as JSON.
        
        Args:
            prompt: Prompt to send to the model
            
        Returns:
            Parsed JSON response or None if failed
        """
        try:
            response = await self._generate_with_retry(prompt)
            
            # Handle multi-part responses
            if hasattr(response, 'parts'):
                text = '\n'.join(part.text for part in response.parts)
            else:
                # Handle case where response might already be a dictionary
                if isinstance(response.text, dict):
                    return response.text
                text = response.text
            
            # Try to find JSON in the response
            try:
                # First check for code blocks
                code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
                if code_block_match:
                    try:
                        return json.loads(code_block_match.group(1))
                    except:
                        pass
                
                # Then try to parse the entire text
                try:
                    return json.loads(text.strip())
                except:
                    pass
                
                # Finally try to find JSON-like content
                text = text.strip()
                if text.startswith('['):
                    # Find matching closing bracket
                    count = 0
                    for i, char in enumerate(text):
                        if char == '[':
                            count += 1
                        elif char == ']':
                            count -= 1
                            if count == 0:
                                try:
                                    return json.loads(text[:i+1])
                                except:
                                    pass
                                break
                
                # Try to find object matches
                matches = re.findall(r'\{[^}]+\}', text)
                if matches:
                    return json.loads(matches[0])
                
                return None
                    
            except Exception as e:
                if self.verbose:
                    print(f"\033[38;5;209mError parsing JSON: {str(e)}\033[0m\n")  # Light orange color
                return None
            
        except Exception as e:
            if self.verbose:
                print(f"\033[38;5;209mError generating response: {str(e)}\033[0m\n")  # Light orange color
            return None
            
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
            response = await self._generate_with_retry(prompt)
            
            # Handle multi-part responses
            if hasattr(response, 'parts'):
                text = '\n'.join(part.text for part in response.parts)
            else:
                text = response.text
                
            # Parse response
            lines = text.strip().split('\n')
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
                print(f"\033[38;5;209mError parsing query: {str(e)}\033[0m\n")  # Light orange color
            raise
    
    async def enumerate_search_space(self, search_space: str) -> Optional[List[str]]:
        """Enumerate the search space into specific items."""
        prompt = f"""Enumerate the specific items in this search space: {search_space}

Rules:
1. Return a JSON array of the EXACT items that match the search space
2. For counties, return the full county name with "County" (e.g., "Collier County")
3. For states, use standard state abbreviations (e.g., "FL" not "Florida")
4. Return REAL locations only, not placeholders or generic items
5. If the search involves rankings (highest, wealthiest, etc), return the actual items in correct order

Examples:

Input: "3 highest-income FL counties"
[
  "Collier County",
  "St. Johns County",
  "Martin County"
]

Input: "top 2 FL beach cities"
[
  "Naples",
  "Sarasota"
]

Input: "Miami-Dade neighborhoods"
[
  "South Beach",
  "Coral Gables",
  "Coconut Grove",
  "Brickell"
]"""

        try:
            response = await self._generate_with_retry(prompt)
            
            if hasattr(response, 'parts'):
                text = '\n'.join(part.text for part in response.parts)
            else:
                text = response.text
                
            items = self._parse_json_response(text)
            if not items or not isinstance(items, list):
                return None
                
            return items
            
        except Exception as e:
            if self.verbose:
                print(f"\033[38;5;209mError enumerating search space: {str(e)}\033[0m\n")
            return None

    async def extract_attributes(self, text: str, url: str, entity_type: str, attributes: List[str]) -> Optional[Dict[str, str]]:
        """Extract requested attributes from text."""
        prompt = f"""Extract ONLY the following attributes for this {entity_type} from the text, if they are explicitly mentioned:
{', '.join(attributes)}

Text to analyze:
{text}
URL: {url}

Rules:
1. Return a JSON object with ONLY the attributes that are explicitly mentioned in the text
2. Do NOT include attributes that are not clearly stated in the list of requested attributes
3. Do NOT make assumptions or infer values
4. For 'address' (if present in the list of requested attributes):
   - If you find an address like "123 Main St, Springfield, MA 01234", decompose it into:
     {{"street_address": "123 Main St", "city": "Springfield", "state": "MA", "zip": "01234"}}
   - If you find "1515 Golden Gate Parkway, Naples", decompose it into:
     {{"street_address": "1515 Golden Gate Parkway", "city": "Naples"}}
   - NEVER return a complete address string - always decompose it
   - Return each component separately
   - Omit any components that aren't present

Examples:

Input text: "The Paradise Coast Sports Complex at 3892 City Gate Blvd. South, Naples FL, 34117"
Output: {{
    "name": "Paradise Coast Sports Complex",
    "street_address": "3892 City Gate Blvd. South",
    "city": "Naples",
    "state": "FL",
    "zip": "34117"
}}

Input text: "Freedom Park at 1515 Golden Gate Parkway, Naples"
Output: {{
    "name": "Freedom Park",
    "street_address": "1515 Golden Gate Parkway",
    "city": "Naples"
}}"""

        try:
            response = await self._generate_and_parse_json(prompt)
            if response:
                return response
                
            return None
            
        except Exception as e:
            return None
            
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
        return await self.extract_attributes(text, "", entity_type, attributes)

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """Parse a JSON response from the LLM."""
        try:
            # First check for code blocks
            code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if code_block_match:
                try:
                    return json.loads(code_block_match.group(1))
                except:
                    pass
            
            # Then try to parse the entire text
            try:
                return json.loads(text.strip())
            except:
                pass
            
            # Finally try to find JSON-like content
            text = text.strip()
            if text.startswith('{'):
                # Find matching closing brace
                count = 0
                for i, char in enumerate(text):
                    if char == '{':
                        count += 1
                    elif char == '}':
                        count -= 1
                        if count == 0:
                            try:
                                return json.loads(text[:i+1])
                            except:
                                pass
                            break
            
            if self.verbose:
                print(f"\033[38;5;209mNo JSON found in response\033[0m\n")  # Light orange color
            return None
            
        except Exception as e:
            if self.verbose:
                print(f"\033[38;5;209mError parsing JSON: {str(e)}\033[0m\n")  # Light orange color
            return None
