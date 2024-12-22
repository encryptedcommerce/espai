"""Gemini AI client for query parsing and result extraction."""

import asyncio
import json
import os
import re
from typing import Dict, List, Optional, Any, Tuple

import google.generativeai as genai

from .models import SearchQuery, EntityResult, Address, Contact, Hours
from .rate_limiter import RateLimiter


class GeminiClient:
    """Client for interacting with Gemini AI."""
    
    MODEL_NAME = "gemini-pro"
    
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
        self.model = genai.GenerativeModel('gemini-pro')
    
    async def _generate_with_retry(self, prompt: str) -> str:
        """Generate text with retries and rate limiting."""
        async def _generate():
            response = await self.model.generate_content_async(
                prompt,
                generation_config={"temperature": self.temperature}
            )
            return response.text
            
        try:
            return await self.rate_limiter.execute(_generate)
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
            if self.verbose:
                print(f"JSON decode error at position {e.pos}")
                print(f"Response text: {text}")
            raise ValueError(f"Failed to parse Gemini response as JSON. Response: {text}")
    
    def _clean_json_text(self, text: str) -> str:
        """Clean text to ensure it's valid JSON."""
        text = text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.endswith('```'):
            text = text[:-3]
        return text.strip()
    
    async def parse_query(self, query: str) -> Tuple[str, List[str], str]:
        """Parse a query to extract entities, attributes, and search space.
        
        Returns:
            Tuple of (entity_type, list of attributes, search space)
        """
        prompt = f"""Given this search query, extract:
1. The entity type we're searching for
2. The attributes we want to extract for each entity
3. The search space we need to enumerate

For example:
Query: "athletic center names and addresses in all AZ zip codes"
{{
    "entity": "athletic centers",
    "attributes": ["name", "address"],
    "search_space": "all AZ zip codes"
}}

Query: "coffee shops in all US States"
{{
    "entity": "coffee shops",
    "attributes": ["name"],
    "search_space": "all US States"
}}

Query: "Find gyms with good ratings and contact info in New York"
{{
    "entity": "gyms",
    "attributes": ["name", "rating", "contact"],
    "search_space": "New York"
}}

Return ONLY a JSON object with "entity", "attributes", and "search_space" fields, no other text.

Query: "{query}"
"""
        try:
            result = await self._generate_and_parse_json(prompt)
            if not result or "entity" not in result or "attributes" not in result or "search_space" not in result:
                raise ValueError("Missing required fields in query parsing result")
            
            if self.verbose:
                print("\nParsed query:")
                print(json.dumps(result, indent=4))
            
            return result["entity"], result["attributes"], result["search_space"]
        except Exception as e:
            if self.verbose:
                print(f"\n[Error] Failed to parse query: {str(e)}")
            raise RuntimeError(f"Error parsing query with Gemini: {str(e)}")
    
    async def enumerate_search_space(self, search_space: str) -> List[str]:
        """Convert a search space description into a list of specific items to search.
        
        Args:
            search_space: Description of the search space (e.g., "all AZ zip codes")
            
        Returns:
            List of specific items to search (e.g., ["85001", "85002", ...])
        """
        # For zip codes, break down by region first
        if "zip codes" in search_space.lower():
            try:
                regions_prompt = f"""Break down this search space into regions that we can enumerate separately.
Return ONLY a JSON array of region descriptions. Each region should cover a specific range of zip codes.

For example:
Query: "all California zip codes"
[
  "Los Angeles County (900xx-904xx)",
  "Orange County (905xx-908xx)", 
  "San Francisco Bay Area (940xx-944xx)",
  "San Diego County (919xx-921xx)",
  "Sacramento Area (956xx-957xx)",
  "Central Valley North (932xx-934xx)",
  "Central Valley South (935xx-937xx)",
  "Inland Empire (917xx-918xx, 923xx-925xx)",
  "North Coast (954xx-955xx, 959xx-960xx)",
  "Central Coast (934xx-935xx, 939xx-940xx)"
]

Query: "{search_space}"
"""
                regions = await self._generate_and_parse_json(regions_prompt)
                if not regions or not isinstance(regions, list) or len(regions) == 0:
                    raise ValueError("Failed to get valid regions list")
                
                all_items = []
                for region in regions:
                    if self.verbose:
                        print(f"Enumerating region: {region}")
                    
                    # Extract zip code range from region description
                    matches = re.findall(r'(\d{3})xx-(\d{3})xx', region)
                    if matches:
                        for start_prefix, end_prefix in matches:
                            start = int(start_prefix + "00")
                            end = int(end_prefix + "99")
                            all_items.extend([f"{i:05d}" for i in range(start, end + 1)])
                    else:
                        # If no range found, ask Gemini for the zip codes
                        region_prompt = f"""List ALL zip codes in this region: {region}
Return ONLY a JSON array of 5-digit zip code strings. Format each zip code as a 5-digit string with leading zeros."""
                        
                        try:
                            items = await self._generate_and_parse_json(region_prompt)
                            if items and isinstance(items, list):
                                all_items.extend(items)
                        except Exception as e:
                            if self.verbose:
                                print(f"Error enumerating region {region}: {str(e)}")
                
                if not all_items:
                    raise ValueError("No items found in any region")
                    
                if self.verbose:
                    print("\nEnumerated search space:")
                    print(json.dumps(all_items, indent=4))
                    
                return sorted(list(set(all_items)))  # Remove duplicates and sort
                
            except Exception as e:
                if self.verbose:
                    print(f"Error in region-based enumeration: {str(e)}")
                # Fall through to basic enumeration
        
        # For other cases, use basic enumeration
        prompt = f"""Convert this search space description into a complete list of specific items to search.
Do not use ellipsis (...) - list ALL items. Return ONLY a valid JSON array.

For example:
Query: "all US States"
["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

Query: "all CA counties"
["Alameda", "Alpine", "Amador", "Butte", "Calaveras", "Colusa", "Contra Costa", "Del Norte", "El Dorado", "Fresno", "Glenn", "Humboldt", "Imperial", "Inyo", "Kern", "Kings", "Lake", "Lassen", "Los Angeles", "Madera", "Marin", "Mariposa", "Mendocino", "Merced", "Modoc", "Mono", "Monterey", "Napa", "Nevada", "Orange", "Placer", "Plumas", "Riverside", "Sacramento", "San Benito", "San Bernardino", "San Diego", "San Francisco", "San Joaquin", "San Luis Obispo", "San Mateo", "Santa Barbara", "Santa Clara", "Santa Cruz", "Shasta", "Sierra", "Siskiyou", "Solano", "Sonoma", "Stanislaus", "Sutter", "Tehama", "Trinity", "Tulare", "Tuolumne", "Ventura", "Yolo", "Yuba"]

Query: "all years between 2001 and 2005"
["2001", "2002", "2003", "2004", "2005"]

Query: "all US state governors"
Return a complete JSON array of ALL US state governors. Do not use ellipsis.

Query: "top 5 tech companies"
["Apple", "Microsoft", "Google", "Amazon", "Meta"]

Query: "{search_space}"
"""
        try:
            result = await self._generate_and_parse_json(prompt)
            if not result or not isinstance(result, list):
                raise ValueError("Expected JSON array of items")
            
            if self.verbose:
                print("\nEnumerated search space:")
                print(json.dumps(result, indent=4))
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"\n[Error] Failed to enumerate search space: {str(e)}")
            raise RuntimeError(f"Error enumerating search space with Gemini: {str(e)}")
    
    def _build_extraction_prompt(self, text: str, entity_type: str, attributes: List[str]) -> str:
        """Build a prompt for extracting attributes from text."""
        prompt = f"""Extract information about {entity_type} from this text. Return ONLY a JSON object with these fields: {', '.join(attributes)}.
If you cannot find valid information for the requested entity type, return an empty object {{}}.

Here are the requirements for each field:
- name: The full name of the {entity_type}
- address: A complete address with street, city, and state

Example valid outputs:
{{
    "name": "Elite Fitness Center",
    "address": "123 Main Street, Boston, MA 02108"
}}

or if no valid information is found:
{{}}

Text to analyze:
{text}

Return ONLY the JSON object, no other text:"""
        return prompt
    
    async def parse_search_result(self, text: str, entity_type: str, attributes: List[str]) -> Optional[Dict[str, Any]]:
        """Parse search result text into structured data based on requested attributes."""
        try:
            if self.verbose:
                print(f"\n[Search Result]:")
                print(text[:200] + "..." if len(text) > 200 else text)
            
            result = await self._generate_and_parse_json(
                self._build_extraction_prompt(text, entity_type, attributes)
            )
            
            if self.verbose and result:
                print("\n[Extracted Data]:")
                print(json.dumps(result, indent=2))
            elif self.verbose:
                print("[No Data Extracted]")
                
            return result if result else None
        except Exception as e:
            if self.verbose:
                print(f"\n[Error] Failed to parse search result: {str(e)}")
            return None
    
    async def extract_attribute(self, entity_name: str, attribute: str, text: str) -> Optional[dict]:
        """Extract a specific attribute from text about an entity."""
        prompt = f"""
        You are an expert at extracting specific information from text.
        
        Find the {attribute} information for this entity:
        Name: "{entity_name}"
        
        Search this text:
        ---
        {text}
        ---
        
        Important:
        1. Only extract information that clearly refers to {entity_name}
        2. Do not extract information about other entities
        3. Do not make up or guess at information
        4. Return an empty object {{}} if you cannot find relevant information
        """
        
        if attribute == "address":
            prompt += """
        Return a valid JSON object with this field:
        {{
          "address": {{
            "street_address": "Full street number and name (e.g., '123 Main St')",
            "city": "City name only (e.g., 'Los Angeles')",
            "state": "State abbreviation (e.g., 'CA')",
            "zip_code": "5-digit ZIP code (e.g., '90210')"
          }}
        }}
        
        Note: Include any address components you find, even if incomplete. For example:
        {{
          "address": {{
            "city": "San Francisco",
            "state": "CA"
          }}
        }}
        """
        elif attribute == "contact":
            prompt += """
        Return a valid JSON object with this field:
        {{
          "contact": {{
            "phone": "Full phone number with area code",
            "email": "Complete email address"
          }}
        }}
        
        Note: Include either phone or email if found, doesn't need both:
        {{
          "contact": {{
            "phone": "(555) 123-4567"
          }}
        }}
        """
        elif attribute == "hours":
            prompt += """
        Return a valid JSON object with this field:
        {{
          "hours": {{
            "monday": "Hours in format '9:00 AM - 5:00 PM'",
            "tuesday": "Hours in format '9:00 AM - 5:00 PM'",
            "wednesday": "Hours in format '9:00 AM - 5:00 PM'",
            "thursday": "Hours in format '9:00 AM - 5:00 PM'",
            "friday": "Hours in format '9:00 AM - 5:00 PM'",
            "saturday": "Hours in format '9:00 AM - 5:00 PM'",
            "sunday": "Hours in format '9:00 AM - 5:00 PM'"
          }}
        }}
        
        Note: Include any days you find hours for, omit others:
        {{
          "hours": {{
            "monday": "6:00 AM - 10:00 PM",
            "saturday": "8:00 AM - 8:00 PM"
          }}
        }}
        """
        else:
            prompt += f"""
        Return a valid JSON object with this field:
        {{
          "{attribute}": "The {attribute} information that specifically refers to {entity_name}"
        }}
        """
        
        try:
            text = await self._generate_with_retry(prompt)
            text = self._clean_json_text(text)
            
            data = json.loads(text)
            if not data:  # Empty object
                return None
            
            # Convert nested dictionaries to proper models if needed
            if attribute == "address" and "address" in data and isinstance(data["address"], dict):
                data["address"] = Address(**data["address"])
            elif attribute == "contact" and "contact" in data and isinstance(data["contact"], dict):
                data["contact"] = Contact(**data["contact"])
            elif attribute == "hours" and "hours" in data and isinstance(data["hours"], dict):
                data["hours"] = Hours(**data["hours"])
            
            return data
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"JSON decode error at position {e.pos}")
                print(f"Response text: {text}")
            return None
        except Exception as e:
            if self.verbose:
                print(f"Error extracting {attribute}: {str(e)}")
            return None
