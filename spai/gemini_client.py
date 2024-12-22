"""Gemini AI client for query parsing and result extraction."""

import asyncio
import json
import os
from typing import Dict, List, Optional, Set

import google.generativeai as genai
from google.api_core import retry

from .models import SearchQuery, EntityResult, Address, Contact, Hours


class GeminiClient:
    """Client for interacting with Gemini AI."""
    
    MODEL_NAME = "gemini-pro"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        top_k: int = 1,
        top_p: float = 0.8,
        max_output_tokens: int = 1024,
        verbose: bool = False,
    ):
        """Initialize the Gemini client."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided")
        
        self.verbose = verbose
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Set up the model with generation config
        generation_config = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
        }
        
        self.model = genai.GenerativeModel(
            model_name=self.MODEL_NAME,
            generation_config=generation_config
        )
    
    async def _generate_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Generate content with retry logic for rate limiting."""
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = await self.model.generate_content_async(prompt)
                if self.verbose:
                    print(f"Raw response text: {response.text}")
                return response.text
            except Exception as e:
                if "429" in str(e) and retry_count < max_retries - 1:
                    retry_count += 1
                    wait_time = 2 ** retry_count  # Exponential backoff
                    if self.verbose:
                        print(f"Rate limited, waiting {wait_time}s before retry {retry_count + 1}/{max_retries}")
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    def _clean_json_text(self, text: str) -> str:
        """Clean text to ensure it's valid JSON."""
        text = text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.endswith('```'):
            text = text[:-3]
        return text.strip()
    
    async def parse_query(self, query: str) -> SearchQuery:
        """Parse a natural language query into structured components."""
        prompt = f"""
        Parse the following search query into structured components:
        Query: "{query}"
        
        Return a valid JSON object with these fields:
        - entities: The main target entities to search for (e.g., "athletic centers", "restaurants")
        - entity_attributes: List of attributes to extract. Common attributes include:
          * name (entity name)
          * address (full address with street, city, state, zip)
          * contact (phone, email)
          * hours (business hours)
          * rating (customer ratings)
          * website (website URL)
          * price (price range or costs)
          * description (general description)
          Extract ANY attributes that are explicitly mentioned or strongly implied by the query.
        - search_space: The space to search within (e.g., "all California zip codes", "top 10 US cities")
        
        Example 1:
        Query: "Find gyms with good ratings and contact info in New York"
        {{
            "entities": "gyms",
            "entity_attributes": ["name", "rating", "contact", "address"],
            "search_space": "New York"
        }}
        
        Example 2:
        Query: "List coffee shops and their business hours in Seattle"
        {{
            "entities": "coffee shops",
            "entity_attributes": ["name", "hours", "address"],
            "search_space": "Seattle"
        }}
        
        Ensure the response is ONLY the JSON object, with no additional text.
        """
        
        try:
            text = await self._generate_with_retry(prompt)
            text = self._clean_json_text(text)
            
            data = json.loads(text)
            return SearchQuery(**data)
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"JSON decode error at position {e.pos}")
                print(f"Response text: {text}")
            raise ValueError(f"Failed to parse Gemini response as JSON. Response: {text}")
        except Exception as e:
            if self.verbose:
                print(f"Error parsing query: {str(e)}")
            raise RuntimeError(f"Error parsing query with Gemini: {str(e)}")
    
    async def parse_search_result(self, text: str, entity_attributes: List[str]) -> Optional[EntityResult]:
        """Parse search result text into structured data based on requested attributes."""
        prompt = f"""
        You are an expert at extracting structured information from text.
        
        Extract relevant information from this text:
        ---
        {text}
        ---
        
        Look carefully for:
        1. The name of the entity
        2. Any other requested attributes
        
        Return a valid JSON object with these fields:
        - name: The official or primary name of the entity
        """
        
        # Add specific field instructions based on requested attributes
        if "address" in entity_attributes:
            prompt += """
        - address: {
            street_address: The street number and name (e.g., "123 Main St"),
            city: City name only (e.g., "Los Angeles"),
            state: State abbreviation (e.g., "CA"),
            zip_code: 5-digit ZIP code (e.g., "90210")
          }
          Note: Include any address components you find, even if incomplete."""
        
        if "contact" in entity_attributes:
            prompt += """
        - contact: {
            phone: Full phone number with area code,
            email: Complete email address
          }"""
        
        if "hours" in entity_attributes:
            prompt += """
        - hours: {
            monday: Hours in format "9:00 AM - 5:00 PM",
            tuesday: Hours in format "9:00 AM - 5:00 PM",
            wednesday: Hours in format "9:00 AM - 5:00 PM",
            thursday: Hours in format "9:00 AM - 5:00 PM",
            friday: Hours in format "9:00 AM - 5:00 PM",
            saturday: Hours in format "9:00 AM - 5:00 PM",
            sunday: Hours in format "9:00 AM - 5:00 PM"
          }"""
        
        if "rating" in entity_attributes:
            prompt += "\n        - rating: Rating out of 5 stars (e.g., '4.5 stars' or '4.5/5')"
        
        if "website" in entity_attributes:
            prompt += "\n        - website: Complete URL starting with http:// or https://"
        
        if "price" in entity_attributes:
            prompt += "\n        - price: Price range (e.g., '$', '$$', '$$$') or specific costs"
        
        if "description" in entity_attributes:
            prompt += "\n        - description: Brief description of the entity and its offerings"
        
        prompt += """
        
        Important:
        1. Return ONLY the JSON object
        2. Include ONLY fields where you found information
        3. If you can't find a name or any relevant information, return an empty object {}
        4. For addresses, include any components you find, even if the address is incomplete
        5. Don't make up or guess at any information - only include what's explicitly in the text
        
        Example good response for partial information:
        {
          "name": "LA Fitness Downtown",
          "address": {
            "city": "Los Angeles",
            "state": "CA"
          }
        }
        """
        
        try:
            text = await self._generate_with_retry(prompt)
            text = self._clean_json_text(text)
            
            data = json.loads(text)
            if not data or "name" not in data:  # Must at least have a name
                return None
                
            # Convert nested dictionaries to proper models
            if "address" in data and isinstance(data["address"], dict):
                data["address"] = Address(**data["address"])
            if "contact" in data and isinstance(data["contact"], dict):
                data["contact"] = Contact(**data["contact"])
            if "hours" in data and isinstance(data["hours"], dict):
                data["hours"] = Hours(**data["hours"])
            
            return EntityResult(**data)
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"JSON decode error at position {e.pos}")
                print(f"Response text: {text}")
            return None
        except Exception as e:
            if self.verbose:
                print(f"Error parsing search result: {str(e)}")
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
        {
          "address": {
            "street_address": "Full street number and name (e.g., '123 Main St')",
            "city": "City name only (e.g., 'Los Angeles')",
            "state": "State abbreviation (e.g., 'CA')",
            "zip_code": "5-digit ZIP code (e.g., '90210')"
          }
        }
        
        Note: Include any address components you find, even if incomplete. For example:
        {
          "address": {
            "city": "San Francisco",
            "state": "CA"
          }
        }
        """
        elif attribute == "contact":
            prompt += """
        Return a valid JSON object with this field:
        {
          "contact": {
            "phone": "Full phone number with area code",
            "email": "Complete email address"
          }
        }
        
        Note: Include either phone or email if found, doesn't need both:
        {
          "contact": {
            "phone": "(555) 123-4567"
          }
        }
        """
        elif attribute == "hours":
            prompt += """
        Return a valid JSON object with this field:
        {
          "hours": {
            "monday": "Hours in format '9:00 AM - 5:00 PM'",
            "tuesday": "Hours in format '9:00 AM - 5:00 PM'",
            "wednesday": "Hours in format '9:00 AM - 5:00 PM'",
            "thursday": "Hours in format '9:00 AM - 5:00 PM'",
            "friday": "Hours in format '9:00 AM - 5:00 PM'",
            "saturday": "Hours in format '9:00 AM - 5:00 PM'",
            "sunday": "Hours in format '9:00 AM - 5:00 PM'"
          }
        }
        
        Note: Include any days you find hours for, omit others:
        {
          "hours": {
            "monday": "6:00 AM - 10:00 PM",
            "saturday": "8:00 AM - 8:00 PM"
          }
        }
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
    
    async def enumerate_search_space(self, search_space: str) -> List[str]:
        """Convert a search space description into a list of specific items."""
        prompt = f"""
        Break down this description into a list of specific items:
        "{search_space}"
        
        Return a JSON array containing each individual item.
        The items should be as specific and atomic as possible.
        Do not include extra text or formatting - just the essential identifier for each item.
        
        Examples:
        "all California zip codes" → ["90001", "94102", "92101", ...]
        "all US States" → ["CA", "AZ", "OR", ...]
        "all years between 2001 and 2005" → ["2001", "2002", "2003", "2004", "2005"]
        "all US state governors" → ["Kay Ivey", "Mike Dunleavy", "Katie Hobbs", ...]
        "top 5 tech companies" → ["Apple", "Microsoft", "Google", "Amazon", "Meta"]
        "primary colors" → ["red", "blue", "yellow"]
        
        Important:
        1. Return ONLY the JSON array
        2. Keep items as concise as possible (e.g., "CA" not "California")
        3. Limit to 20-30 items for large sets to keep the search efficient
        4. For ranges (years, numbers), include all items in the range
        5. For finite sets (colors, states), include all items
        """
        
        try:
            text = await self._generate_with_retry(prompt)
            text = self._clean_json_text(text)
            
            items = json.loads(text)
            if not isinstance(items, list):
                raise ValueError("Expected JSON array of items")
            
            return items
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"JSON decode error at position {e.pos}")
                print(f"Response text: {text}")
            raise ValueError(f"Failed to parse Gemini response as JSON. Response: {text}")
        except Exception as e:
            if self.verbose:
                print(f"Error enumerating search space: {str(e)}")
            raise RuntimeError(f"Error enumerating search space with Gemini: {str(e)}")
