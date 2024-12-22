"""Gemini AI client for query parsing and result extraction."""

import json
import os
from typing import Dict, List, Optional

import google.generativeai as genai

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
            response = await self.model.generate_content_async(prompt)
            if self.verbose:
                print(f"Raw response text: {response.text}")
            
            # Clean the response text to ensure it's valid JSON
            text = response.text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            
            data = json.loads(text)
            return SearchQuery(**data)
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"JSON decode error at position {e.pos}")
                print(f"Response text: {response.text}")
            raise ValueError(f"Failed to parse Gemini response as JSON. Response: {response.text}")
        except Exception as e:
            if self.verbose:
                print(f"Error parsing query: {str(e)}")
                print(f"Response text: {response.text}")
            raise RuntimeError(f"Error parsing query with Gemini: {str(e)}")
    
    async def parse_search_result(self, text: str, entity_attributes: List[str]) -> EntityResult:
        """Parse search result text into structured data based on requested attributes."""
        prompt = f"""
        Extract structured information from the following text:
        {text}
        
        Return a valid JSON object with these fields where available:
        - name: Entity name or title
        """
        
        # Add specific field instructions based on requested attributes
        if "address" in entity_attributes:
            prompt += """
        - address:
            street_address: Full street address
            city: City name
            state: State name
            zip_code: ZIP or postal code"""
        
        if "contact" in entity_attributes:
            prompt += """
        - contact:
            phone: Phone number
            email: Email address"""
        
        if "hours" in entity_attributes:
            prompt += """
        - hours:
            monday: Operating hours for Monday
            tuesday: Operating hours for Tuesday
            wednesday: Operating hours for Wednesday
            thursday: Operating hours for Thursday
            friday: Operating hours for Friday
            saturday: Operating hours for Saturday
            sunday: Operating hours for Sunday"""
        
        if "rating" in entity_attributes:
            prompt += "\n        - rating: Rating information (e.g., '4.5 stars')"
        
        if "website" in entity_attributes:
            prompt += "\n        - website: Website URL"
        
        if "price" in entity_attributes:
            prompt += "\n        - price: Price range or cost information"
        
        if "description" in entity_attributes:
            prompt += "\n        - description: General description or details"
        
        prompt += """
        
        Ensure the response is ONLY the JSON object, with no additional text.
        If a field is not found in the text, omit it from the JSON rather than including null or empty values.
        Format addresses consistently, e.g., '123 Main St, City, State 12345'
        """
        
        try:
            response = await self.model.generate_content_async(prompt)
            if self.verbose:
                print(f"Raw response text: {response.text}")
            
            # Clean the response text to ensure it's valid JSON
            text = response.text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            
            data = json.loads(text)
            
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
                print(f"Response text: {response.text}")
            raise ValueError(f"Failed to parse Gemini response as JSON. Response: {response.text}")
        except Exception as e:
            if self.verbose:
                print(f"Error parsing search result: {str(e)}")
                print(f"Response text: {response.text}")
            raise RuntimeError(f"Error parsing search result with Gemini: {str(e)}")
    
    async def enumerate_search_space(self, search_space: str) -> List[str]:
        """Convert a search space description into a list of specific items."""
        prompt = f"""
        Generate a list of specific items for this search space:
        "{search_space}"
        
        Return a valid JSON array of strings representing each item in the search space.
        For example, if the search space is "all California zip codes", return an array of ZIP codes.
        If the search space is "top 10 US cities by population", return those city names.
        
        Ensure the response is ONLY the JSON array, with no additional text.
        Limit the response to a maximum of 50 items to keep the search manageable.
        """
        
        try:
            response = await self.model.generate_content_async(prompt)
            if self.verbose:
                print(f"Raw response text: {response.text}")
            
            # Clean the response text to ensure it's valid JSON
            text = response.text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            
            return json.loads(text)
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"JSON decode error at position {e.pos}")
                print(f"Response text: {response.text}")
            raise ValueError(f"Failed to parse Gemini response as JSON. Response: {response.text}")
        except Exception as e:
            if self.verbose:
                print(f"Error enumerating search space: {str(e)}")
                print(f"Response text: {response.text}")
            raise RuntimeError(f"Error enumerating search space with Gemini: {str(e)}")
