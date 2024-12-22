"""Gemini AI client for query parsing and result extraction."""

import json
import os
from typing import Dict, List, Optional

import google.generativeai as genai

from .models import SearchQuery


class GeminiClient:
    """Client for interacting with Gemini AI."""
    
    MODEL_NAME = "gemini-2.0-flash-exp"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        top_k: int = 1,
        top_p: float = 0.8,
        max_output_tokens: int = 1024,
    ):
        """Initialize the Gemini client.
        
        Args:
            api_key: Gemini API key
            temperature: Controls randomness in the output (0.0 to 1.0)
            top_k: Number of highest probability tokens to consider
            top_p: Cumulative probability cutoff for token selection
            max_output_tokens: Maximum number of tokens to generate
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided")
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Set up the model with generation config
        self.model = genai.GenerativeModel(
            model_name=self.MODEL_NAME,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
            )
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
            response = await self.model.aio.generate_content(prompt)
            data = json.loads(response.text)
            return SearchQuery(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Gemini response as JSON: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error parsing query with Gemini: {str(e)}")
    
    async def parse_search_result(self, text: str, entity_attributes: List[str]) -> Dict:
        """Parse search result text into structured data based on requested attributes."""
        # Create a dynamic instruction for each attribute
        attribute_instructions = []
        for attr in entity_attributes:
            if attr == "address":
                attribute_instructions.append(
                    '- address: {\n'
                    '    street_address: Full street address\n'
                    '    city: City name\n'
                    '    state: State name\n'
                    '    zip_code: ZIP or postal code\n'
                    '  }'
                )
            elif attr == "contact":
                attribute_instructions.append(
                    '- contact: {\n'
                    '    phone: Phone number\n'
                    '    email: Email address\n'
                    '  }'
                )
            elif attr == "hours":
                attribute_instructions.append(
                    '- hours: {\n'
                    '    monday: Operating hours for Monday\n'
                    '    tuesday: Operating hours for Tuesday\n'
                    '    wednesday: Operating hours for Wednesday\n'
                    '    thursday: Operating hours for Thursday\n'
                    '    friday: Operating hours for Friday\n'
                    '    saturday: Operating hours for Saturday\n'
                    '    sunday: Operating hours for Sunday\n'
                    '  }'
                )
            else:
                # For simple attributes, just add them directly
                attribute_instructions.append(f'- {attr}: Relevant {attr} information from the text')
        
        prompt = f"""
        Extract structured information from the following text:
        {text}
        
        Return a valid JSON object with these fields where available:
        {chr(10).join(attribute_instructions)}
        
        Ensure the response is ONLY the JSON object, with no additional text.
        If a field is not found in the text, omit it from the JSON rather than including null or empty values.
        """
        
        try:
            response = await self.model.aio.generate_content(prompt)
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Gemini response as JSON: {str(e)}")
        except Exception as e:
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
            response = await self.model.aio.generate_content(prompt)
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Gemini response as JSON: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error enumerating search space with Gemini: {str(e)}")
