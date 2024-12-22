"""Gemini AI client for query parsing and result extraction."""

import asyncio
import json
import os
from typing import Dict, List, Optional

import vertexai
from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel

from .models import SearchQuery


class GeminiClient:
    """Client for interacting with Gemini AI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini client."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided")
        
        # Initialize Vertex AI with credentials
        vertexai.init(project="your-project-id")
        self.model = TextGenerationModel.from_pretrained("gemini-2.0-flash-exp")
        
    async def parse_query(self, query: str) -> SearchQuery:
        """Parse a natural language query into structured components."""
        prompt = f"""
        Parse the following search query into structured components:
        Query: "{query}"
        
        Return a valid JSON object with these fields:
        - entities: The main target entities to search for
        - entity_attributes: List of attributes to extract
        - search_space: The space to search within
        
        Example format:
        {{
            "entities": "athletic centers",
            "entity_attributes": ["name", "address"],
            "search_space": "all California zip codes"
        }}
        
        Ensure the response is ONLY the JSON object, with no additional text.
        """
        
        try:
            response = await self._generate_text(prompt)
            data = json.loads(response)
            return SearchQuery(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Gemini response as JSON: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error parsing query with Gemini: {str(e)}")
    
    async def parse_search_result(self, text: str) -> Dict:
        """Parse search result text into structured data."""
        prompt = f"""
        Extract structured information from the following text:
        {text}
        
        Return a valid JSON object with these fields where available:
        - name: Name of the entity
        - address: {{
            street_address: Street address
            city: City name
            state: State name
            zip_code: ZIP code
          }}
        - website_url: Website URL if present
        - additional_details: Any other relevant information
        
        Ensure the response is ONLY the JSON object, with no additional text.
        If a field is not found in the text, omit it from the JSON rather than including null or empty values.
        """
        
        try:
            response = await self._generate_text(prompt)
            return json.loads(response)
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
            response = await self._generate_text(prompt)
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Gemini response as JSON: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error enumerating search space with Gemini: {str(e)}")
    
    async def _generate_text(self, prompt: str) -> str:
        """Generate text using Gemini model with async support."""
        loop = asyncio.get_event_loop()
        try:
            # Run the synchronous Gemini call in a thread pool
            response = await loop.run_in_executor(
                None,
                lambda: self.model.predict(
                    prompt,
                    temperature=0.1,  # Low temperature for more deterministic responses
                    max_output_tokens=1024,
                    top_k=1,
                    top_p=0.8,
                ).text
            )
            return response.strip()
        except Exception as e:
            raise RuntimeError(f"Error generating text with Gemini: {str(e)}")
