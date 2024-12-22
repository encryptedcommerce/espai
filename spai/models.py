"""Data models for the spai package."""

from typing import List, Optional
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """Structured representation of a user's search query."""
    entities: str = Field(..., description="Main target entities to search for")
    entity_attributes: List[str] = Field(..., description="Attributes to extract for each entity")
    search_space: str = Field(..., description="The space to search within")


class Address(BaseModel):
    """Structured address information."""
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None


class EntityResult(BaseModel):
    """Structured result for a single entity."""
    entity_id: str = Field(..., description="Unique identifier for the entity")
    name: str = Field(..., description="Name of the entity")
    address: Optional[Address] = None
    website_url: Optional[str] = None
    additional_details: dict = Field(default_factory=dict)
    search_space_item: str = Field(..., description="The search space item this result came from")
