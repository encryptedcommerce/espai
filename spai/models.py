"""Data models for the spai package."""

from typing import Dict, List, Optional
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


class Contact(BaseModel):
    """Structured contact information."""
    phone: Optional[str] = None
    email: Optional[str] = None


class Hours(BaseModel):
    """Structured business hours information."""
    monday: Optional[str] = None
    tuesday: Optional[str] = None
    wednesday: Optional[str] = None
    thursday: Optional[str] = None
    friday: Optional[str] = None
    saturday: Optional[str] = None
    sunday: Optional[str] = None


class EntityResult(BaseModel):
    """Structured result for a single entity."""
    entity_id: str = Field(..., description="Unique identifier for the entity")
    search_space_item: str = Field(..., description="The search space item this result came from")
    
    # Optional structured fields
    address: Optional[Address] = None
    contact: Optional[Contact] = None
    hours: Optional[Hours] = None
    
    # Allow any additional fields
    class Config:
        extra = "allow"
