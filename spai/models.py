"""Data models for the application."""

from typing import List, Optional, Set

from pydantic import BaseModel


class Address(BaseModel):
    """Address model."""
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None

class Contact(BaseModel):
    """Contact information model."""
    phone: Optional[str] = None
    email: Optional[str] = None

class Hours(BaseModel):
    """Business hours model."""
    monday: Optional[str] = None
    tuesday: Optional[str] = None
    wednesday: Optional[str] = None
    thursday: Optional[str] = None
    friday: Optional[str] = None
    saturday: Optional[str] = None
    sunday: Optional[str] = None

class SearchQuery(BaseModel):
    """Search query model."""
    entities: str
    entity_attributes: List[str]
    search_space: str

class EntityResult(BaseModel):
    """Entity result model with dynamic attributes."""
    name: Optional[str] = None
    address: Optional[Address] = None
    contact: Optional[Contact] = None
    hours: Optional[Hours] = None
    rating: Optional[str] = None
    website: Optional[str] = None
    price: Optional[str] = None
    description: Optional[str] = None
    
    def to_flat_dict(self, requested_attributes: Set[str]) -> dict:
        """Convert the model to a flat dictionary with dot notation for nested fields.
        
        Args:
            requested_attributes: Set of attributes that were requested in the query.
        """
        result = {}
        
        # Always include name as it's the primary identifier
        result['name'] = self.name
        
        # Handle nested structures only if they were requested
        if 'address' in requested_attributes and self.address:
            for field in ['street_address', 'city', 'state', 'zip_code']:
                result[f'address_{field}'] = getattr(self.address, field)
        
        if 'contact' in requested_attributes and self.contact:
            for field in ['phone', 'email']:
                result[f'contact_{field}'] = getattr(self.contact, field)
        
        if 'hours' in requested_attributes and self.hours:
            for field in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
                result[f'hours_{field}'] = getattr(self.hours, field)
        
        # Handle simple fields only if they were requested
        for field in ['rating', 'website', 'price', 'description']:
            if field in requested_attributes and hasattr(self, field):
                result[field] = getattr(self, field)
        
        return result
    
    class Config:
        """Allow extra fields in the model."""
        extra = "allow"
