"""Data models for espai."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EntityResult:
    """A single entity result with attributes."""
    
    name: str
    search_space: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    email: Optional[str] = None
    description: Optional[str] = None
    hours: Optional[str] = None
    rating: Optional[str] = None
    price: Optional[str] = None
    category: Optional[str] = None
