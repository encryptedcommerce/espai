"""Data models for espai."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchResult:
    """Search result from a provider."""
    title: str
    url: str
    snippet: str
    domain: str
    published_date: Optional[str] = None


@dataclass
class EntityResult:
    """Extracted entity data."""
    name: str
    search_space: str
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    website: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict, search_space: str) -> "EntityResult":
        """Create an EntityResult from a dictionary.
        
        Args:
            data: Dictionary of entity data
            search_space: Search space this entity was found in
            
        Returns:
            EntityResult instance
        """
        # Normalize name
        name = data.get("name", "").strip()
        if not name:
            return None
            
        # Parse address if present
        address = data.get("address", "")
        street = city = state = zip_code = None
        if address:
            parts = [p.strip() for p in address.split(",")]
            if len(parts) >= 3:  # street, city, state zip
                street = parts[0]
                city = parts[1]
                state_zip = parts[2].strip().split()
                if len(state_zip) >= 2:
                    state = state_zip[0]
                    zip_code = state_zip[1]
            elif len(parts) == 2:  # city, state zip
                city = parts[0]
                state_zip = parts[1].strip().split()
                if len(state_zip) >= 2:
                    state = state_zip[0]
                    zip_code = state_zip[1]
        
        return cls(
            name=name,
            search_space=search_space,
            street_address=street,
            city=city,
            state=state,
            zip_code=zip_code,
            website=data.get("website"),
            phone=data.get("phone"),
            email=data.get("email")
        )
    
    def __eq__(self, other: object) -> bool:
        """Compare two EntityResults for equality."""
        if not isinstance(other, EntityResult):
            return NotImplemented
        return self.name.lower() == other.name.lower()
    
    def __hash__(self) -> int:
        """Hash based on normalized name."""
        return hash(self.name.lower())
