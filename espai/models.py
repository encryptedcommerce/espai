"""Data models for espai."""

from dataclasses import dataclass
from typing import Optional, Dict


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
    """A result entity with its attributes."""
    name: str
    search_space: Optional[str] = None
    website: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    
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
            website=data.get("website"),
            phone=data.get("phone"),
            email=data.get("email"),
            address=address,
            street_address=street,
            city=city,
            state=state,
            zip=zip_code
        )
    
    def to_dict(self) -> Dict[str, Optional[str]]:
        """Convert to dictionary with all attributes."""
        return {
            'name': self.name,
            'search_space': self.search_space or "unknown",
            'website': self.website or "unknown",
            'phone': self.phone or "unknown",
            'email': self.email or "unknown",
            'address': self.address or "unknown",
            'street_address': self.street_address or "unknown",
            'city': self.city or "unknown",
            'state': self.state or "unknown",
            'zip': self.zip or "unknown"
        }
        
    def get_attribute(self, attr: str) -> Optional[str]:
        """Get an attribute value safely."""
        return getattr(self, attr, None) or "unknown"
        
    def set_attribute(self, attr: str, value: Optional[str]) -> None:
        """Set an attribute value safely."""
        if hasattr(self, attr):
            setattr(self, attr, value if value and value.lower() != "unknown" else None)
    
    def __eq__(self, other: object) -> bool:
        """Compare two EntityResults for equality."""
        if not isinstance(other, EntityResult):
            return NotImplemented
        return self.name.lower() == other.name.lower()
    
    def __hash__(self) -> int:
        """Hash based on normalized name."""
        return hash(self.name.lower())
