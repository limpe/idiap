from abc import ABC, abstractmethod

class BaseFilter(ABC):
    """Base class untuk implementasi filter"""
    
    @abstractmethod
    async def filter_content(self, content: str) -> str:
        """
        Method dasar yang harus diimplementasikan oleh semua filter
        """
        pass
