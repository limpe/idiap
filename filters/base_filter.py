from abc import ABC, abstractmethod

class BaseFilter(ABC):
    @abstractmethod
    def filter(self, text: str) -> str:
        """
        Filter the input text according to specific rules
        
        Args:
            text (str): Input text to be filtered
            
        Returns:
            str: Filtered text
        """
        pass
