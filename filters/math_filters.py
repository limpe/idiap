import re
from .base_filter import BaseFilter
from .constants import MATH_SYMBOLS, SUPERSCRIPT_MAP

class MathFilter(BaseFilter):
    def __init__(self):
        self.math_symbols = MATH_SYMBOLS
        self.superscript_map = SUPERSCRIPT_MAP
    
    def _convert_superscripts(self, text: str) -> str:
        """Convert regular numbers after ^ to superscript numbers"""
        def replace_superscript(match):
            number = match.group(1)
            return ''.join(self.superscript_map.get(digit, digit) for digit in number)
        
        return re.sub(r'\^(\d+)', replace_superscript, text)
    
    def _replace_fractions(self, text: str) -> str:
        """Replace LaTeX fractions with plain text fractions"""
        # Replace \frac{num}{den} with num/den
        pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
        return re.sub(pattern, r'\1/\2', text)
    
    def _clean_math_delimiters(self, text: str) -> str:
        """Remove LaTeX math mode delimiters"""
        text = re.sub(r'\$\$(.*?)\$\$', r'\1', text)
        text = re.sub(r'\$(.*?)\$', r'\1', text)
        text = re.sub(r'\\begin\{.*?\}(.*?)\\end\{.*?\}', r'\1', text, flags=re.DOTALL)
        return text
    
    def filter(self, text: str) -> str:
        """
        Filter mathematical expressions to plain text format
        
        Args:
            text (str): Input text containing LaTeX math
            
        Returns:
            str: Filtered text with simplified math notation
        """
        # Replace all LaTeX math symbols with their plain text equivalents
        for latex, symbol in self.math_symbols.items():
            text = text.replace(latex, symbol)
        
        # Apply specific transformations
        text = self._clean_math_delimiters(text)
        text = self._convert_superscripts(text)
        text = self._replace_fractions(text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
