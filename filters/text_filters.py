from typing import Optional
from .base_filter import BaseFilter
from .math_filters import MathSymbolFilter

class TextFilter(BaseFilter):
    """Menangani filter teks umum"""
    
    def __init__(self):
        self.math_filter = MathSymbolFilter()

    async def filter_content(self, content: str) -> str:
        """
        Implementasi filter_content untuk teks umum
        """
        return await self.filter_text(content)
        
    async def filter_text(self, text: str) -> str:
        """
        Filter teks dengan menangani teks biasa dan ekspresi matematika
        """
        # Basic text filtering
        filtered_text = (text
            .replace("*", "")
            .replace("#", "")
            .replace("Mistral AI", "PAIDI")
            .replace("Mistral", "PAIDI")
        )
        
        # Split into lines
        lines = filtered_text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if await self.math_filter.contains_math(line):
                cleaned_line = await self.math_filter.clean_math_expression(line)
            else:
                cleaned_line = line
            
            if cleaned_line.strip():
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines).strip()
