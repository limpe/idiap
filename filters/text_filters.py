from typing import Optional
from .base_filter import BaseFilter
from .math_filter import MathFilter

class TextFilter(BaseFilter):
    """Filter untuk menangani teks dan konten matematis"""
    
    def __init__(self):
        self.math_filter = MathFilter()

    def filter(self, text: str) -> str:
        """
        Filter teks dengan menangani teks biasa dan ekspresi matematika.
        Mengimplementasikan method abstract dari BaseFilter.
        
        Args:
            text (str): Teks yang akan difilter
            
        Returns:
            str: Teks yang sudah difilter
        """
        # Basic text filtering
        filtered_text = (text
            .replace("*", "")
            .replace("#", "")
            .replace("Mistral AI", "PAIDI")
            .replace("Mistral", "PAIDI")
        )
        
        # Split into lines untuk memproses math content per baris
        lines = filtered_text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Check if line contains math expressions
            if any(math_symbol in line for math_symbol in ['\\[', '\\]', '\\frac', '^', '_']):
                # Gunakan math filter untuk membersihkan ekspresi matematika
                cleaned_line = self.math_filter.filter(line)
            else:
                cleaned_line = line.strip()
            
            # Hanya tambahkan baris yang memiliki konten
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        # Join lines kembali dan bersihkan whitespace berlebih
        result = '\n'.join(cleaned_lines)
        result = ' '.join(result.split())  # Normalize spaces
        
        return result.strip()

    def _clean_formatting(self, text: str) -> str:
        """
        Membersihkan formatting teks yang umum
        
        Args:
            text (str): Teks yang akan dibersihkan
            
        Returns:
            str: Teks yang sudah dibersihkan
        """
        # Hapus multiple spaces
        text = ' '.join(text.split())
        
        # Hapus karakter formatting yang tidak diinginkan
        text = text.replace('`', '')
        text = text.replace('~', '')
        text = text.replace('>', '')
        text = text.replace('|', '')
        
        return text.strip()

    def _clean_special_chars(self, text: str) -> str:
        """
        Membersihkan karakter khusus yang tidak diinginkan
        
        Args:
            text (str): Teks dengan karakter khusus
            
        Returns:
            str: Teks yang sudah dibersihkan
        """
        # Daftar karakter yang ingin dihapus atau diganti
        replacements = {
            '\u200b': '',  # Zero-width space
            '\u200c': '',  # Zero-width non-joiner
            '\u200d': '',  # Zero-width joiner
            '\ufeff': '',  # Zero-width no-break space
            '&nbsp;': ' ', # HTML non-breaking space
            '&lt;': '<',   # HTML less than
            '&gt;': '>',   # HTML greater than
            '&amp;': '&',  # HTML ampersand
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
            
        return text
