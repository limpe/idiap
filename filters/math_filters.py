from typing import Dict, List
from .base_filter import BaseFilter

class MathSymbolFilter:
    """Class untuk menangani konversi simbol matematika"""
    
    @staticmethod
    def get_math_replacements() -> dict:
        """
        Returns dictionary of LaTeX math symbols and their Unicode replacements
        """
        return {
            # Basic Operations
            '\\frac': '',          
            '\\sqrt': '√',         
            '\\times': '×',        
            '\\div': '÷',          
            '\\pm': '±',           
            '\\mp': '∓',           
            '\\cdot': '·',         
            
            # Comparison Operators
            '\\leq': '≤',          
            '\\geq': '≥',          
            '\\neq': '≠',          
            '\\approx': '≈',       
            '\\equiv': '≡',        
            '\\sim': '∼',          
            
            # Powers and Indices
            '^2': '²',             
            '^3': '³',             
            '^n': 'ⁿ',            
            '_2': '₂',            
            '_3': '₃',            
            '_n': 'ₙ',            
            
            # Greek Letters
            '\\alpha': 'α',        
            '\\beta': 'β',         
            '\\gamma': 'γ',        
            '\\delta': 'δ',        
            '\\epsilon': 'ε',      
            '\\zeta': 'ζ',        
            '\\eta': 'η',         
            '\\theta': 'θ',        
            '\\iota': 'ι',        
            '\\kappa': 'κ',       
            '\\lambda': 'λ',      
            '\\mu': 'μ',          
            '\\nu': 'ν',          
            '\\xi': 'ξ',          
            '\\pi': 'π',          
            '\\rho': 'ρ',         
            '\\sigma': 'σ',       
            '\\tau': 'τ',         
            '\\upsilon': 'υ',     
            '\\phi': 'φ',         
            '\\chi': 'χ',         
            '\\psi': 'ψ',         
            '\\omega': 'ω',       
            
            # Capital Greek Letters
            '\\Gamma': 'Γ',       
            '\\Delta': 'Δ',       
            '\\Theta': 'Θ',       
            '\\Lambda': 'Λ',      
            '\\Xi': 'Ξ',          
            '\\Pi': 'Π',          
            '\\Sigma': 'Σ',       
            '\\Upsilon': 'Υ',     
            '\\Phi': 'Φ',         
            '\\Psi': 'Ψ',         
            '\\Omega': 'Ω',       
            
            # Set Theory
            '\\in': '∈',          
            '\\notin': '∉',       
            '\\subset': '⊂',      
            '\\supset': '⊃',      
            '\\subseteq': '⊆',    
            '\\supseteq': '⊇',    
            '\\cup': '∪',         
            '\\cap': '∩',         
            '\\emptyset': '∅',    
            
            # Logic Symbols
            '\\forall': '∀',      
            '\\exists': '∃',      
            '\\nexists': '∄',     
            '\\land': '∧',        
            '\\lor': '∨',         
            '\\neg': '¬',         
            '\\implies': '⇒',     
            '\\iff': '⇔',         
            '\\therefore': '∴',   
            '\\because': '∵',     
            
            # Calculus
            '\\partial': '∂',     
            '\\nabla': '∇',       
            '\\infty': '∞',       
            '\\int': '∫',         
            '\\iint': '∬',        
            '\\iiint': '∭',       
            '\\oint': '∮',        
            
            # Arrows
            '\\rightarrow': '→',   
            '\\leftarrow': '←',    
            '\\leftrightarrow': '↔',
            '\\Rightarrow': '⇒',   
            '\\Leftarrow': '⇐',    
            '\\Leftrightarrow': '⇔',
            '\\uparrow': '↑',      
            '\\downarrow': '↓',    
            
            # Miscellaneous
            '\\degree': '°',       
            '\\prime': '′',        
            '\\dprime': '″',       
            '\\tprime': '‴',       
            '\\perp': '⊥',        
            '\\parallel': '∥',     
            '\\angle': '∠',        
            '\\triangle': '△',     
            '\\square': '□',       
            '\\propto': '∝',      
            
            # Cleaning commands
            '{': '',               
            '}': '',               
            '\\left': '',          
            '\\right': '',         
            '\\text': '',          
            '\\mathrm': '',        
        }

    @staticmethod
    def get_equation_markers() -> List[str]:
        """
        Returns list of common LaTeX equation markers
        """
        return [
            '\\[', '\\]', '\\(', '\\)', 
            '\\begin{equation}', '\\end{equation}',
            '\\begin{align}', '\\end{align}',
            '\\begin{aligned}', '\\end{aligned}'
        ]

    @staticmethod
    async def clean_math_expression(text: str) -> str:
        """
        Clean mathematical expressions by converting LaTeX to Unicode
        """
        # Remove equation markers
        for marker in MathSymbolFilter.get_equation_markers():
            text = text.replace(marker, '')
            
        # Apply symbol replacements
        for latex, replacement in MathSymbolFilter.get_math_replacements().items():
            text = text.replace(latex, replacement)
        
        # Clean up spaces and line breaks
        text = ' '.join(line.strip() for line in text.split('\n') if line.strip())
        text = ' '.join(text.split())
        
        return text.strip()

    @staticmethod
    async def contains_math(text: str) -> bool:
        """
        Check if text contains mathematical expressions
        """
        math_indicators = [
            '\\[', '\\]', '\\(', '\\)', '\\frac', '^', '_', '\\sqrt',
            '\\alpha', '\\beta', '\\theta', '\\pi', '\\sum', '\\int'
        ]
        return any(indicator in text for indicator in math_indicators)
