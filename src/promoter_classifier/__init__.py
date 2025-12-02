"""
Tools for classifying promoter vs non-promoter DNA sequences using transformer-based models.

This package provides utilities and models for classifying DNA sequences as either
promoters or non-promoters using advanced machine learning techniques.
"""

from typing import Dict

from .__about__ import __version__

def get_project_info() -> Dict[str, str]:
    """
    Return basic project metadata such as title, description and version.
    
    Returns:
        Dict[str, str]: A dictionary containing project metadata with keys:
            - "title": Project title
            - "description": Project description
            - "version": Project version
            - "author": Project authors
            - "author_email": Contact email
    """
    from .__about__ import __title__, __description__, __author__, __author_email__
    
    return {
        "title": __title__,
        "description": __description__,
        "version": __version__,
        "author": __author__,
        "author_email": __author_email__
    }