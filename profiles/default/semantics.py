# Minimal semantics profile to avoid fallback warnings.
# Extend with real semantic category definitions as project evolves.

SEMANTIC_CATEGORIES = {
    "general": {
        "description": "General knowledge and default reasoning category.",
        "weight": 1.0,
    }
}

def get_semantic_categories():
    return SEMANTIC_CATEGORIES
