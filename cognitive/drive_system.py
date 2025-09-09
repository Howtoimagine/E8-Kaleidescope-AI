"""
Drive System - Basic motivational drives and needs tracking.

The DriveSystem tracks various internal drives that influence behavior
and decision-making in the cognitive system.
"""

from typing import Dict, List, Tuple

class DriveSystem:
    """
    Manages basic motivational drives that influence cognitive behavior.
    Drives decay over time and can be reinforced through rewards.
    """
    
    def __init__(self):
        self.drives = {
            "curiosity": 0.5,
            "coherence": 0.5, 
            "novelty": 0.5,
            "intelligibility": 0.5,
            "fluidity": 0.5
        }

    def decay(self):
        """Apply temporal decay to all drives."""
        for k in self.drives:
            self.drives[k] = max(0.0, self.drives[k] - 0.01)

    def reward(self, key: str, amount: float = 0.1):
        """Increase a specific drive by the given amount."""
        if key in self.drives:
            self.drives[key] = min(1.0, self.drives[key] + amount)

    def get_top_needs(self, k: int = 2) -> List[Tuple[str, float]]:
        """Get the k highest drive levels."""
        return sorted(self.drives.items(), key=lambda item: -item[1])[:k]
