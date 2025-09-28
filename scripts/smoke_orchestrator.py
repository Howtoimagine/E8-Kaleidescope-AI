import os
import sys

# So the script can import the local package when run directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Tame heavy subsystems for a fast smoke
os.environ.setdefault("E8_NON_INTERACTIVE", "1")
os.environ.setdefault("E8_VAE_ENABLE", "0")
os.environ.setdefault("E8_MARKET_FEED_ENABLED", "0")
os.environ.setdefault("E8_SPACETIME_CURVATURE", "0")
os.environ.setdefault("E8_HOLOGRAPHIC_COMPRESSION", "0")
os.environ.setdefault("E8_INFO_CONSERVATION_CHECK", "0")

from e8_mind.core.mind import new_default_mind


def main():
    mind = new_default_mind(semantic_domain="Smoke Test", embed_in_dim=1536)
    # Print a couple of core attributes that should exist after init
    print("E8Mind instantiated")
    print("run_id:", getattr(mind, 'run_id', None))
    print("state_dim:", getattr(mind, 'state_dim', None))
    print("action_dim:", getattr(mind, 'action_dim', None))


if __name__ == "__main__":
    main()
