import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.database import create_tables
from src.backend.models import user, conversation

if __name__ == "__main__":
    create_tables()
    print("Tables created ✓")