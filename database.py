# database.py - Stub for a future database service

import random

class DatabaseService:
    def __init__(self):
        """Initialize the database stub."""
        self.mock_data = {
            1: {"name": "Sample Item 1", "value": random.randint(1, 100)},
            2: {"name": "Sample Item 2", "value": random.randint(1, 100)},
            3: {"name": "Sample Item 3", "value": random.randint(1, 100)}
        }

    def get_data(self, item_id):
        """Mock function to simulate a database query."""
        return self.mock_data.get(item_id, {"error": "Item not found"})

# If running directly, demonstrate fetching an item
if __name__ == "__main__":
    db = DatabaseService()
    print(db.get_data(1))  # Example query
