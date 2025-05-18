import requests
import json
from typing import Dict, List, Optional
import time
from datetime import datetime

class CS2APIClient:
    def __init__(self, base_url: str = "http://localhost:8080/cs2api"):
        """Initialize the CS2 API client.
        
        Args:
            base_url (str): Base URL for the CS2 API. Defaults to localhost.
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_all_skins(self) -> List[Dict]:
        """Get all skins from the API.
        
        Returns:
            List[Dict]: List of skin data dictionaries
        """
        response = self.session.get(f"{self.base_url}/skins")
        response.raise_for_status()
        return response.json()
    
    def get_skin_by_name(self, name: str) -> Optional[Dict]:
        """Get a specific skin by its name.
        
        Args:
            name (str): Name of the skin to search for
            
        Returns:
            Optional[Dict]: Skin data if found, None otherwise
        """
        response = self.session.get(f"{self.base_url}/skins/search/n", params={"name": name})
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    
    def get_collections(self) -> List[str]:
        """Get all available collections.
        
        Returns:
            List[str]: List of collection names
        """
        response = self.session.get(f"{self.base_url}/collections")
        response.raise_for_status()
        return response.json()
    
    def stream_skin_updates(self, interval: int = 3600):
        """Stream skin updates at regular intervals.
        
        Args:
            interval (int): Time between updates in seconds. Defaults to 1 hour.
        """
        last_data = None
        
        while True:
            try:
                current_data = self.get_all_skins()
                
                if last_data is not None:
                    # Compare with previous data to detect changes
                    changes = self._detect_changes(last_data, current_data)
                    if changes:
                        print(f"Changes detected at {datetime.now()}:")
                        for change in changes:
                            print(json.dumps(change, indent=2))
                
                last_data = current_data
                time.sleep(interval)
                
            except Exception as e:
                print(f"Error during streaming: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def _detect_changes(self, old_data: List[Dict], new_data: List[Dict]) -> List[Dict]:
        """Detect changes between two sets of skin data.
        
        Args:
            old_data (List[Dict]): Previous skin data
            new_data (List[Dict]): Current skin data
            
        Returns:
            List[Dict]: List of changes detected
        """
        changes = []
        
        # Create lookup dictionaries for faster comparison
        old_lookup = {skin['id']: skin for skin in old_data}
        new_lookup = {skin['id']: skin for skin in new_data}
        
        # Check for price changes
        for skin_id, new_skin in new_lookup.items():
            if skin_id in old_lookup:
                old_skin = old_lookup[skin_id]
                if new_skin['price'] != old_skin['price'] or new_skin['stattrack_price'] != old_skin['stattrack_price']:
                    changes.append({
                        'type': 'price_change',
                        'skin_id': skin_id,
                        'name': new_skin['name'],
                        'old_price': old_skin['price'],
                        'new_price': new_skin['price'],
                        'old_stattrack': old_skin['stattrack_price'],
                        'new_stattrack': new_skin['stattrack_price']
                    })
        
        return changes

# Example usage
if __name__ == "__main__":
    # Initialize the client
    client = CS2APIClient()
    
    # Example: Get all skins
    skins = client.get_all_skins()
    print(f"Total skins available: {len(skins)}")
    
    # Example: Search for a specific skin
    ak_skin = client.get_skin_by_name("AK-47")
    if ak_skin:
        print(f"Found AK-47 skin: {json.dumps(ak_skin, indent=2)}")
    
    # Example: Get all collections
    collections = client.get_collections()
    print(f"Available collections: {collections}")
    
    # Example: Start streaming updates
    print("Starting to stream skin updates...")
    client.stream_skin_updates(interval=300)  # Check every 5 minutes 