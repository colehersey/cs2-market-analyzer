from api_client import CS2APIClient
import json

def test_api_connection():
    try:
        # Initialize the client
        client = CS2APIClient()
        
        # Test 1: Get all skins
        print("Testing: Get all skins...")
        skins = client.get_all_skins()
        print(f"✓ Successfully retrieved {len(skins)} skins")
        
        # Test 2: Get collections
        print("\nTesting: Get collections...")
        collections = client.get_collections()
        print(f"✓ Successfully retrieved {len(collections)} collections")
        print("Collections:", json.dumps(collections, indent=2))
        
        # Test 3: Search for a specific skin
        print("\nTesting: Search for AK-47...")
        ak_skin = client.get_skin_by_name("AK-47")
        if ak_skin:
            print("✓ Found AK-47 skin:")
            print(json.dumps(ak_skin, indent=2))
        else:
            print("✗ AK-47 skin not found")
        
        print("\nAll tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the CS2 API is running (docker compose up)")
        print("2. Check if you can access http://localhost:8080/cs2api in your browser")
        print("3. Verify your network connection")
        print("4. Check if port 8080 is available")
        return False

if __name__ == "__main__":
    test_api_connection() 