import requests
import pandas as pd

def test_connection():
    response = requests.get("https://httpbin.org/get")
    if response.status_code == 200:
        print("API connection successful")
        print("Sample response:", response.json)
    else:
        print("Connection failed with status code: ", response.status_code)

def main():
    print("Starting API connection test...")
    test_connection()

if __name__ == "__main__":
    main()


