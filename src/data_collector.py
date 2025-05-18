import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Optional
from api_client import CS2APIClient
import os

class SkinDataCollector:
    def __init__(self, data_dir: str = "data"):
        """Initialize the skin data collector.
        
        Args:
            data_dir (str): Directory to store collected data
        """
        self.client = CS2APIClient()
        self.data_dir = data_dir
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "processed"), exist_ok=True)
    
    def collect_current_data(self) -> pd.DataFrame:
        """Collect current skin data and save to CSV.
        
        Returns:
            pd.DataFrame: Current skin data
        """
        # Get all skins
        skins = self.client.get_all_skins()
        
        # Convert to DataFrame
        df = pd.DataFrame(skins)
        
        # Add timestamp
        df['timestamp'] = datetime.now()
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_file = os.path.join(self.data_dir, "raw", f"skins_{timestamp}.csv")
        df.to_csv(raw_file, index=False)
        
        return df
    
    def process_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process price data to extract numerical values.
        
        Args:
            df (pd.DataFrame): Raw skin data
            
        Returns:
            pd.DataFrame: Processed data with numerical prices
        """
        # Create a copy to avoid modifying original
        processed = df.copy()
        
        # Extract numerical values from price ranges
        def extract_price(price_str: str) -> float:
            if pd.isna(price_str):
                return np.nan
            # Remove $ and split by -
            prices = price_str.replace('$', '').split('-')
            # Take average of range
            return np.mean([float(p.strip()) for p in prices])
        
        # Process regular and stattrak prices
        processed['price_avg'] = processed['price'].apply(extract_price)
        processed['stattrack_price_avg'] = processed['stattrack_price'].apply(extract_price)
        
        # Calculate price premium for stattrak
        processed['stattrak_premium'] = processed['stattrack_price_avg'] - processed['price_avg']
        processed['stattrak_premium_pct'] = (processed['stattrak_premium'] / processed['price_avg']) * 100
        
        return processed
    
    def collect_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Collect and combine historical data.
        
        Args:
            days (int): Number of days of historical data to collect
            
        Returns:
            pd.DataFrame: Combined historical data
        """
        # This would be implemented to collect historical data
        # For now, we'll just collect current data
        current_data = self.collect_current_data()
        processed_data = self.process_price_data(current_data)
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_file = os.path.join(self.data_dir, "processed", f"processed_skins_{timestamp}.csv")
        processed_data.to_csv(processed_file, index=False)
        
        return processed_data

# Example usage
if __name__ == "__main__":
    collector = SkinDataCollector()
    
    # Collect and process current data
    print("Collecting current skin data...")
    current_data = collector.collect_current_data()
    print(f"Collected data for {len(current_data)} skins")
    
    # Process the data
    print("\nProcessing price data...")
    processed_data = collector.process_price_data(current_data)
    
    # Display some statistics
    print("\nPrice Statistics:")
    print(f"Average skin price: ${processed_data['price_avg'].mean():.2f}")
    print(f"Average StatTrak premium: {processed_data['stattrak_premium_pct'].mean():.2f}%")
    
    # Save the processed data
    print("\nSaving processed data...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_file = os.path.join(collector.data_dir, "processed", f"processed_skins_{timestamp}.csv")
    processed_data.to_csv(processed_file, index=False)
    print(f"Data saved to {processed_file}") 