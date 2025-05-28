"""
Fixed Multi-Ticker Data Collection Script

This script contains the fixed version of your code to pull financial data for multiple tickers.

Key fixes:
1. Proper error handling to prevent empty concatenation
2. Better API response handling for different data formats
3. Data validation and cleaning
4. Progress tracking and summary reporting
5. Rate limiting to avoid API issues
"""

import requests
import pandas as pd
import time
from typing import Optional, List, Dict, Any

API = "7cNMpVzb43GKtm05iRTDWJtyJXSylX8J"

def get_json(url: str, params: Dict[str, Any] = {}) -> Optional[List[Dict]]:
    """Safely get JSON data from API with error handling"""
    try:
        params["apikey"] = API
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        js = response.json()
        
        # Handle different response formats
        if isinstance(js, dict) and "historical" in js:
            return js["historical"]
        elif isinstance(js, list):
            return js
        else:
            print(f"Unexpected response format: {type(js)}")
            return None
    except Exception as e:
        print(f"Error fetching data from {url}: {e}")
        return None

def process_ticker_data(ticker: str) -> Optional[pd.DataFrame]:
    """Process data for a single ticker and return DataFrame with required columns"""
    print(f"Processing {ticker}...")
    
    try:
        # Get balance sheet data
        bs = get_json(
            f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}", 
            {"period": "quarter", "limit": 20}
        )
        
        # Get market cap data
        mc = get_json(
            f"https://financialmodelingprep.com/api/v3/historical-market-capitalization/{ticker}", 
            {"limit": 1000}
        )
        
        # Get price data
        px = get_json(
            f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}", 
            {"serietype": "line", "timeseries": 1000}
        )
        
        # Get company profile
        profile = get_json(f"https://financialmodelingprep.com/api/v3/profile/{ticker}")
        
        # Check if all data is available
        if not all([bs, mc, px, profile]):
            print(f"Missing data for {ticker}, skipping.")
            return None
        
        # Extract company info
        industry = profile[0].get("industry", "Unknown")
        sector = profile[0].get("sector", "Unknown")
        
        # Process balance sheet data
        bs_df = (
            pd.DataFrame(bs)
            .loc[:, ["date", "shortTermDebt", "longTermDebt", "totalAssets"]]
            .assign(
                date=lambda d: pd.to_datetime(d.date),
                quarter=lambda d: d.date.dt.to_period("Q"),
                debt_to_assets=lambda d: (
                    (d.shortTermDebt.fillna(0) + d.longTermDebt.fillna(0)) / 
                    d.totalAssets.replace(0, pd.NA)
                )
            )
            .dropna(subset=["debt_to_assets"])
        )
        
        # Process market cap data
        mc_df = (
            pd.DataFrame(mc)
            .assign(
                date=lambda d: pd.to_datetime(d.date),
                quarter=lambda d: d.date.dt.to_period("Q")
            )
            .sort_values("date")
            .drop_duplicates("quarter", keep="last")
            .rename(columns={"marketCap": "mkt_cap"})
            [["quarter", "mkt_cap"]]
        )
        
        # Process price data
        px_df = (
            pd.DataFrame(px)
            .assign(
                date=lambda d: pd.to_datetime(d.date),
                quarter=lambda d: d.date.dt.to_period("Q")
            )
            .sort_values("date")
            .drop_duplicates("quarter", keep="last")
            .rename(columns={"close": "stock_price"})
            [["quarter", "stock_price"]]
        )
        
        # Merge all data
        merged = (
            bs_df.merge(mc_df, on="quarter", how="left")
                 .merge(px_df, on="quarter", how="left")
                 .assign(ticker=ticker, industry=industry, sector=sector)
                 [["quarter", "ticker", "industry", "sector", "debt_to_assets", "mkt_cap", "stock_price"]]
                 .dropna()  # Remove rows with missing data
        )
        
        if len(merged) == 0:
            print(f"No valid data after merging for {ticker}")
            return None
            
        return merged
        
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None

def get_ticker_data(tickers: List[str], max_tickers: int = 50) -> pd.DataFrame:
    """Get data for multiple tickers with proper error handling"""
    all_data = []
    successful_tickers = []
    failed_tickers = []
    
    for i, ticker in enumerate(tickers[:max_tickers]):
        print(f"\nProgress: {i+1}/{min(len(tickers), max_tickers)}")
        
        ticker_data = process_ticker_data(ticker)
        
        if ticker_data is not None and len(ticker_data) > 0:
            all_data.append(ticker_data)
            successful_tickers.append(ticker)
            print(f"✓ Successfully processed {ticker} - {len(ticker_data)} quarters")
        else:
            failed_tickers.append(ticker)
            print(f"✗ Failed to process {ticker}")
        
        # Rate limiting
        time.sleep(0.5)
    
    print(f"\n=== SUMMARY ===")
    print(f"Successful: {len(successful_tickers)} tickers")
    print(f"Failed: {len(failed_tickers)} tickers")
    
    if len(all_data) == 0:
        print("No data collected. Returning empty DataFrame.")
        return pd.DataFrame(columns=["quarter", "ticker", "industry", "sector", "debt_to_assets", "mkt_cap", "stock_price"])
    
    # Combine all data
    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.sort_values(["ticker", "quarter"]).reset_index(drop=True)
    
    print(f"Final dataset: {len(final_df)} rows, {final_df['ticker'].nunique()} unique tickers")
    return final_df

def main():
    """Main function to run the data collection"""
    
    # Get list of US tickers
    print("Fetching ticker list...")
    tickers_data = get_json("https://financialmodelingprep.com/api/v3/stock/list")
    
    if tickers_data:
        # Filter for US exchanges and remove penny stocks
        us_tickers = [
            d["symbol"] for d in tickers_data 
            if d["exchangeShortName"] in ["NYSE", "NASDAQ"] 
            and (d.get("price") is not None and d.get("price", 0) > 5)  # Filter out penny stocks and None prices
            and len(d["symbol"]) <= 5  # Filter out complex symbols
            and "." not in d["symbol"]  # Filter out preferred shares
        ]
        
        print(f"Found {len(us_tickers)} US tickers")
        print(f"Sample tickers: {us_tickers[:10]}")
    else:
        print("Failed to fetch ticker list. Using sample tickers.")
        us_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V"]
    
    # Test with a single ticker first
    print("\nTesting with a single ticker (AAPL)...")
    test_data = process_ticker_data("AAPL")
    
    if test_data is not None:
        print("\nTest successful! Sample data:")
        print(test_data.head())
        print(f"\nData types:")
        print(test_data.dtypes)
    else:
        print("Test failed. Please check your API key.")
        return
    
    # Process the tickers (start with a small number for testing)
    print("\nStarting data collection...")
    final_dataset = get_ticker_data(us_tickers, max_tickers=20)  # Start with 20 tickers
    
    # Display results
    if len(final_dataset) > 0:
        print("\n=== SAMPLE DATA ===")
        print(final_dataset.head(10))
        
        print("\n=== DATA SUMMARY ===")
        print(f"Total rows: {len(final_dataset)}")
        print(f"Unique tickers: {final_dataset['ticker'].nunique()}")
        print(f"Date range: {final_dataset['quarter'].min()} to {final_dataset['quarter'].max()}")
        print(f"Industries: {final_dataset['industry'].nunique()}")
        print(f"Sectors: {final_dataset['sector'].nunique()}")
        
        # Save to CSV
        final_dataset.to_csv("stock_data.csv", index=False)
        print("\nData saved to 'stock_data.csv'")
    else:
        print("No data was collected. Please check your API key and network connection.")

if __name__ == "__main__":
    main() 