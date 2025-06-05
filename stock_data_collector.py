#!/usr/bin/env python3
"""
Stock Data Collector with Market Cap Filtering
Collects financial data for US stocks with market cap > $1B
Includes: debt/assets, market cap, price, book-to-market, earnings yield
"""

import requests
import pandas as pd
import time
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import json
import os
import argparse

# API Configuration
API_KEY = "7cNMpVzb43GKtm05iRTDWJtyJXSylX8J"

# Rate limiting configuration
API_CALLS_PER_MINUTE = 750
SECONDS_PER_CALL = 60 / API_CALLS_PER_MINUTE  # 0.08 seconds per call

# Market cap threshold (1 billion)
MARKET_CAP_THRESHOLD = 1e9

def get_json(url: str, params: Dict[str, Any] = {}) -> Optional[Any]:
    """Safely get JSON data from API with error handling and rate limit retry"""
    try:
        params["apikey"] = API_KEY
        response = requests.get(url, params=params, timeout=10)
        
        # Handle rate limiting
        if response.status_code == 429:
            print(f"⚠️  Rate limit hit! Waiting 30 seconds...")
            time.sleep(30)
            return get_json(url, params)  # Retry
            
        response.raise_for_status()
        js = response.json()
        
        # Handle different response formats
        if isinstance(js, dict) and "historical" in js:
            return js["historical"]
        elif isinstance(js, list):
            return js
        else:
            return js
            
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error {e.response.status_code}: {e}")
        return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def check_market_cap(ticker: str, year: int) -> Tuple[bool, Optional[float]]:
    """Check if ticker had market cap above threshold in given year"""
    try:
        # Get market cap for the year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        mc_data = get_json(
            f"https://financialmodelingprep.com/api/v3/historical-market-capitalization/{ticker}",
            {"from": start_date, "to": end_date}
        )
        
        if not mc_data:
            return False, None
            
        # Calculate average market cap for the year
        mc_df = pd.DataFrame(mc_data)
        avg_market_cap = mc_df["marketCap"].mean()
        
        return avg_market_cap > MARKET_CAP_THRESHOLD, avg_market_cap
        
    except Exception as e:
        print(f"Error checking market cap for {ticker}: {e}")
        return False, None

def process_ticker_year(ticker: str, year: int) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], int]:
    """Process data for a single ticker for a specific year"""
    error_log = {"ticker": ticker, "year": year, "errors": []}
    api_calls = 0
    
    try:
        # First check market cap (1 API call)
        is_large_cap, avg_market_cap = check_market_cap(ticker, year)
        api_calls += 1
        
        if not is_large_cap:
            error_log["errors"].append(f"Market cap below threshold (avg: ${avg_market_cap:,.0f})")
            return None, error_log, api_calls
        
        # Calculate date range for the specific year
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        
        # Get balance sheet data (API call 2)
        bs = get_json(
            f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}", 
            {"period": "quarter", "limit": 20}
        )
        api_calls += 1
        
        # Get income statement data for EPS (API call 3)
        inc = get_json(
            f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}",
            {"period": "quarter", "limit": 20}
        )
        api_calls += 1
        
        # Get market cap data (API call 4)
        mc = get_json(
            f"https://financialmodelingprep.com/api/v3/historical-market-capitalization/{ticker}", 
            {"from": start_date.strftime("%Y-%m-%d"), "to": end_date.strftime("%Y-%m-%d")}
        )
        api_calls += 1
        
        # Get price data (API call 5)
        px = get_json(
            f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}", 
            {"from": start_date.strftime("%Y-%m-%d"), "to": end_date.strftime("%Y-%m-%d")}
        )
        api_calls += 1
        
        # Get company profile (API call 6)
        profile = get_json(f"https://financialmodelingprep.com/api/v3/profile/{ticker}")
        api_calls += 1
        
        # Track missing data
        if not bs:
            error_log["errors"].append("No balance sheet data")
        if not inc:
            error_log["errors"].append("No income statement data")
        if not mc:
            error_log["errors"].append("No market cap data")
        if not px:
            error_log["errors"].append("No price data")
        if not profile:
            error_log["errors"].append("No profile data")
            
        if not all([bs, inc, mc, px, profile]):
            return None, error_log, api_calls
        
        # Extract company info
        industry = profile[0].get("industry", "Unknown")
        sector = profile[0].get("sector", "Unknown")
        
        # Process balance sheet data
        bs_df = pd.DataFrame(bs)
        bs_df['date'] = pd.to_datetime(bs_df['date'])
        # Filter for specific year
        bs_df = bs_df[bs_df['date'].dt.year == year]
        
        if len(bs_df) == 0:
            error_log["errors"].append(f"No balance sheet data for year {year}")
            return None, error_log, api_calls
            
        bs_df = (
            bs_df[['date', 'shortTermDebt', 'longTermDebt', 'totalAssets',
                   'totalStockholdersEquity', 'commonStockSharesOutstanding']]
            .assign(
                quarter=lambda d: d.date.dt.to_period("Q"),
                debt_to_assets=lambda d: (
                    (d.shortTermDebt.fillna(0) + d.longTermDebt.fillna(0)) /
                    d.totalAssets.replace(0, pd.NA)
                ),
                book_value=lambda d: d.totalStockholdersEquity
            )
            .dropna(subset=["debt_to_assets"])
        )
        
        # Process income statement data
        inc_df = pd.DataFrame(inc)
        inc_df['date'] = pd.to_datetime(inc_df['date'])
        inc_df = inc_df[inc_df['date'].dt.year == year]
        
        inc_df = (
            inc_df[['date', 'eps', 'weightedAverageShsOut']]
            .assign(quarter=lambda d: d.date.dt.to_period("Q"))
            .rename(columns={"weightedAverageShsOut": "shares_outstanding"})
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
            [['quarter', 'mkt_cap']]
        )
        
        # Process price data (using adjusted close)
        px_df = (
            pd.DataFrame(px)
            .assign(
                date=lambda d: pd.to_datetime(d.date),
                quarter=lambda d: d.date.dt.to_period("Q")
            )
            .sort_values("date")
            .drop_duplicates("quarter", keep="last")
            .rename(columns={"adjClose": "stock_price"})  # Using adjusted close
            [['quarter', 'stock_price']]
        )
        
        # Merge all data
        merged = (
            bs_df.merge(inc_df, on="quarter", how="left")
                 .merge(mc_df, on="quarter", how="left")
                 .merge(px_df, on="quarter", how="left")
        )
        
        # Calculate book-to-market and earnings yield
        merged = merged.assign(
            ticker=ticker,
            industry=industry,
            sector=sector,
            # Book-to-Market = Book Value per Share / Price per Share
            book_to_market=lambda d: (
                d.book_value
                /
                d.shares_outstanding.fillna(d.commonStockSharesOutstanding).replace(0, pd.NA)
            ) / d.stock_price,
            # Earnings Yield = EPS / Price
            earnings_yield=lambda d: d.eps / d.stock_price
        )
        
        # Select final columns
        merged = merged[[
            'quarter', 'ticker', 'industry', 'sector', 'debt_to_assets', 
            'mkt_cap', 'stock_price', 'book_to_market', 'earnings_yield'
        ]].dropna()
        
        if len(merged) == 0:
            error_log["errors"].append("No valid data after merging")
            return None, error_log, api_calls
            
        return merged, error_log, api_calls
        
    except Exception as e:
        error_log["errors"].append(f"Exception: {str(e)}")
        return None, error_log, api_calls

def collect_year_data(tickers: List[str], year: int, max_tickers: Optional[int] = None, 
                     save_progress: bool = True, progress_interval: int = 100) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """Collect data for multiple tickers for a specific year with strict rate limiting"""
    all_data = []
    all_errors = []
    successful_tickers = []
    failed_tickers = []
    skipped_tickers = []
    total_api_calls = 0
    
    tickers_to_process = tickers[:max_tickers] if max_tickers else tickers
    total_tickers = len(tickers_to_process)
    
    print(f"\n{'='*70}")
    print(f"  COLLECTING DATA FOR YEAR {year}")
    print(f"{'='*70}")
    print(f"Total tickers to check: {total_tickers}")
    print(f"Market cap filter: >${MARKET_CAP_THRESHOLD/1e9:.0f}B")
    print(f"API rate limit: {API_CALLS_PER_MINUTE} calls/minute")
    print(f"Progress saves: Every {progress_interval} tickers")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    last_call_time = time.time()
    
    for i, ticker in enumerate(tickers_to_process):
        # Progress update
        if i > 0 and i % 20 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (total_tickers - i) * avg_time
            success_rate = len(successful_tickers) / (len(successful_tickers) + len(failed_tickers) + len(skipped_tickers)) * 100 if (len(successful_tickers) + len(failed_tickers) + len(skipped_tickers)) > 0 else 0
            
            print(f"\n[Progress: {i}/{total_tickers} ({i/total_tickers*100:.1f}%)]")
            print(f"  Time: {elapsed/60:.1f}min elapsed, ~{remaining/60:.1f}min remaining")
            print(f"  Success: {len(successful_tickers)}, Failed: {len(failed_tickers)}, Skipped (small cap): {len(skipped_tickers)}")
            print(f"  API calls: {total_api_calls} ({total_api_calls/elapsed*60:.0f}/minute avg)")
            print(f"  Current batch: ", end="")
        
        # Rate limiting
        time_since_last_call = time.time() - last_call_time
        if time_since_last_call < SECONDS_PER_CALL:
            time.sleep(SECONDS_PER_CALL - time_since_last_call)
        last_call_time = time.time()
        
        # Process ticker
        ticker_data, error_log, api_calls = process_ticker_year(ticker, year)
        total_api_calls += api_calls
        
        if ticker_data is not None and len(ticker_data) > 0:
            all_data.append(ticker_data)
            successful_tickers.append(ticker)
            print("✓", end="", flush=True)
        elif any("Market cap below threshold" in err for err in error_log.get("errors", [])):
            skipped_tickers.append(ticker)
            print("○", end="", flush=True)
        else:
            failed_tickers.append(ticker)
            all_errors.append(error_log)
            print("✗", end="", flush=True)
        
        # Save progress periodically
        if save_progress and (i + 1) % progress_interval == 0 and all_data:
            temp_df = pd.concat(all_data, ignore_index=True)
            temp_df['mkt_cap_rank'] = temp_df.groupby('quarter')['mkt_cap'].rank(method='dense', ascending=False).astype(int)
            progress_filename = f"progress_{year}_tickers_{i+1}.csv"
            temp_df.to_csv(progress_filename, index=False)
            print(f"\n  💾 Progress saved: {progress_filename} ({len(temp_df)} rows)")
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"\n\n{'='*70}")
    print(f"  YEAR {year} COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Successful: {len(successful_tickers)} tickers")
    print(f"Failed: {len(failed_tickers)} tickers")
    print(f"Skipped (small cap): {len(skipped_tickers)} tickers")
    print(f"Total API calls: {total_api_calls:,} ({total_api_calls/total_time*60:.0f}/minute avg)")
    
    if len(all_data) == 0:
        print("\n⚠️  No data collected!")
        return pd.DataFrame(columns=["quarter", "ticker", "industry", "sector", 
                                    "debt_to_assets", "mkt_cap", "stock_price", 
                                    "book_to_market", "earnings_yield", "mkt_cap_rank"]), all_errors, {
            "total_api_calls": total_api_calls,
            "successful_tickers": successful_tickers,
            "failed_tickers": failed_tickers,
            "skipped_tickers": skipped_tickers
        }
    
    # Combine all data
    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.sort_values(['ticker', 'quarter']).reset_index(drop=True)
    
    # Add market cap ranking
    final_df['mkt_cap_rank'] = final_df.groupby('quarter')['mkt_cap'].rank(method='dense', ascending=False).astype(int)
    
    print(f"\n📊 Final dataset: {len(final_df)} rows, {final_df['ticker'].nunique()} tickers")
    print(f"   Quarters: {sorted(final_df['quarter'].unique())}")
    
    # Save error log
    if all_errors:
        error_filename = f"errors_{year}.json"
        with open(error_filename, 'w') as f:
            json.dump(all_errors, f, indent=2, default=str)
        print(f"\n📝 Error log saved: {error_filename} ({len(all_errors)} errors)")
    
    # Clean up progress files
    if save_progress:
        for progress_file in [f for f in os.listdir('.') if f.startswith(f'progress_{year}_')]:
            os.remove(progress_file)
        print(f"🧹 Cleaned up progress files")
    
    return final_df, all_errors, {
        "total_api_calls": total_api_calls,
        "successful_tickers": successful_tickers,
        "failed_tickers": failed_tickers,
        "skipped_tickers": skipped_tickers
    }

def estimate_collection_stats(num_tickers: int, success_rate: float = 0.15) -> Dict[str, Any]:
    """Estimate API calls and storage for data collection"""
    # Estimate number of large cap stocks (>$1B market cap)
    # Historically about 15-20% of all stocks are large cap
    estimated_large_cap = int(num_tickers * success_rate)
    
    # API calls per ticker: 1 (market cap check) + 5 (data) for large cap
    api_calls_small_cap = num_tickers - estimated_large_cap  # Just 1 call each
    api_calls_large_cap = estimated_large_cap * 6  # 6 calls each
    total_api_calls = api_calls_small_cap + api_calls_large_cap
    
    # Time estimate
    time_minutes = total_api_calls / API_CALLS_PER_MINUTE
    
    # Storage estimate (rough approximation)
    # Each ticker-quarter generates about 500 bytes
    # Large cap stocks have 4 quarters per year
    rows_per_year = estimated_large_cap * 4
    bytes_per_row = 500
    size_mb = (rows_per_year * bytes_per_row) / (1024 * 1024)
    
    return {
        "total_tickers": num_tickers,
        "estimated_large_cap": estimated_large_cap,
        "total_api_calls": total_api_calls,
        "time_minutes": time_minutes,
        "time_hours": time_minutes / 60,
        "size_mb": size_mb,
        "rows": rows_per_year
    }

def main():
    parser = argparse.ArgumentParser(description='Collect stock financial data by year')
    parser.add_argument('year', type=int, help='Year to collect data for (2019-2024)')
    parser.add_argument('--max-tickers', type=int, default=None, help='Maximum number of tickers to process (for testing)')
    parser.add_argument('--ticker-file', type=str, default=None, help='File containing list of tickers (one per line)')
    parser.add_argument('--estimate-only', action='store_true', help='Only show estimates, don\'t collect data')
    args = parser.parse_args()
    
    # Validate year
    if args.year < 2019 or args.year > 2024:
        print(f"Error: Year must be between 2019 and 2024")
        return
    
    # Get tickers
    if args.ticker_file and os.path.exists(args.ticker_file):
        print(f"Loading tickers from {args.ticker_file}...")
        with open(args.ticker_file, 'r') as f:
            us_tickers = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(us_tickers)} tickers from file")
    else:
        # Get list of US tickers from API
        print("Fetching ticker list from API...")
        tickers_data = get_json("https://financialmodelingprep.com/api/v3/stock/list")
        
        if tickers_data:
            # Filter for US exchanges and remove penny stocks
            us_tickers = [
                d["symbol"] for d in tickers_data 
                if d["exchangeShortName"] in ["NYSE", "NASDAQ"] 
                and (d.get("price") is not None and d.get("price", 0) > 5)
                and len(d["symbol"]) <= 5
                and "." not in d["symbol"]
            ]
            
            print(f"✅ Found {len(us_tickers)} US tickers")
            
            # Save ticker list for future use
            with open("us_tickers.txt", 'w') as f:
                for ticker in us_tickers:
                    f.write(f"{ticker}\n")
            print(f"   Saved to us_tickers.txt")
        else:
            print("❌ Failed to fetch ticker list")
            return
    
    # Apply max tickers limit if specified
    if args.max_tickers:
        us_tickers = us_tickers[:args.max_tickers]
    
    # Show estimates
    print(f"\n📊 Estimates for year {args.year}:")
    stats = estimate_collection_stats(len(us_tickers))
    print(f"   Total tickers to check: {stats['total_tickers']:,}")
    print(f"   Estimated large cap (>$1B): ~{stats['estimated_large_cap']:,} stocks")
    print(f"   Estimated API calls: ~{stats['total_api_calls']:,}")
    print(f"   Estimated time: ~{stats['time_hours']:.1f} hours")
    print(f"   Estimated data size: ~{stats['size_mb']:.1f} MB")
    print(f"   Estimated rows: ~{stats['rows']:,}")
    
    if args.estimate_only:
        return
    
    # Confirm before proceeding
    if not args.max_tickers or args.max_tickers > 100:
        response = input(f"\nProceed with collecting data for {len(us_tickers)} tickers? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Collect data
    data, errors, stats = collect_year_data(us_tickers, args.year, max_tickers=args.max_tickers)
    
    if len(data) > 0:
        filename = f"stock_data_{args.year}.csv"
        data.to_csv(filename, index=False)
        print(f"\n✅ Data saved to '{filename}'")
        
        # Show summary statistics
        print(f"\n📈 Summary Statistics:")
        print(f"   Debt/Assets - Mean: {data['debt_to_assets'].mean():.3f}, Median: {data['debt_to_assets'].median():.3f}")
        print(f"   Book/Market - Mean: {data['book_to_market'].mean():.3f}, Median: {data['book_to_market'].median():.3f}")
        print(f"   Earnings Yield - Mean: {data['earnings_yield'].mean():.3f}, Median: {data['earnings_yield'].median():.3f}")
        
        # Show top companies
        latest_quarter = data['quarter'].max()
        latest_data = data[data['quarter'] == latest_quarter]
        if len(latest_data) > 0:
            print(f"\n🏆 Top 10 companies by market cap ({latest_quarter}):")
            top_10 = latest_data.nsmallest(10, 'mkt_cap_rank')[['ticker', 'mkt_cap', 'book_to_market', 'earnings_yield', 'industry']]
            top_10['mkt_cap'] = top_10['mkt_cap'].apply(lambda x: f"${x/1e9:.1f}B")
            print(top_10.to_string(index=False))

if __name__ == "__main__":
    main() 