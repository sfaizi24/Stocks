import requests
import pandas as pd
import time
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import json
import os
import numpy as np # For pd.NA

# Your API key
API = "7cNMpVzb43GKtm05iRTDWJtyJXSylX8J" # Please replace with your actual key if different

# Rate limiting configuration
API_CALLS_PER_MINUTE = 300
# For the main data processing loop (BS, Price, Profile, Income Stmt)
API_CALLS_PER_TICKER_PROCESSING = 4 
TICKERS_PER_MINUTE_PROCESSING = API_CALLS_PER_MINUTE // API_CALLS_PER_TICKER_PROCESSING
SECONDS_PER_TICKER_PROCESSING = 60 / TICKERS_PER_MINUTE_PROCESSING if TICKERS_PER_MINUTE_PROCESSING > 0 else 0.8 # Default to 0.8s if calc is zero

# For individual API calls (e.g., during pre-filter)
SINGLE_CALL_DELAY = 60.0 / API_CALLS_PER_MINUTE

print(f"Rate limit configured: {API_CALLS_PER_MINUTE} API calls/minute.")
print(f"Processing: {TICKERS_PER_MINUTE_PROCESSING} tickers/minute ({SECONDS_PER_TICKER_PROCESSING:.2f} seconds/ticker, {API_CALLS_PER_TICKER_PROCESSING} calls/ticker).")
print(f"Min delay per single API call: {SINGLE_CALL_DELAY:.2f} seconds.")

# Define the target columns for the final DataFrame
TARGET_COLUMNS = [
    "quarter", "ticker", "industry", "sector", 
    "debt_to_assets", "mkt_cap", "stock_price", 
    "book_to_market", "earnings_yield", "mkt_cap_rank"
]

def get_json(url: str, params: Dict[str, Any] = {}) -> Optional[List[Dict]]:
    """Safely get JSON data from API with error handling and rate limit retry"""
    try:
        params["apikey"] = API
        response = requests.get(url, params=params, timeout=20) # Increased timeout
        
        if response.status_code == 429:
            print(f"\n⚠️ Rate limit hit! Waiting 60 seconds...")
            time.sleep(61) # Wait a bit more than 60s
            return get_json(url, params)
            
        response.raise_for_status()
        js = response.json()
        
        if isinstance(js, dict) and "historical" in js: # For historical-price-full
            return js["historical"]
        elif isinstance(js, list):
            return js
        elif isinstance(js, dict) and len(js) == 0: # Empty dict response
             print(f"Empty JSON response (dict) from {url}")
             return [] # Return empty list to signify no data, but not an error
        else:
            print(f"Unexpected response format: {type(js)} from {url}. Content: {str(js)[:200]}")
            return None
            
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error {e.response.status_code} for {e.request.url}: {e}")
        # Specific handling for 401/403 (API key issues) or 404 (Not Found)
        if e.response.status_code in [401, 403]:
            print("Critical API key error. Please check your API key.")
        elif e.response.status_code == 404:
            print("Endpoint not found or ticker does not exist.")
        return None
    except requests.exceptions.Timeout:
        print(f"Timeout occurred for {url}")
        return None
    except requests.exceptions.RequestException as e: # Catch other request exceptions
        print(f"Request exception for {url}: {e}")
        return None
    except Exception as e:
        print(f"Error fetching or parsing JSON data from {url}: {e}")
        return None

def process_ticker_year(ticker: str, year: int, prefetched_mc_data: Optional[List[Dict]] = None) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    error_log = {"ticker": ticker, "year": year, "errors": []}
    
    try:
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        
        # API Call 1: Balance Sheet
        bs_url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}"
        bs_params = {"period": "quarter", "limit": 12} # Get more quarters for robust year filtering
        bs = get_json(bs_url, bs_params)
        
        # API Call 2: Market Cap (potentially prefetched)
        if prefetched_mc_data is not None:
            mc = prefetched_mc_data
        else: # Fallback if not prefetched (should ideally not happen with new workflow)
            print(f"Warning: Fetching MC for {ticker} in process_ticker_year (should be prefetched).")
            mc_url = f"https://financialmodelingprep.com/api/v3/historical-market-capitalization/{ticker}"
            mc_params = {"from": start_date.strftime("%Y-%m-%d"), "to": end_date.strftime("%Y-%m-%d")}
            mc = get_json(mc_url, mc_params)

        # API Call 3: Price Data
        px_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
        px_params = {"from": start_date.strftime("%Y-%m-%d"), "to": end_date.strftime("%Y-%m-%d")}
        px = get_json(px_url, px_params)
        
        # API Call 4: Company Profile
        profile_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}"
        profile = get_json(profile_url)

        # API Call 5: Income Statement (for EPS and possibly shares outstanding)
        is_url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}"
        is_params = {"period": "quarter", "limit": 12} 
        income_stmt = get_json(is_url, is_params)

        # Validate data presence
        if not bs: error_log["errors"].append("No balance sheet data")
        if not mc: error_log["errors"].append("No market cap data")
        if not px: error_log["errors"].append("No price data")
        if not profile: error_log["errors"].append("No profile data")
        if not income_stmt: error_log["errors"].append("No income statement data")
            
        if not all([bs, mc, px, profile, income_stmt]):
            return None, error_log
        
        industry = profile[0].get("industry", "Unknown")
        sector = profile[0].get("sector", "Unknown")
        
        # Process Balance Sheet
        bs_df = pd.DataFrame(bs)
        bs_df['date'] = pd.to_datetime(bs_df['date'])
        bs_df = bs_df[bs_df['date'].dt.year == year].copy() # Filter for specific year
        if bs_df.empty:
            error_log["errors"].append(f"No balance sheet data for year {year}")
            return None, error_log
        
        bs_df.loc[:, 'quarter'] = bs_df['date'].dt.to_period("Q")
        bs_df.loc[:, 'debt_to_assets'] = (
            (bs_df['shortTermDebt'].fillna(0) + bs_df['longTermDebt'].fillna(0)) /
            bs_df['totalAssets'].replace(0, pd.NA)
        )
        bs_df = bs_df[['quarter', 'debt_to_assets', 'totalStockholdersEquity', 'commonStockSharesOutstanding']].drop_duplicates("quarter", keep="last")

        # Process Market Cap
        mc_df = pd.DataFrame(mc)
        mc_df['date'] = pd.to_datetime(mc_df['date'])
        mc_df = mc_df[mc_df['date'].dt.year == year].copy()
        if mc_df.empty:
             error_log["errors"].append(f"No market cap data for year {year}")
             # MC is crucial, so let's return None
             return None, error_log
        mc_df.loc[:, 'quarter'] = mc_df['date'].dt.to_period("Q")
        mc_df = mc_df.sort_values("date").drop_duplicates("quarter", keep="last")
        mc_df = mc_df.rename(columns={"marketCap": "mkt_cap"})[['quarter', 'mkt_cap']]

        # Process Price Data
        px_df = pd.DataFrame(px)
        px_df['date'] = pd.to_datetime(px_df['date'])
        # px_df = px_df[px_df['date'].dt.year == year].copy() # No, historical takes care of year
        px_df.loc[:, 'quarter'] = px_df['date'].dt.to_period("Q")
        px_df = px_df.sort_values("date").drop_duplicates("quarter", keep="last")
        px_df = px_df.rename(columns={"adjClose": "stock_price"})[['quarter', 'stock_price']]

        # Process Income Statement
        is_df = pd.DataFrame(income_stmt)
        is_df['date'] = pd.to_datetime(is_df['date'])
        is_df = is_df[is_df['date'].dt.year == year].copy()
        if is_df.empty: # If no IS data for the year, B/M and E/P will be NA. This is acceptable.
            # error_log["errors"].append(f"No income statement data for year {year}")
            # We can continue and have NA for B/M and E/P
            # Create empty df with expected columns to prevent merge errors
            is_df = pd.DataFrame(columns=['quarter', 'eps', 'weightedAverageShsOutDil'])
        else:
            is_df.loc[:, 'quarter'] = is_df['date'].dt.to_period("Q")
            # Use diluted EPS if available, otherwise basic EPS
            is_df.loc[:, 'eps'] = is_df['epsdiluted'] if 'epsdiluted' in is_df.columns else is_df.get('eps', pd.NA)
            is_df = is_df[['quarter', 'eps', 'weightedAverageShsOutDil']].drop_duplicates("quarter", keep="last")


        # Merge data
        merged = (
            mc_df.merge(px_df, on="quarter", how="left")
                 .merge(bs_df, on="quarter", how="left")
                 .merge(is_df, on="quarter", how="left")
        )

        if merged.empty:
            error_log["errors"].append("No common quarterly data after merging")
            return None, error_log

        # Calculate Book-to-Market and Earnings Yield
        # Shares outstanding: Prefer balance sheet, fallback to income statement
        merged['shares_outstanding'] = merged['commonStockSharesOutstanding'].fillna(merged['weightedAverageShsOutDil'])
        
        merged['book_value_per_share'] = merged['totalStockholdersEquity'] / merged['shares_outstanding'].replace(0, pd.NA)
        merged['book_to_market'] = merged['book_value_per_share'] / merged['stock_price'].replace(0, pd.NA)
        
        merged['earnings_yield'] = merged['eps'] / merged['stock_price'].replace(0, pd.NA)

        merged = merged.assign(ticker=ticker, industry=industry, sector=sector)
        
        # Select and order final columns
        final_cols_present = [col for col in TARGET_COLUMNS if col in merged.columns or col == "mkt_cap_rank"] # mkt_cap_rank added later
        # Ensure all TARGET_COLUMNS are present, adding missing ones with NA
        for col in TARGET_COLUMNS:
            if col not in merged.columns and col != "mkt_cap_rank":
                merged[col] = pd.NA
        
        merged = merged[final_cols_present]
        
        # Drop rows where essential financial data is missing
        merged = merged.dropna(subset=['mkt_cap', 'stock_price', 'debt_to_assets'])

        if merged.empty:
            error_log["errors"].append("No valid data after final processing and NA drop")
            return None, error_log
            
        return merged, error_log
        
    except Exception as e:
        import traceback
        error_log["errors"].append(f"Exception in process_ticker_year: {str(e)} - {traceback.format_exc()}")
        return None, error_log

def collect_year_data(tickers: List[str], year: int, max_tickers: Optional[int] = None, 
                     save_progress: bool = True, progress_interval: int = 100) -> Tuple[pd.DataFrame, List[Dict]]:
    all_data_list = []
    all_errors_list = []
    
    initial_tickers_to_process = tickers[:max_tickers] if max_tickers else tickers
    num_initial_tickers = len(initial_tickers_to_process)

    print(f"\n{'='*70}")
    print(f"  PRE-FILTERING AND COLLECTING DATA FOR YEAR {year}")
    print(f"{'='*70}")
    print(f"Starting with {num_initial_tickers} tickers for pre-filtering.")

    # --- Market Cap Pre-filter ---
    passed_filter_tickers_with_mc = []
    pre_filter_api_calls = 0
    pre_filter_start_time = time.time()
    print(f"Running market cap pre-filter (>{1e9} USD) for year {year}...")

    for i, ticker in enumerate(initial_tickers_to_process):
        call_start_time = time.time()
        
        mc_url = f"https://financialmodelingprep.com/api/v3/historical-market-capitalization/{ticker}"
        mc_params = {"from": f"{year}-01-01", "to": f"{year}-12-31"}
        hist_mc_data = get_json(mc_url, mc_params)
        pre_filter_api_calls += 1
        
        if hist_mc_data:
            mc_df_filter = pd.DataFrame(hist_mc_data)
            if not mc_df_filter.empty and 'date' in mc_df_filter.columns and 'marketCap' in mc_df_filter.columns:
                mc_df_filter['date'] = pd.to_datetime(mc_df_filter['date'])
                mc_df_filter_year = mc_df_filter[mc_df_filter['date'].dt.year == year]
                if not mc_df_filter_year.empty and mc_df_filter_year["marketCap"].mean() > 1e9:
                    passed_filter_tickers_with_mc.append((ticker, hist_mc_data)) # Store ticker and its MC data
                # else: print(f"  Ticker {ticker} filtered out (avg mkt cap <= 1B or no data for {year}).")
            # else: print(f"  Ticker {ticker} no valid MC data structure for pre-filter.")
        # else: print(f"  Ticker {ticker} no MC data for pre-filter in {year}.")

        # Enforce rate limit for the pre-filter call
        call_elapsed = time.time() - call_start_time
        if call_elapsed < SINGLE_CALL_DELAY:
            time.sleep(SINGLE_CALL_DELAY - call_elapsed)
        
        if (i + 1) % 50 == 0 or (i+1) == num_initial_tickers:
            elapsed_pre_filter = time.time() - pre_filter_start_time
            print(f"  Pre-filter progress: {i+1}/{num_initial_tickers} checked. {len(passed_filter_tickers_with_mc)} passed. "
                  f"Time: {elapsed_pre_filter/60:.1f}min.")

    pre_filter_time = time.time() - pre_filter_start_time
    print(f"Market cap pre-filter complete: {len(passed_filter_tickers_with_mc)} tickers passed out of {num_initial_tickers}.")
    print(f"Pre-filter took {pre_filter_time/60:.1f} minutes, made {pre_filter_api_calls} API calls.")

    if not passed_filter_tickers_with_mc:
        print("⚠️ No tickers passed the pre-filter. Skipping data collection for this year.")
        return pd.DataFrame(columns=TARGET_COLUMNS), all_errors_list

    # --- Main Data Collection ---
    tickers_to_process_final = [item[0] for item in passed_filter_tickers_with_mc]
    prefetched_mc_map = dict(passed_filter_tickers_with_mc)
    total_tickers_to_process = len(tickers_to_process_final)
    
    print(f"\n{'='*40}")
    print(f"  Starting Main Data Collection for {total_tickers_to_process} Filtered Tickers")
    print(f"{'='*40}")
    # API_CALLS_PER_TICKER_PROCESSING is 4 (BS, Px, Profile, IS)
    print(f"Estimated time for main collection: {(total_tickers_to_process * SECONDS_PER_TICKER_PROCESSING / 60):.1f} minutes")
    print(f"API calls for main collection: {total_tickers_to_process * API_CALLS_PER_TICKER_PROCESSING} (at {TICKERS_PER_MINUTE_PROCESSING} tickers/min)")
    print(f"Progress saves: Every {progress_interval} tickers")
    print(f"{'='*40}\n")

    collection_start_time = time.time()
    successful_tickers_count = 0
    failed_tickers_list = []
    
    for i, ticker in enumerate(tickers_to_process_final):
        ticker_processing_start_time = time.time()
        
        if i > 0 and i % 20 == 0:
            elapsed_collection = time.time() - collection_start_time
            avg_time_per_ticker = elapsed_collection / i if i > 0 else 0
            remaining_tickers = total_tickers_to_process - i
            remaining_time_est = remaining_tickers * avg_time_per_ticker
            success_rate = (successful_tickers_count / i * 100) if i > 0 else 0
            
            print(f"\n[Progress: {i}/{total_tickers_to_process} ({i/total_tickers_to_process*100:.1f}%)]")
            print(f"  Time: {elapsed_collection/60:.1f}min elapsed, ~{remaining_time_est/60:.1f}min remaining")
            print(f"  Success rate: {success_rate:.1f}% ({successful_tickers_count}/{i})")
            print(f"  Current batch: ", end="")
        
        ticker_mc_data = prefetched_mc_map.get(ticker) # Get prefetched MC data
        ticker_data_df, error_log_ticker = process_ticker_year(ticker, year, prefetched_mc_data=ticker_mc_data)
        
        if ticker_data_df is not None and not ticker_data_df.empty:
            all_data_list.append(ticker_data_df)
            successful_tickers_count += 1
            print(f"✓", end="")
        else:
            failed_tickers_list.append(ticker)
            if error_log_ticker not in all_errors_list: # Avoid duplicate error logs if retry happened within process_ticker_year
                 all_errors_list.append(error_log_ticker)
            print(f"✗", end="")
        
        if save_progress and (i + 1) % progress_interval == 0 and all_data_list:
            temp_df = pd.concat(all_data_list, ignore_index=True)
            if not temp_df.empty and 'quarter' in temp_df.columns and 'mkt_cap' in temp_df.columns:
                temp_df['mkt_cap_rank'] = temp_df.groupby('quarter')['mkt_cap'].rank(method='dense', ascending=False).fillna(0).astype(int)
            progress_filename = f"progress_{year}_tickers_{i+1}.csv"
            temp_df.to_csv(progress_filename, index=False)
            print(f"\n  💾 Progress saved: {progress_filename} ({len(temp_df)} rows)")
        
        ticker_processing_elapsed = time.time() - ticker_processing_start_time
        if ticker_processing_elapsed < SECONDS_PER_TICKER_PROCESSING:
            time.sleep(SECONDS_PER_TICKER_PROCESSING - ticker_processing_elapsed)
    
    total_collection_time = time.time() - collection_start_time
    
    print(f"\n\n{'='*70}")
    print(f"  YEAR {year} COLLECTION COMPLETE (Processed {total_tickers_to_process} filtered tickers)")
    print(f"{'='*70}")
    print(f"Total time for main collection: {total_collection_time/60:.1f} minutes ({total_collection_time/3600:.2f} hours)")
    if total_tickers_to_process > 0:
        print(f"Successful: {successful_tickers_count} tickers ({successful_tickers_count/total_tickers_to_process*100:.1f}% of filtered)")
        print(f"Failed: {len(failed_tickers_list)} tickers")
        if total_collection_time > 0:
             print(f"Actual processing rate: {total_tickers_to_process/total_collection_time*60:.1f} tickers/minute")
    else:
        print("No tickers were processed in the main collection phase.")

    if not all_data_list:
        print(f"\n⚠️ No data collected for year {year}!")
        return pd.DataFrame(columns=TARGET_COLUMNS), all_errors_list
    
    final_df = pd.concat(all_data_list, ignore_index=True)
    final_df = final_df.sort_values(['ticker', 'quarter']).reset_index(drop=True)
    
    if 'quarter' in final_df.columns and 'mkt_cap' in final_df.columns:
        final_df['mkt_cap_rank'] = final_df.groupby('quarter')['mkt_cap'].rank(method='dense', ascending=False).fillna(0).astype(int)
    else:
        final_df['mkt_cap_rank'] = pd.NA

    # Ensure all target columns are present
    for col in TARGET_COLUMNS:
        if col not in final_df.columns:
            final_df[col] = pd.NA
    final_df = final_df[TARGET_COLUMNS]


    print(f"\n📊 Final dataset for {year}: {len(final_df)} rows, {final_df['ticker'].nunique()} tickers")
    if 'quarter' in final_df.columns and not final_df['quarter'].empty:
        print(f"   Quarters: {sorted(final_df['quarter'].unique())}")
    
    if all_errors_list:
        error_filename = f"errors_{year}.json"
        with open(error_filename, 'w') as f:
            json.dump(all_errors_list, f, indent=2, default=str)
        print(f"\n📝 Error log saved: {error_filename} ({len(all_errors_list)} errors)")
    
    if save_progress:
        cleaned_count = 0
        for progress_file_name in os.listdir('.'):
            if progress_file_name.startswith(f'progress_{year}_'):
                try:
                    os.remove(progress_file_name)
                    cleaned_count +=1
                except OSError as e:
                    print(f"Error removing progress file {progress_file_name}: {e}")
        if cleaned_count > 0:
            print(f"🧹 Cleaned up {cleaned_count} progress files for year {year}")
    
    return final_df, all_errors_list

def get_us_tickers():
    print("Fetching ticker list...")
    # Cache ticker list to a file to avoid refetching too often during testing/runs
    ticker_cache_file = "us_tickers_list.json"
    us_tickers = []

    if os.path.exists(ticker_cache_file):
        try:
            with open(ticker_cache_file, 'r') as f:
                us_tickers = json.load(f)
            print(f"Loaded {len(us_tickers)} US tickers from cache.")
            return us_tickers
        except Exception as e:
            print(f"Could not load tickers from cache: {e}. Fetching live.")

    tickers_data = get_json("https://financialmodelingprep.com/api/v3/stock/list")
    
    if tickers_data:
        us_tickers = [
            d["symbol"] for d in tickers_data 
            if d.get("exchangeShortName") in ["NYSE", "NASDAQ"] 
            and (d.get("price") is not None and d.get("price", 0) > 1) # Lowered penny stock filter to $1
            and d.get("type") == "stock" # Ensure it's a stock
            and d.get("symbol") is not None 
            and len(d["symbol"]) <= 5 # Standard ticker length
            and "." not in d["symbol"] # Exclude non-common stocks like .WS, .U etc.
            and "/" not in d["symbol"] # Exclude some odd ones
        ]
        # Further filter based on delisted companies, though API should handle this.
        # Some symbols might be old or delisted. FMP data for delisted might be sparse.
        
        print(f"✅ Found {len(us_tickers)} US tickers meeting criteria.")
        print(f"   Sample: {us_tickers[:10]}")
        try:
            with open(ticker_cache_file, 'w') as f:
                json.dump(us_tickers, f)
            print(f"Saved {len(us_tickers)} tickers to cache: {ticker_cache_file}")
        except Exception as e:
            print(f"Could not save tickers to cache: {e}")
    else:
        print("❌ Failed to fetch ticker list. Using a small sample for testing.")
        us_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V"]
    return us_tickers

def run_collection_for_year(year_to_collect: int, all_us_tickers: List[str], max_tickers_override: Optional[int] = None):
    print(f"\n{'='*80}")
    print(f"  STARTING DATA COLLECTION FOR YEAR: {year_to_collect}")
    print(f"{'='*80}\n")
    
    # MAX_TICKERS = None # Set to a number (e.g., 100) for testing, None for full run
    # Override if specific value is passed
    current_max_tickers = max_tickers_override 

    data_df, errors_list = collect_year_data(
        all_us_tickers, 
        year=year_to_collect, 
        max_tickers=current_max_tickers
    )

    if not data_df.empty:
        filename = f"stock_data_{year_to_collect}.csv"
        data_df.to_csv(filename, index=False)
        print(f"\n✅ Data for {year_to_collect} saved to '{filename}' ({len(data_df)} rows)")
        
        # Show top companies (use latest quarter available for the year)
        if 'quarter' in data_df.columns and not data_df['quarter'].empty:
            latest_quarter = data_df['quarter'].max()
            latest_data = data_df[data_df['quarter'] == latest_quarter]
            if not latest_data.empty and 'mkt_cap_rank' in latest_data.columns:
                print(f"\n🏆 Top 10 companies by market cap ({latest_quarter}, Year {year_to_collect}):")
                top_10 = latest_data.nsmallest(10, 'mkt_cap_rank')[['ticker', 'mkt_cap_rank', 'mkt_cap', 'industry', 'sector']]
                print(top_10.to_string(index=False))
        
        if 'quarter' in data_df.columns and not data_df['quarter'].empty:
             print(f"\n📅 Available quarters for {year_to_collect}: {sorted(data_df['quarter'].unique())}")
    else:
        print(f"\n❌ No data collected for {year_to_collect}.")

    # Total API calls estimation per year:
    # N_initial = len(all_us_tickers) if max_tickers_override is None else min(len(all_us_tickers), max_tickers_override)
    # N_filtered = data_df['ticker'].nunique() if not data_df.empty else 0
    # total_api_calls_this_year = N_initial * 1 (pre-filter) + N_filtered * API_CALLS_PER_TICKER_PROCESSING (data)
    # print(f"Estimated API calls for {year_to_collect}: ~{total_api_calls_this_year} (exact count depends on pre-filter success)")


def combine_all_years_data(years_list: List[int]):
    print(f"\n{'='*80}")
    print(f"  COMBINING DATA FOR YEARS: {', '.join(map(str, years_list))}")
    print(f"{'='*80}\n")
    
    all_yearly_data = []
    for year_val in years_list:
        filename = f"stock_data_{year_val}.csv"
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                # Ensure quarter is period, handle potential errors if mixed format
                df['quarter'] = pd.PeriodIndex(df['quarter'], freq='Q') 
                all_yearly_data.append(df)
                print(f"✓ Loaded {filename}: {len(df)} rows")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
        else:
            print(f"✗ {filename} not found, skipping.")
    
    if all_yearly_data:
        combined_df = pd.concat(all_yearly_data, ignore_index=True)
        combined_df = combined_df.sort_values(['ticker', 'quarter']).reset_index(drop=True)
        
        # Recalculate mkt_cap_rank on the fully combined data
        if 'quarter' in combined_df.columns and 'mkt_cap' in combined_df.columns:
            combined_df['mkt_cap_rank'] = combined_df.groupby('quarter')['mkt_cap'].rank(method='dense', ascending=False).fillna(0).astype(int)

        # Ensure all target columns are present in the final combined file
        for col in TARGET_COLUMNS:
            if col not in combined_df.columns:
                combined_df[col] = pd.NA # Add missing columns with NA
        combined_df = combined_df[TARGET_COLUMNS] # Enforce order and selection

        combined_filename = f"stock_data_combined_{len(years_list)}years.csv"
        combined_df.to_csv(combined_filename, index=False)
        
        print(f"\n✅ Combined dataset saved to '{combined_filename}'")
        print(f"   Total rows: {len(combined_df):,}")
        print(f"   Unique tickers: {combined_df['ticker'].nunique()}")
        if 'quarter' in combined_df.columns and not combined_df['quarter'].empty:
             print(f"   Period: {combined_df['quarter'].min()} to {combined_df['quarter'].max()}")
    else:
        print("❌ No data files found to combine.")

def main():
    # --- Step 1: Get List of US Tickers ---
    us_tickers_list = get_us_tickers()
    if not us_tickers_list:
        print("Could not obtain ticker list. Exiting.")
        return

    # --- Step 2: Define Years to Collect ---
    # For example, collect for 2023 and 2024
    # YEARS_TO_COLLECT = [2024, 2023, 2022, 2021, 2020, 2019] 
    YEARS_TO_COLLECT = [2023] # Example: Collect for a single year first
    
    # Set to a small number for testing, e.g., 20. Set to None for all tickers passed from get_us_tickers().
    MAX_TICKERS_OVERALL = 100 # For testing purposes, limit total tickers for each year
    # MAX_TICKERS_OVERALL = None # For full run


    # --- Step 3: Collect Data Year by Year ---
    for year_item in YEARS_TO_COLLECT:
        run_collection_for_year(year_item, us_tickers_list, max_tickers_override=MAX_TICKERS_OVERALL)
        # Optional: Add a small delay between years if needed, though rate limits are per API call / per ticker batch
        # time.sleep(60) 
    
    print("\nAll specified years have been processed.")

    # --- Step 4: (Optional) Combine all collected data ---
    # You can run this separately after all individual years are collected.
    # combine_all_years_data(YEARS_TO_COLLECT)


if __name__ == "__main__":
    # Check for API key
    if API == "YOUR_API_KEY" or not API: # Update placeholder
        print("CRITICAL ERROR: API key is not set. Please edit the script to include your Financial Modeling Prep API key.")
    else:
        main() 