import requests
import pandas as pd
import time
import json
import os
from typing import Optional, List, Dict, Any, Set
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv(".env")
API="7cNMpVzb43GKtm05iRTDWJtyJXSylX8J"
print(API)

# Rate limiting configuration
API_CALLS_PER_MINUTE = 750
SECONDS_PER_CALL = 60 / API_CALLS_PER_MINUTE  # 0.08 seconds per call

# Session and timer for rate limiting
session = requests.Session()
LAST_API_CALL = 0.0

# Market cap threshold (1 billion)
MARKET_CAP_THRESHOLD = 1e9

print(f"Rate limit configured: {API_CALLS_PER_MINUTE} calls/minute ({SECONDS_PER_CALL:.2f} seconds/call)")
print(f"Market cap filter: > ${MARKET_CAP_THRESHOLD/1e9:.0f}B")

# ============================================================================
# CORE HELPER FUNCTIONS
# ============================================================================

def get_json(url: str, params: Dict[str, Any] = None) -> Optional[Any]:
    """Safely get JSON data from API with error handling and rate limit retry"""
    global LAST_API_CALL, session
    try:
        # Fix mutable default argument issue
        if params is None:
            params = {}
        else:
            params = params.copy()  # Make a copy to avoid modifying original
        
        params['apikey'] = API
        elapsed = time.time() - LAST_API_CALL
        if elapsed < SECONDS_PER_CALL:
            time.sleep(SECONDS_PER_CALL - elapsed)
        response = session.get(url, params=params, timeout=10)
        LAST_API_CALL = time.time()
        if response.status_code == 429:
            print('⚠️  Rate limit hit! Waiting 30 seconds...')
            time.sleep(30)
            return get_json(url, params)
        response.raise_for_status()
        js = response.json()
        if isinstance(js, dict) and 'historical' in js:
            return js['historical']
        elif isinstance(js, list):
            return js
        else:
            return js
    except requests.exceptions.HTTPError as e:
        print(f'HTTP Error {e.response.status_code}: {e}')
        return None
    except Exception as e:
        print(f'Error fetching data: {e}')
        return None

def get_historical_tickers(year: int) -> List[str]:
    """Get list of US tickers that existed in a specific year"""
    print(f"Fetching ticker list for year {year}...")
    
    # Try to get historical ticker list from end of previous year
    date = f"{year-1}-12-31"
    
    # First try to get available stocks for that date
    available_stocks = get_json(
        f"https://financialmodelingprep.com/api/v3/available-traded/list",
        {"date": date}
    )
    
    if available_stocks:
        # Filter for US exchanges
        us_tickers = [
            stock["symbol"] for stock in available_stocks 
            if stock.get("exchangeShortName") in ["NYSE", "NASDAQ", "AMEX"]
            and len(stock["symbol"]) <= 5
            and "." not in stock["symbol"]
        ]
        print(f"✅ Found {len(us_tickers)} US tickers for {year}")
        return us_tickers
    
    # Fallback: use current ticker list with a warning
    print(f"⚠️  Could not get historical ticker list for {year}, using current list")
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
        
        print(f"✅ Found {len(us_tickers)} current US tickers")
        return us_tickers
    else:
        print("❌ Failed to fetch ticker list. Using sample tickers.")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V"]

def check_market_cap_for_year(ticker: str, year: int) -> tuple[bool, Optional[float]]:
    """Check if ticker had market cap above threshold in given year"""
    try:
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        mc_data = get_json(
            f'https://financialmodelingprep.com/api/v3/historical-market-capitalization/{ticker}',
            {'from': start_date, 'to': end_date}
        )
        if not mc_data:
            return False, None
        mc_df = pd.DataFrame(mc_data)
        avg_market_cap = mc_df['marketCap'].mean()
        return avg_market_cap > MARKET_CAP_THRESHOLD, avg_market_cap
    except Exception as e:
        print(f'Error checking market cap for {ticker} in {year}: {e}')
        return False, None

# ============================================================================
# TICKER COLLECTION FUNCTIONS
# ============================================================================

def get_all_qualified_tickers(start_year: int, end_year: int) -> Set[str]:
    """Get all unique tickers that had market cap > $1B in any year within the range"""
    print(f"\n{'='*80}")
    print(f"COLLECTING ALL QUALIFIED TICKERS ({start_year}-{end_year})")
    print(f"{'='*80}")
    
    all_qualified_tickers = set()
    api_calls_used = 0
    
    for year in range(start_year, end_year + 1):
        print(f"\n--- Processing year {year} ---")
        
        # Get tickers that existed in this year
        year_tickers = get_historical_tickers(year)
        api_calls_used += 1
        
        qualified_for_year = set()
        failed_tickers = []
        
        # Check market cap for each ticker in batches
        batch_size = 50
        for i in range(0, len(year_tickers), batch_size):
            batch = year_tickers[i:i+batch_size]
            
            for ticker in batch:
                is_qualified, avg_mc = check_market_cap_for_year(ticker, year)
                api_calls_used += 1
                
                if is_qualified:
                    qualified_for_year.add(ticker)
                    all_qualified_tickers.add(ticker)
                    print("✓", end="", flush=True)
                else:
                    failed_tickers.append(ticker)
                    print("○", end="", flush=True)
            
            # Progress update
            if (i + batch_size) % 500 == 0:
                print(f"\n    Processed {i + batch_size}/{len(year_tickers)} tickers for {year}")
        
        print(f"\n✅ Year {year}: {len(qualified_for_year)} qualified tickers")
        print(f"   API calls used: {api_calls_used}")
        
        # Save progress
        with open(f'qualified_tickers_{year}.json', 'w') as f:
            json.dump(list(qualified_for_year), f)
    
    print(f"\n{'='*80}")
    print(f"TICKER COLLECTION COMPLETE")
    print(f"{'='*80}")
    print(f"Total unique qualified tickers: {len(all_qualified_tickers)}")
    print(f"Total API calls used: {api_calls_used}")
    
    # Save final list
    with open('all_qualified_tickers.json', 'w') as f:
        json.dump(list(all_qualified_tickers), f)
    
    return all_qualified_tickers

# ============================================================================
# COMPREHENSIVE DATA COLLECTION FUNCTIONS
# ============================================================================

def get_comprehensive_ticker_data(ticker: str, start_year: int, end_year: int) -> Dict[str, Any]:
    """Get all historical data for a single ticker"""
    print(f"\n--- Processing {ticker} ---")
    
    data = {
        'ticker': ticker,
        'profile': None,
        'income_statements': [],
        'balance_sheets': [],
        'cash_flows': [],
        'stock_prices': [],
        'market_caps': [],
        'ratios': [],
        'errors': []
    }
    
    api_calls = 0
    
    try:
        # Get company profile
        profile = get_json(f'https://financialmodelingprep.com/api/v3/profile/{ticker}')
        api_calls += 1
        if profile and len(profile) > 0:
            data['profile'] = profile[0]
        else:
            data['errors'].append('No profile data')
            return data, api_calls
        
        # Date range for data collection
        start_date = f'{start_year}-01-01'
        end_date = f'{end_year}-12-31'
        
        # Get income statements (quarterly)
        income_statements = get_json(
            f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}',
            {'period': 'quarter', 'limit': 100}
        )
        api_calls += 1
        if income_statements:
            data['income_statements'] = income_statements
        else:
            data['errors'].append('No income statement data')
        
        # Get balance sheets (quarterly)
        balance_sheets = get_json(
            f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}',
            {'period': 'quarter', 'limit': 100}
        )
        api_calls += 1
        if balance_sheets:
            data['balance_sheets'] = balance_sheets
        else:
            data['errors'].append('No balance sheet data')
        
        # Get cash flow statements (quarterly)
        cash_flows = get_json(
            f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}',
            {'period': 'quarter', 'limit': 100}
        )
        api_calls += 1
        if cash_flows:
            data['cash_flows'] = cash_flows
        else:
            data['errors'].append('No cash flow data')
        
        # Get historical stock prices
        stock_prices = get_json(
            f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}',
            {'from': start_date, 'to': end_date}
        )
        api_calls += 1
        if stock_prices:
            data['stock_prices'] = stock_prices
        else:
            data['errors'].append('No stock price data')
        
        # Get market capitalization
        market_caps = get_json(
            f'https://financialmodelingprep.com/api/v3/historical-market-capitalization/{ticker}',
            {'from': start_date, 'to': end_date}
        )
        api_calls += 1
        if market_caps:
            data['market_caps'] = market_caps
        else:
            data['errors'].append('No market cap data')
        
        # Get financial ratios (quarterly)
        ratios = get_json(
            f'https://financialmodelingprep.com/api/v3/ratios/{ticker}',
            {'period': 'quarter', 'limit': 100}
        )
        api_calls += 1
        if ratios:
            data['ratios'] = ratios
        else:
            data['errors'].append('No ratios data')
        
        print(f"✅ {ticker}: {api_calls} API calls")
        
    except Exception as e:
        data['errors'].append(f'Exception: {str(e)}')
        print(f"❌ {ticker}: Error - {str(e)}")
    
    return data, api_calls

# ============================================================================
# DATA PROCESSING AND ORGANIZATION
# ============================================================================

def process_and_organize_data(all_ticker_data: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """Process all ticker data and organize into separate tables"""
    print(f"\n{'='*80}")
    print("PROCESSING AND ORGANIZING DATA")
    print(f"{'='*80}")
    
    # Initialize lists for each data type
    profiles_list = []
    income_statements_list = []
    balance_sheets_list = []
    cash_flows_list = []
    stock_prices_list = []
    market_caps_list = []
    ratios_list = []
    
    for ticker_data in all_ticker_data:
        ticker = ticker_data['ticker']
        
        # Process profile data
        if ticker_data['profile']:
            profile = ticker_data['profile'].copy()
            profile['ticker'] = ticker
            profiles_list.append(profile)
        
        # Process income statements
        for statement in ticker_data['income_statements']:
            statement_copy = statement.copy()
            statement_copy['ticker'] = ticker
            income_statements_list.append(statement_copy)
        
        # Process balance sheets
        for statement in ticker_data['balance_sheets']:
            statement_copy = statement.copy()
            statement_copy['ticker'] = ticker
            balance_sheets_list.append(statement_copy)
        
        # Process cash flows
        for statement in ticker_data['cash_flows']:
            statement_copy = statement.copy()
            statement_copy['ticker'] = ticker
            cash_flows_list.append(statement_copy)
        
        # Process stock prices
        for price in ticker_data['stock_prices']:
            price_copy = price.copy()
            price_copy['ticker'] = ticker
            stock_prices_list.append(price_copy)
        
        # Process market caps
        for mc in ticker_data['market_caps']:
            mc_copy = mc.copy()
            mc_copy['ticker'] = ticker
            market_caps_list.append(mc_copy)
        
        # Process ratios
        for ratio in ticker_data['ratios']:
            ratio_copy = ratio.copy()
            ratio_copy['ticker'] = ticker
            ratios_list.append(ratio_copy)
    
    # Convert to DataFrames
    tables = {}
    
    if profiles_list:
        tables['profiles'] = pd.DataFrame(profiles_list)
        print(f"✅ Profiles table: {len(tables['profiles'])} rows")
    
    if income_statements_list:
        tables['income_statements'] = pd.DataFrame(income_statements_list)
        print(f"✅ Income statements table: {len(tables['income_statements'])} rows")
    
    if balance_sheets_list:
        tables['balance_sheets'] = pd.DataFrame(balance_sheets_list)
        print(f"✅ Balance sheets table: {len(tables['balance_sheets'])} rows")
    
    if cash_flows_list:
        tables['cash_flows'] = pd.DataFrame(cash_flows_list)
        print(f"✅ Cash flows table: {len(tables['cash_flows'])} rows")
    
    if stock_prices_list:
        tables['stock_prices'] = pd.DataFrame(stock_prices_list)
        print(f"✅ Stock prices table: {len(tables['stock_prices'])} rows")
    
    if market_caps_list:
        tables['market_caps'] = pd.DataFrame(market_caps_list)
        print(f"✅ Market caps table: {len(tables['market_caps'])} rows")
    
    if ratios_list:
        tables['ratios'] = pd.DataFrame(ratios_list)
        print(f"✅ Ratios table: {len(tables['ratios'])} rows")
    
    return tables

def save_all_tables(tables: Dict[str, pd.DataFrame], suffix: str = "") -> None:
    """Save all tables to CSV files"""
    print(f"\n{'='*80}")
    print("SAVING ALL TABLES")
    print(f"{'='*80}")
    
    for table_name, df in tables.items():
        filename = f"{table_name}{suffix}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Saved {filename}: {len(df)} rows, {df['ticker'].nunique()} unique tickers")

# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def run_comprehensive_collection(start_year: int = 2015, end_year: int = 2025, 
                                max_tickers: Optional[int] = None,
                                load_existing_tickers: bool = False) -> None:
    """Run the complete comprehensive data collection process"""
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE STOCK DATA COLLECTION ({start_year}-{end_year})")
    print(f"{'='*100}")
    
    start_time = time.time()
    total_api_calls = 0
    
    # Step 1: Get all qualified tickers
    if load_existing_tickers and os.path.exists('all_qualified_tickers.json'):
        print("Loading existing qualified tickers...")
        with open('all_qualified_tickers.json', 'r') as f:
            qualified_tickers = set(json.load(f))
        print(f"✅ Loaded {len(qualified_tickers)} qualified tickers")
    else:
        qualified_tickers = get_all_qualified_tickers(start_year, end_year)
    
    # Limit tickers if specified
    if max_tickers:
        qualified_tickers = list(qualified_tickers)[:max_tickers]
        print(f"🔢 Limited to {max_tickers} tickers for testing")
    
    # Step 2: Collect comprehensive data for each ticker
    print(f"\n{'='*80}")
    print(f"COLLECTING COMPREHENSIVE DATA FOR {len(qualified_tickers)} TICKERS")
    print(f"{'='*80}")
    
    all_ticker_data = []
    successful_tickers = []
    failed_tickers = []
    
    for i, ticker in enumerate(qualified_tickers):
        # Progress update
        if i > 0 and i % 50 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (len(qualified_tickers) - i) * avg_time
            print(f"\n[Progress: {i}/{len(qualified_tickers)} ({i/len(qualified_tickers)*100:.1f}%)]")
            print(f"  Time: {elapsed/60:.1f}min elapsed, ~{remaining/60:.1f}min remaining")
            print(f"  Success: {len(successful_tickers)}, Failed: {len(failed_tickers)}")
            print(f"  API calls: {total_api_calls}")
        
        # Get comprehensive data for ticker
        ticker_data, api_calls = get_comprehensive_ticker_data(ticker, start_year, end_year)
        total_api_calls += api_calls
        
        all_ticker_data.append(ticker_data)
        
        if ticker_data['errors']:
            failed_tickers.append(ticker)
        else:
            successful_tickers.append(ticker)
        
        # Save progress every 100 tickers
        if (i + 1) % 100 == 0:
            with open(f'progress_data_{i+1}_tickers.json', 'w') as f:
                json.dump(all_ticker_data, f, default=str)
            print(f"\n💾 Progress saved: {i+1} tickers processed")
    
    # Step 3: Process and organize data
    tables = process_and_organize_data(all_ticker_data)
    
    # Step 4: Save all tables
    save_all_tables(tables, suffix=f"_{start_year}_{end_year}")
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE COLLECTION COMPLETE")
    print(f"{'='*100}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Total tickers processed: {len(qualified_tickers)}")
    print(f"Successful: {len(successful_tickers)}")
    print(f"Failed: {len(failed_tickers)}")
    print(f"Total API calls: {total_api_calls:,} ({total_api_calls/total_time*60:.0f}/minute avg)")
    print(f"Tables created: {len(tables)}")
    
    # Save error summary
    errors_summary = []
    for ticker_data in all_ticker_data:
        if ticker_data['errors']:
            errors_summary.append({
                'ticker': ticker_data['ticker'],
                'errors': ticker_data['errors']
            })
    
    if errors_summary:
        with open(f'errors_summary_{start_year}_{end_year}.json', 'w') as f:
            json.dump(errors_summary, f, indent=2)
        print(f"📝 Error summary saved: {len(errors_summary)} tickers with errors")
    
    # Clean up progress files
    for f in os.listdir('.'):
        if f.startswith('progress_data_') and f.endswith('.json'):
            os.remove(f)
    print("🧹 Cleaned up progress files")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Test with limited tickers first
    print("🧪 Testing with limited tickers first...")
    run_comprehensive_collection(
        start_year=2015, 
        end_year=2025, 
        max_tickers=10,  # Test with 10 tickers
        load_existing_tickers=False
    )
    
    # Uncomment below to run full collection
    # print("\n🚀 Starting full collection...")
    # run_comprehensive_collection(
    #     start_year=2015, 
    #     end_year=2025, 
    #     max_tickers=None,  # All tickers
    #     load_existing_tickers=True  # Load existing if available
    # ) 