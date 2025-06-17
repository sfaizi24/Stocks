import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the fixed function from the notebook
import requests
import pandas as pd
import time
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv

# Load API key - try multiple approaches
load_dotenv('../.env')
load_dotenv('.env')
load_dotenv()  # Try current directory
API = os.getenv('API')

if not API:
    print("❌ No API key found")
    exit(1)

# Configuration
API_CALLS_PER_MINUTE = 750
SECONDS_PER_CALL = 60 / API_CALLS_PER_MINUTE
session = requests.Session()
LAST_API_CALL = 0.0
MARKET_CAP_THRESHOLD = 1e9

def get_json(url: str, params: Dict[str, Any] = {}):
    global LAST_API_CALL, session
    try:
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
    except Exception as e:
        print(f'Error fetching data: {e}')
        return None

def check_market_cap(ticker: str, year: int, precomputed: Optional[float] = None) -> Tuple[bool, Optional[float]]:
    """Check if ticker had market cap above threshold in given year"""
    if precomputed is not None:
        return precomputed > MARKET_CAP_THRESHOLD, precomputed
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
        print(f'Error checking market cap for {ticker}: {e}')
        return False, None

def get_bulk_profiles(tickers: List[str]) -> Dict[str, Any]:
    """Fetch company profiles in bulk."""
    data = get_json(f'https://financialmodelingprep.com/api/v3/profile/{",".join(tickers)}')
    profiles = {}
    if isinstance(data, list):
        for item in data:
            symbol = item.get('symbol')
            profiles[symbol] = item
    return profiles

# FIXED FUNCTION with increased limit
def process_ticker_year_separated_fixed(ticker: str, year: int, profile_data: Optional[Dict[str, Any]] = None, 
                                       avg_market_cap: Optional[float] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any], int]:
    """Process data for a single ticker for a specific year - returns separated price and statement data"""
    error_log = {'ticker': ticker, 'year': year, 'errors': []}
    api_calls = 0
    
    try:
        # Check market cap
        is_large_cap, avg_market_cap = check_market_cap(ticker, year, precomputed=avg_market_cap)
        if avg_market_cap is None:
            api_calls += 1
        
        if not is_large_cap:
            error_log['errors'].append(f'Market cap below threshold (avg: ${avg_market_cap:,.0f})')
            return None, None, error_log, api_calls
        
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        
        # Get all the data - use broader date range for financial statements
        # Extend date range for financial statements to capture fiscal years
        fs_start_date = datetime(year - 1, 1, 1)
        fs_end_date = datetime(year + 1, 12, 31)
        
        # INCREASED LIMIT TO GET MORE HISTORICAL DATA
        bs = get_json(f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}', 
                     {'period': 'quarter', 'limit': 80})
        api_calls += 1
        
        inc = get_json(f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}', 
                      {'period': 'quarter', 'limit': 80})
        api_calls += 1
        
        # Get market cap data for broader range to cover all fiscal quarters
        mc = get_json(f'https://financialmodelingprep.com/api/v3/historical-market-capitalization/{ticker}', 
                     {'from': fs_start_date.strftime('%Y-%m-%d'), 'to': fs_end_date.strftime('%Y-%m-%d')})
        api_calls += 1
        
        # Get price data for broader range to cover all fiscal quarters
        px = get_json(f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}', 
                     {'from': fs_start_date.strftime('%Y-%m-%d'), 'to': fs_end_date.strftime('%Y-%m-%d')})
        api_calls += 1
        
        if profile_data is None:
            profile = get_json(f'https://financialmodelingprep.com/api/v3/profile/{ticker}')
            api_calls += 1
        else:
            profile = [profile_data] if isinstance(profile_data, dict) else profile_data
        
        if not all([bs, inc, mc, px, profile]):
            if not bs: error_log['errors'].append('No balance sheet data')
            if not inc: error_log['errors'].append('No income statement data')
            if not mc: error_log['errors'].append('No market cap data')
            if not px: error_log['errors'].append('No price data')
            if not profile: error_log['errors'].append('No profile data')
            return None, None, error_log, api_calls
        
        # Extract profile info
        profile_info = profile[0] if profile and len(profile) > 0 else {}
        company_name = profile_info.get('companyName', '')
        industry = profile_info.get('industry', 'Unknown')
        sector = profile_info.get('sector', 'Unknown')
        is_etf = profile_info.get('isEtf', False)
        is_fund = profile_info.get('isFund', False)
        
        # Process all data
        bs_df = pd.DataFrame(bs)
        bs_df['date'] = pd.to_datetime(bs_df['date'])
        
        inc_df = pd.DataFrame(inc)
        inc_df['date'] = pd.to_datetime(inc_df['date'])
        
        mc_df = pd.DataFrame(mc)
        mc_df['date'] = pd.to_datetime(mc_df['date'])
        
        px_df = pd.DataFrame(px)
        px_df['date'] = pd.to_datetime(px_df['date'])
        
        # Create calendar quarter end dates
        quarter_dates = [
            f"{year}-03-31",
            f"{year}-06-30",
            f"{year}-09-30",
            f"{year}-12-31"
        ]
        
        # 1. Create Stock Price Data (aligned to calendar quarters)
        price_data_list = []
        for quarter_date in quarter_dates:
            # Find closest price to quarter end
            quarter_dt = pd.to_datetime(quarter_date)
            px_quarter = px_df[abs(px_df['date'] - quarter_dt) <= pd.Timedelta(days=7)]
            
            if len(px_quarter) > 0:
                # Get closest date
                closest_idx = abs(px_quarter['date'] - quarter_dt).idxmin()
                price_row = px_quarter.loc[closest_idx]
                
                # Get market cap for this date
                mc_quarter = mc_df[abs(mc_df['date'] - quarter_dt) <= pd.Timedelta(days=7)]
                if len(mc_quarter) > 0:
                    closest_mc_idx = abs(mc_quarter['date'] - quarter_dt).idxmin()
                    market_cap = mc_quarter.loc[closest_mc_idx, 'marketCap']
                else:
                    market_cap = None
                
                if market_cap and market_cap >= MARKET_CAP_THRESHOLD:
                    price_data_list.append({
                        'ticker': ticker,
                        'company_name': company_name,
                        'quarter_end_date': quarter_date,
                        'stock_price': price_row['adjClose'],
                        'market_cap': market_cap,
                        'industry': industry,
                        'sector': sector,
                        'isETF': is_etf,
                        'isFund': is_fund
                    })
        
        # 2. Create Financial Statement Data (find best 4 quarters for the calendar year)
        statement_data_list = []
        
        # Merge balance sheet and income statement by date
        bs_quarters = bs_df[['date', 'shortTermDebt', 'longTermDebt', 'totalAssets', 
                             'totalStockholdersEquity', 'commonStock']].copy()
        inc_quarters = inc_df[['date', 'eps', 'weightedAverageShsOut', 'period', 
                              'calendarYear', 'netIncome']].copy()
        
        # Join on date
        merged_statements = pd.merge(bs_quarters, inc_quarters, on='date', how='inner')
        
        if len(merged_statements) == 0:
            # If no merged statements, just return price data
            price_df = pd.DataFrame(price_data_list) if price_data_list else None
            return price_df, None, error_log, api_calls
        
        # Sort by date
        merged_statements = merged_statements.sort_values('date')
        
        # Find the 4 quarters that best overlap with the calendar year
        # Score each quarter based on how well it represents the calendar year
        scored_quarters = []
        for _, row in merged_statements.iterrows():
            quarter_date = row['date']
            
            # Calculate relevance score for this quarter to the calendar year
            # Quarters ending in the calendar year get highest score
            if quarter_date.year == year:
                if quarter_date.month in [3, 6, 9, 12]:  # Standard quarters
                    score = 10 + (12 - abs(quarter_date.month - 6))  # Prefer middle of year
                else:
                    score = 8
            # Quarters ending in Q1 of following year (for companies with Dec fiscal year)
            elif quarter_date.year == year + 1 and quarter_date.month <= 3:
                score = 7
            # Quarters ending in Q4 of previous year (for companies with early fiscal year)
            elif quarter_date.year == year - 1 and quarter_date.month >= 10:
                score = 6
            else:
                score = 0
            
            if score > 0:
                scored_quarters.append((score, row))
        
        # Sort by score (descending) and take top 4
        scored_quarters.sort(key=lambda x: x[0], reverse=True)
        top_quarters = [quarter for score, quarter in scored_quarters[:4]]
        
        for row in top_quarters:
            fiscal_date = row['date'].strftime('%Y-%m-%d')
            
            # Get market cap for this fiscal date using the broader dataset we already have
            mc_fiscal = mc_df[abs(mc_df['date'] - row['date']) <= pd.Timedelta(days=10)]
            if len(mc_fiscal) > 0:
                closest_mc_idx = abs(mc_fiscal['date'] - row['date']).idxmin()
                market_cap = mc_fiscal.loc[closest_mc_idx, 'marketCap']
            else:
                # Skip this quarter if no market cap data
                continue
            
            if market_cap < MARKET_CAP_THRESHOLD:
                continue
            
            # Get stock price for this date using the broader dataset we already have  
            px_fiscal = px_df[abs(px_df['date'] - row['date']) <= pd.Timedelta(days=10)]
            if len(px_fiscal) > 0:
                closest_px_idx = abs(px_fiscal['date'] - row['date']).idxmin()
                stock_price = px_fiscal.loc[closest_px_idx, 'adjClose']
            else:
                # Skip this quarter if no price data
                continue
            
            # Calculate ratios
            total_debt = (row['shortTermDebt'] or 0) + (row['longTermDebt'] or 0)
            debt_to_assets = total_debt / row['totalAssets'] if row['totalAssets'] > 0 else None
            
            if stock_price and row['weightedAverageShsOut'] > 0:
                book_to_market = (row['totalStockholdersEquity'] / row['weightedAverageShsOut']) / stock_price
                earnings_yield = row['eps'] / stock_price if row['eps'] is not None else None
            else:
                book_to_market = None
                earnings_yield = None
            
            statement_data_list.append({
                'ticker': ticker,
                'company_name': company_name,
                'fiscal_quarter': row['period'],
                'fiscal_year': row['calendarYear'],
                'calendar_date': fiscal_date,
                'debt_to_assets': debt_to_assets,
                'book_to_market': book_to_market,
                'earnings_yield': earnings_yield,
                'industry': industry,
                'sector': sector
            })
        
        # Convert to DataFrames
        price_df = pd.DataFrame(price_data_list) if price_data_list else None
        statement_df = pd.DataFrame(statement_data_list) if statement_data_list else None
        
        if price_df is None and statement_df is None:
            error_log['errors'].append('No valid data after processing')
            return None, None, error_log, api_calls
        
        return price_df, statement_df, error_log, api_calls
        
    except Exception as e:
        error_log['errors'].append(f'Exception: {str(e)}')
        return None, None, error_log, api_calls

# Test the fixed function
print("Testing FIXED function with AAPL 2018...")
profile_data = get_bulk_profiles(["AAPL"]).get("AAPL")
price_data, statement_data, error_log, api_calls = process_ticker_year_separated_fixed("AAPL", 2018, profile_data)

print(f"\nTest completed with {api_calls} API calls")

if price_data is not None:
    print(f"\n✅ Price data collected: {len(price_data)} records")
    print(price_data)
else:
    print("\n❌ No price data collected")

if statement_data is not None:
    print(f"\n✅ Statement data collected: {len(statement_data)} records")
    print(statement_data)
else:
    print("\n❌ No statement data collected")

if error_log['errors']:
    print(f"\nErrors: {error_log}")

print("\n" + "="*80)
print("Testing FIXED function with AAPL 2012 (earliest available)...")
print("="*80)

price_data_2012, statement_data_2012, error_log_2012, api_calls_2012 = process_ticker_year_separated_fixed("AAPL", 2012, profile_data)

print(f"\nTest completed with {api_calls_2012} API calls")

if price_data_2012 is not None:
    print(f"\n✅ Price data collected: {len(price_data_2012)} records")
    print(price_data_2012.head())
else:
    print("\n❌ No price data collected")

if statement_data_2012 is not None:
    print(f"\n✅ Statement data collected: {len(statement_data_2012)} records")
    print(statement_data_2012.head())
else:
    print("\n❌ No statement data collected")

if error_log_2012['errors']:
    print(f"\nErrors: {error_log_2012}") 