import pandas as pd
import sys

def query_stocks(query):
    # Read the CSV file
    df = pd.read_csv('yearly runs/stock_data_2024.csv')
    
    # Example queries:
    if query == "top_tech":
        # Top 5 technology companies by market cap
        result = df[df['sector'] == 'Technology'].nlargest(5, 'mkt_cap')[['ticker', 'industry', 'mkt_cap', 'stock_price', 'earnings_yield']]
    
    elif query == "high_yield":
        # Top 10 companies by earnings yield
        result = df.nlargest(10, 'earnings_yield')[['ticker', 'sector', 'industry', 'earnings_yield', 'stock_price']]
    
    elif query == "value_stocks":
        # Value stocks (high book-to-market ratio)
        result = df[df['mkt_cap'] > 1e9].nlargest(10, 'book_to_market')[['ticker', 'sector', 'book_to_market', 'earnings_yield', 'stock_price']]
    
    elif query == "sectors":
        # Sector summary
        result = df.groupby('sector').agg({
            'mkt_cap': 'sum',
            'earnings_yield': 'mean',
            'book_to_market': 'mean'
        }).sort_values('mkt_cap', ascending=False)
        
    else:
        print("Available queries: top_tech, high_yield, value_stocks, sectors")
        return

    print("\nQuery Results:")
    print("=" * 80)
    print(result.to_string())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python query_stocks.py <query_name>")
        print("Available queries: top_tech, high_yield, value_stocks, sectors")
        sys.exit(1)
    
    query_stocks(sys.argv[1]) 