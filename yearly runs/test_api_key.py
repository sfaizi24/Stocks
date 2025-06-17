import requests

# Test API key
API_KEY = "7cNMpVzb43GKtm05iRTDWJtyJXSylX8J"

def test_api_key():
    """Test if the API key works with a simple call"""
    print(f"Testing API key: {API_KEY}")
    
    # Simple test call to get AAPL profile
    url = "https://financialmodelingprep.com/api/v3/profile/AAPL"
    params = {"apikey": API_KEY}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS! Got data: {len(data)} items")
            if data and len(data) > 0:
                print(f"Company: {data[0].get('companyName', 'N/A')}")
                print(f"Symbol: {data[0].get('symbol', 'N/A')}")
        else:
            print(f"❌ FAILED! Status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_api_key() 