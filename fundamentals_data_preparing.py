# import WakaTime
# email
# regulr password

import pandas as pd
import requests

from data_preparing import load_top_40_tech_companies_names
from twelve_data_api import Twelve_data_API_key
from runtime_config import get_where_the_code_runs


"""
Here will we create a new file with fundamental data of the stocks.
We do not save with the OCHLV data, because this data changes slowly, and there is no need 
to update it all the time.
Not like the OCHLV data.
Will take data once a month.
add newest data before the date of the row
"""



"""
these are the fields we can get from this API url:
{
    "symbol": "AAPL",
    "name": "Apple Inc",
    "exchange": "NASDAQ",
    "mic_code": "XNAS",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "employees": 147000,
    "website": "http://www.apple.com",
    "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and...",
    "type": "Common Stock",
    "CEO": "Mr. Timothy D. Cook", 
    "address": "One Apple Park Way",
    "address2": "Cupertino, CA 95014",
    "city": "Cupertino",
    "zip": "95014",
    "state": "CA",
    "country": "US",
    "phone": "408-996-1010"
}
"""

# this URl is not free: https://api.twelvedata.com/profile
# we cant use it
def select_relevant_fields(symbols, relevant_fields):
    all_companies = []  # Use a list to collect data rows

    for sym in symbols:
        url = f"https://api.twelvedata.com/profile?symbol={sym}&apikey={Twelve_data_API_key}"

        response = requests.get(url)
        data = response.json()

        # Check if the API returned an error (like 'api key invalid' or 'not found')
        if "status" in data and data["status"] == "error":
            print(f"Error fetching {sym}: {data.get('message')}")
            continue

        # Extract only the fields you asked for
        selected_data = {}
        for field in relevant_fields:
            # data.get(field) returns None if the field doesn't exist
            selected_data[field] = data.get(field)

        all_companies.append(selected_data)

    # Create the DataFrame once from the list of dictionaries
    df = pd.DataFrame(all_companies)
    return df



def main() -> None:
    where_the_code_runs = get_where_the_code_runs()

    # Example usage:
    my_fields = ['symbol', 'sector', 'industry']
    # symbols_list = ['AAPL', 'MSFT', 'GOOGL']

    top_40_tech_names = load_top_40_tech_companies_names(where_the_code_runs)
    df_results = select_relevant_fields(top_40_tech_names, my_fields)
    print(df_results)


if __name__ == '__main__':
    main()



