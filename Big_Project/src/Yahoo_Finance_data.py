import yfinance as yahooFinance

# Get the historical market data for Facebook (Meta)
GetFacebookInformation = yahooFinance.Ticker("META")
history_data = GetFacebookInformation.history(period="2y")

# Print the historical data (optional, for verification)
print(history_data)

# Save the historical data to a CSV file
history_data.to_csv("META_history_2y.csv")
