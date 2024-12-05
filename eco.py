import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the dataset
df = pd.read_excel('C:/Users/Labab/Desktop/project eco/Wheat Data-All Years.xlsx', sheet_name=1, header=6)

# Check the current column names and count
print("Original Columns:", df.columns)
print("Number of Columns:", len(df.columns))

# Assign proper column names based on the actual structure
df.columns = ['Category', 'Year', 'Placeholder', 'Production', 'Supply', 'Yield', 'Price']  # Match this to the actual number of columns

# Drop unnecessary columns
df_cleaned = df.drop(columns=['Category', 'Placeholder'])

# Convert 'Year' to proper format by extracting the first part of the year string (e.g., '1871/72' -> '1871')
df_cleaned['Year'] = df_cleaned['Year'].astype(str).apply(lambda x: x.split('/')[0])

# Convert 'Year' to datetime format
df_cleaned['Year'] = pd.to_datetime(df_cleaned['Year'], format='%Y')

# Ensure all columns that should be numeric are converted
numeric_columns = ['Production', 'Supply', 'Price', 'Yield']
for col in numeric_columns:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

# Drop rows with any NaN values after conversion to numeric types
df_cleaned.dropna(subset=numeric_columns, inplace=True)

# Set 'Year' as the index
df_cleaned.set_index('Year', inplace=True)

# Display the cleaned dataset
print(df_cleaned.head(10))

# Visualization 1: Price over Time
import matplotlib.pyplot as plt
df_cleaned = df_cleaned[~df_cleaned.index.duplicated(keep='first')]
df_cleaned_aggregated = df_cleaned.groupby(df_cleaned.index).agg({'Price': 'mean'}).reset_index()

# Use the aggregated or cleaned data
plt.figure(figsize=(10, 6))
plt.plot(df_cleaned_aggregated['Year'], df_cleaned_aggregated['Price'], label='Price', color='blue')
plt.title('Wheat Price Over Time')
plt.xlabel('Year')
plt.ylabel('Price per Bushel')
plt.grid(True)
plt.legend()
plt.show()




# Calculate rolling standard deviation for volatility
df_cleaned = df_cleaned[~df_cleaned.index.duplicated(keep='first')]

# Calculate rolling standard deviation for volatility
rolling_window = 12  # 12 periods for monthly data (adjust if data is yearly)
df_cleaned['Rolling Std Dev'] = df_cleaned['Price'].rolling(window=rolling_window).std()

# Plot the price and its volatility
plt.figure(figsize=(10, 6))

# Price Plot
plt.plot(df_cleaned.index, df_cleaned['Price'], label='Price', color='blue', alpha=0.7)

# Rolling Standard Deviation (Volatility)
plt.plot(df_cleaned.index, df_cleaned['Rolling Std Dev'], label='Rolling Std Dev (Volatility)', color='red', linestyle='--')

# Add titles and labels
plt.title('Wheat Price Volatility (Rolling Standard Deviation)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price / Volatility', fontsize=12)
plt.legend()
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Aggregate data to avoid overlaps
df_cleaned_aggregated = df_cleaned.groupby('Year').agg({'Production': 'sum', 'Supply': 'sum'}).reset_index()

# Create subplots: one for Production and one for Supply
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot Production on the first subplot
ax1.plot(df_cleaned_aggregated['Year'], df_cleaned_aggregated['Production'], label='Production', color='green', linewidth=2)
ax1.set_title('Wheat Production Over Time')
ax1.set_ylabel('Million Tons')
ax1.grid(True)
ax1.legend()

# Plot Supply on the second subplot
ax2.plot(df_cleaned_aggregated['Year'], df_cleaned_aggregated['Supply'], label='Supply', color='red', linewidth=2)
ax2.set_title('Wheat Supply Over Time')
ax2.set_xlabel('Year')
ax2.set_ylabel('Million Tons')
ax2.grid(True)
ax2.legend()

# Adjust layout
plt.tight_layout()
plt.show()


# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# Visualization 3: Price vs Yield
plt.figure(figsize=(10, 6))
plt.plot(df_cleaned.index, df_cleaned['Price'], label='Price', color='blue')
plt.plot(df_cleaned.index, df_cleaned['Yield'], label='Yield', color='orange')
plt.title('Wheat Price vs Yield Over Time')
plt.xlabel('Year')
plt.ylabel('Price / Yield')
plt.grid(True)
plt.legend()
plt.show()

# Ensure the required columns are available
if 'Production' in df_cleaned.columns and 'Yield' in df_cleaned.columns:
    # Calculate Harvested Acreage (in Acres)
    df_cleaned['Harvested Acreage'] = (df_cleaned['Production'] / df_cleaned['Yield']) * 2.47105  # Convert from metric tons/hectare to acres

    # Display the updated dataset with Harvested Acreage
    print(df_cleaned.head(10))

    # Plot Harvested Acreage Over Time
    plt.figure(figsize=(10, 6))
    plt.plot(df_cleaned.index, df_cleaned['Harvested Acreage'], color='orange', marker='o', linestyle='-', linewidth=2, markersize=5)
    plt.title('Harvested Acreage Over Time', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Harvested Acreage (Million Acres)', fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Error: Required columns 'Production' and 'Yield' are not available to calculate Harvested Acreage.")


import numpy as np
import pandas as pd

# Assuming df contains the marketing year data (replace with actual df if necessary)
# Example years data (replace with actual data)
years = df['Year']
price=np.random.uniform(100, 300, size=len(years))

# Simulating economic indicators data
exchange_rate = price * 0.001 + np.random.uniform(1, 1.9, size=len(years))  # Slight positive correlation with price
inflation_rate = price * 0.03 + np.random.uniform(1.5, 15, size=len(years))  # Slight positive correlation with price
interest_rate = price * 0.6 + np.random.uniform(40, 100, size=len(years))  # Negative correlation with price
gdp_growth = price * 0.008 + np.random.uniform(1.0, 3.5, size=len(years))  # Slight positive correlation with price

# Adding these simulated data to the original DataFrame (df should already have the actual data)
df['Exchange Rate (USD to EUR)'] = exchange_rate
df['Inflation Rate (CPI)'] = inflation_rate
df['Interest Rate'] = interest_rate
df['GDP Growth (%)'] = gdp_growth
df['Price']= price


numeric_df = df.select_dtypes(include=['number'])

# Perform correlation analysis (Pearson correlation)
correlation_matrix = numeric_df.corr()

# Display the correlation matrix
print(correlation_matrix)


# Plotting the correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Wheat Price with Economic Indicators', fontsize=16)
plt.tight_layout()
plt.show()

# Visualization 4: Correlation Matrix (Production, Supply, Price)
correlation_matrix = df_cleaned[['Production', 'Supply', 'Price']].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
)
plt.title('Correlation Matrix: Production, Supply, Price')
plt.show()

# Regression Analysis: Impact of Production and Supply on Price
X = df_cleaned[['Production', 'Supply']]  # Independent variables
y = df_cleaned['Price']  # Dependent variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict using the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

import pandas as pd
import statsmodels.api as sm

# Select the relevant independent variables for GDP growth, inflation, and exchange rates
X = df[['GDP Growth (%)', 'Inflation Rate (CPI)', 'Exchange Rate (USD to EUR)', 'Interest Rate']]

# Dependent variable: Wheat price
y = df['Price']

# Convert to numeric and handle non-numeric data
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Drop rows with missing values
data = pd.concat([X, y], axis=1).dropna()
X = data[X.columns]
y = data[y.name]

# Add a constant to the model (this is the intercept in the regression equation)
X = sm.add_constant(X)

# Fit the OLS regression model
model = sm.OLS(y, X).fit()

# Show the regression results
print(model.summary())

# Access regression coefficients
print("\nRegression Coefficients:")
print(model.params)


# Extract regression coefficients from the model
coefficients = model.params[1:]  # Exclude the constant term (intercept)
variables = coefficients.index  # Variable names
values = coefficients.values  # Coefficient values

# Create a bar chart
plt.figure(figsize=(8, 6))
plt.bar(variables, values, color='skyblue')
plt.title('Impact of GDP, Inflation, and Interest Rates on Wheat Prices', fontsize=14)
plt.xlabel('Economic Indicators', fontsize=12)
plt.ylabel('Regression Coefficients', fontsize=12)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add a horizontal line at y=0
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Scatter plot showing the relationship between GDP Growth and Wheat Prices
plt.figure(figsize=(8, 6))
plt.scatter(data['GDP Growth (%)'], data['Price'], color='blue', alpha=0.6)
plt.title('Relationship Between GDP Growth and Wheat Prices', fontsize=14)
plt.xlabel('GDP Growth (%)', fontsize=12)
plt.ylabel('Wheat Prices (USD)', fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(data['Inflation Rate (CPI)'], data['Price'], color='blue', alpha=0.6)
plt.title('Relationship Inflation Rate and Wheat Prices', fontsize=14)
plt.xlabel('Inflation Rate (CPI)', fontsize=12)
plt.ylabel('Wheat Prices (USD)', fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(data['Exchange Rate (USD to EUR)'], data['Price'], color='blue', alpha=0.6)
plt.title('Relationship Between Exchange rate and Wheat Prices', fontsize=14)
plt.xlabel('Exchange Rate (USD to EUR)', fontsize=12)
plt.ylabel('Wheat Prices (USD)', fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(data['Interest Rate'], data['Price'], color='blue', alpha=0.6)
plt.title('Relationship Between Interest Rate and Wheat Prices', fontsize=14)
plt.xlabel('Interest Rate', fontsize=12)
plt.ylabel('Wheat Prices (USD)', fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()


from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import matplotlib.pyplot as plt

# Ensure 'Price' is the time series we're working with
ts = df_cleaned['Price']

# Fit SARIMAX model
# SARIMAX(p, d, q) x (P, D, Q, s), where 's' is the seasonal period
# Example: Seasonal period is 12 for monthly data
model = SARIMAX(ts, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()

# Forecast the next 6 months (assuming monthly data; adjust steps accordingly for yearly data)
forecast_steps = 36
forecast = model_fit.forecast(steps=forecast_steps)

# Generate the future date range (next 6 months)
forecast_dates = pd.date_range(ts.index[-1], periods=forecast_steps + 1, freq='M')[1:]

# Plot the historical data and the forecast
plt.figure(figsize=(10, 6))
plt.plot(ts.index, ts, label='Historical Price', color='blue')
plt.plot(forecast_dates, forecast, label='Forecasted Price', color='red', linestyle='--')
plt.title('Wheat Price Forecast (SARIMAX) for the Next 6 Months')
plt.xlabel('Date')
plt.ylabel('Price per Bushel')

# Customize x-ticks (5-year intervals for better readability)
x_ticks = pd.date_range(start=ts.index.min(), end=forecast_dates[-1], freq='5YS')
plt.xticks(ticks=x_ticks, labels=x_ticks.strftime('%Y'), rotation=45)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print model summary for additional details
print(model_fit.summary())



# Save the cleaned dataframe as a new CSV file (optional step)
df_cleaned.to_csv('cleaned_wheat_data.csv')
