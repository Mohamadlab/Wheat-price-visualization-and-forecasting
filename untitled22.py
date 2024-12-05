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
plt.figure(figsize=(10, 6))
plt.plot(df_cleaned.index, df_cleaned['Price'], label='Price', color='blue')
plt.title('Wheat Price Over Time')
plt.xlabel('Year')
plt.ylabel('Price per Bushel')
plt.grid(True)
plt.legend()
plt.show()



# Calculate rolling standard deviation for volatility
rolling_window = 12  # 12 periods for monthly data
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



# Visualization 2: Production vs Supply
plt.figure(figsize=(10, 6))
plt.plot(df_cleaned.index, df_cleaned['Production'], label='Production', color='green')
plt.plot(df_cleaned.index, df_cleaned['Supply'], label='Supply', color='red')
plt.title('Wheat Production and Supply Over Time')
plt.xlabel('Year')
plt.ylabel('Million Tons')
plt.grid(True)
plt.legend()
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

np.random.seed(42)

# Assuming df contains the marketing year data
years = df['Year']

# Simulating economic indicators data
exchange_rate = np.random.uniform(0.8, 1.2, size=len(years))  # USD/EUR exchange rate
inflation_rate = np.random.uniform(1.5, 3.5, size=len(years))  # CPI inflation (1.5% to 3.5%)
interest_rate = np.random.uniform(0.5, 5.0, size=len(years))  # Interest rates (0.5% to 5.0%)
oil_prices = np.random.uniform(40, 100, size=len(years))  # Oil prices (40 to 100 USD)
global_production = np.random.uniform(2000, 3000, size=len(years))  # Global wheat production in million bushels
gdp_growth = np.random.uniform(1.0, 3.5, size=len(years))  # GDP growth (1% to 3.5%)

# Adding these simulated data to the original DataFrame
df['Exchange Rate (USD to EUR)'] = exchange_rate
df['Inflation Rate (CPI)'] = inflation_rate
df['Interest Rate (%)'] = interest_rate
df['Crude Oil Price (USD)'] = oil_prices
df['Global Wheat Production (Million Bushels)'] = global_production
df['GDP Growth (%)'] = gdp_growth

# Keep only numeric columns for correlation analysis
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
X = df[['GDP Growth (%)', 'Inflation Rate (CPI)', 'Exchange Rate (USD to EUR)']]

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



# Display results
print(f'Linear Regression Model Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')
print(f'Mean Squared Error (MSE): {mse}')

# Visualization of Regression Model (Actual vs Predicted Prices)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title('Regression: Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Prepare the dataset
# Assume 'df' is your DataFrame, and 'Price' is the column with wheat prices
df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date is in datetime format
df.set_index('Date', inplace=True)
price_series = df['Price']

# Split the data into training and testing sets
train_size = int(len(price_series) * 0.8)
train, test = price_series[:train_size], price_series[train_size:]

# Fit an ARIMA model (you can adjust the order as needed)
model = ARIMA(train, order=(1, 1, 1))  # Adjust p, d, q based on data
model_fit = model.fit()

# Forecast on the test set
forecast = model_fit.forecast(steps=len(test))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train, label='Training Data', color='blue')
plt.plot(test, label='Actual Prices', color='green')
plt.plot(test.index, forecast, label='Forecasted Prices', color='orange', linestyle='--')
plt.title('ARIMA Model: Wheat Price Forecast')
plt.xlabel('Date')
plt.ylabel('Wheat Price (USD)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()








# Save the cleaned dataframe as a new CSV file (optional step)
df_cleaned.to_csv('cleaned_wheat_data.csv')
