import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
file_path = 'Cleaned_Wheat_Dataset2.xlsx'
xls = pd.ExcelFile('Cleaned_Wheat_Dataset2.xlsx')

# Load all sheets into a dictionary of DataFrames
sheets_data = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}

# Extract production data for the most recent year (2022/23) from Table25
production_data = sheets_data.get('Table25')  # Likely global wheat production data
production_latest = production_data[['Geography 1/', '2022/23']].dropna()
production_latest.columns = ['Country', 'Production']
production_latest['Production'] = pd.to_numeric(production_latest['Production'], errors='coerce')
production_latest = production_latest.dropna()
production_latest = production_latest.sort_values(by='Production', ascending=False).head(10)

# Extract export data for the most recent year (2013/14) from Table26 (U.S. wheat exports)
consumption_data = sheets_data.get('Table26')  # Likely U.S. consumption or export data
consumption_latest = consumption_data[['Geography 1/', '2013/14']].dropna()
consumption_latest.columns = ['Country', 'Exports']
consumption_latest['Exports'] = pd.to_numeric(consumption_latest['Exports'], errors='coerce')
consumption_latest = consumption_latest.dropna()
consumption_latest = consumption_latest.sort_values(by='Exports', ascending=False).head(10)

# Plotting the bar chart for top producers and top consumers
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the Top Producers
ax.barh(production_latest['Country'], production_latest['Production'], color='skyblue', label='Top Producers')

# Plot the Top Consumers
ax.barh(consumption_latest['Country'], consumption_latest['Exports'], color='salmon', label='Top Consumers')

# Add labels and title
ax.set_xlabel('Amount (metric tons)')
ax.set_title('Top Wheat Producers and Consumers (2022/23 Production, 2013/14 Exports)')
ax.legend()

# Display the chart
plt.tight_layout()
plt.show()

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.barh(production_latest['Country'], production_latest['Production'], color='skyblue')
ax1.set_xlabel('Production (metric tons)')
ax1.set_title('Top Wheat Producers (2022/23)')
plt.tight_layout()
plt.show()

# Bar chart for Top Consumers
fig, ax2 = plt.subplots(figsize=(10, 6))
ax2.barh(consumption_latest['Country'], consumption_latest['Exports'], color='salmon')
ax2.set_xlabel('Exports (metric tons)')
ax2.set_title('Top Wheat Consumers (2013/14 Exports)')
plt.tight_layout()
plt.show()

