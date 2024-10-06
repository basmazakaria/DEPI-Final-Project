# -*- coding: utf-8 -*-
"""Untitled23.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1POpAFWbWY3quDlWZgjSqlr89G5hEzOuu
"""

import pandas as pd
import matplotlib.pyplot as plt


# Load the dataset
file_path = 'data_cleaned_final (1).csv'
data = pd.read_csv(file_path)

# Show basic information about the dataset and the first few rows
data_info = data.info()
data_head = data.head()

data_info, data_head

import plotly.express as px

# Order Success Rate
order_success_rate = (data['is_successful'].sum() / len(data)) * 100

# Create the plot
fig = px.bar(x=[order_success_rate], y=['Order Success Rate'], orientation='h',
             labels={'x': 'Percentage'}, title='Order Success Rate')
fig.update_traces(marker_color='#4CAF50', text=[f'{order_success_rate:.2f}%'], textposition='auto')
fig.show()

# Average Order Value (AOV)
average_order_value = data['gmv_amount_lc'].mean()

# Create the plot
fig = px.bar(x=[average_order_value], y=['Average Order Value (LC)'], orientation='h',
             labels={'x': 'Local Currency'}, title='Average Order Value')
fig.update_traces(marker_color='#2196F3', text=[f'{average_order_value:.2f}'], textposition='auto')
fig.show()

# Average Order Delay
average_order_delay = data['order_delay'].mean()

# Create the plot
fig = px.bar(x=[average_order_delay], y=['Average Order Delay (mins)'], orientation='h',
             labels={'x': 'Minutes'}, title='Average Order Delay')
fig.update_traces(marker_color='#FF5722', text=[f'{average_order_delay:.2f} mins'], textposition='auto')
fig.show()

# Customer Acquisition Rate
acquisition_rate = (data['is_acquisition'].sum() / len(data)) * 100

# Create the plot
fig = px.bar(x=[acquisition_rate], y=['Customer Acquisition Rate'], orientation='h',
             labels={'x': 'Percentage'}, title='Customer Acquisition Rate')
fig.update_traces(marker_color='#FFC107', text=[f'{acquisition_rate:.2f}%'], textposition='auto')
fig.show()

# Delivery Success Rate: Orders with a delay less than or equal to 0 are considered successful.
on_time_deliveries = data[data['order_delay'] <= 0].shape[0]
delivery_success_rate = (on_time_deliveries / total_orders) * 100

# Plotting Delivery Success Rate
fig = px.bar(x=[delivery_success_rate], y=['Delivery Success Rate'], orientation='h',
             labels={'x': 'Percentage'}, title='Delivery Success Rate')
fig.update_traces(marker_color='#8BC34A', text=[f'{delivery_success_rate:.2f}%'], textposition='auto')
fig.show()

# Average Delivery Fee
average_delivery_fee = data['delivery_fee_amount_lc'].mean()

# Plotting Average Delivery Fee
fig = px.bar(x=[average_delivery_fee], y=['Average Delivery Fee'], orientation='h',
             labels={'x': 'Local Currency'}, title='Average Delivery Fee')
fig.update_traces(marker_color='#673AB7', text=[f'{average_delivery_fee:.2f}'], textposition='auto')
fig.show()

# Top 5 Cities by Order Count
top_5_cities = data['City'].value_counts().nlargest(5)

# Plotting Top 5 Cities
fig = px.bar(x=top_5_cities.values, y=top_5_cities.index, orientation='h',
             labels={'x': 'Order Count', 'y': 'City'}, title='Top 5 Cities by Order Count')
fig.update_traces(marker_color='#FF9800', text=top_5_cities.values, textposition='auto')
fig.show()

# Calculate success rate per platform
successful_orders_by_platform = data.groupby('platform')['is_successful'].mean() * 100

# Plotting Successful Orders by Platform
fig = px.bar(successful_orders_by_platform, x=successful_orders_by_platform.index,
             y=successful_orders_by_platform.values,
             labels={'x': 'Platform', 'y': 'Success Rate (%)'}, title='Successful Orders by Platform')
fig.update_traces(marker_color='#4CAF50', text=successful_orders_by_platform.values.round(2), textposition='auto')
fig.show()

# Calculate average order value per city
average_order_value_by_city = data.groupby('City')['gmv_amount_lc'].mean().nlargest(10)

# Plotting Average Order Value by City
fig = px.bar(x=average_order_value_by_city.values, y=average_order_value_by_city.index, orientation='h',
             labels={'x': 'Average Order Value (LC)', 'y': 'City'}, title='Top 10 Cities by Average Order Value')
fig.update_traces(marker_color='#2196F3', text=average_order_value_by_city.values.round(2), textposition='auto')
fig.show()

# Calculate acquisition rate per platform
acquisition_rate_by_platform = data.groupby('platform')['is_acquisition'].mean() * 100

# Plotting Customer Acquisition Rate by Platform
fig = px.bar(acquisition_rate_by_platform, x=acquisition_rate_by_platform.index,
             y=acquisition_rate_by_platform.values,
             labels={'x': 'Platform', 'y': 'Acquisition Rate (%)'}, title='Customer Acquisition Rate by Platform')
fig.update_traces(marker_color='#FF9800', text=acquisition_rate_by_platform.values.round(2), textposition='auto')
fig.show()

# Calculate average actual and promised delivery time
average_times = data[['actual_delivery_time', 'promised_delivery_time']].mean()

# Plotting Average Delivery Time vs. Promised Time
fig = px.bar(x=average_times.index, y=average_times.values,
             labels={'x': 'Time Type', 'y': 'Time (mins)'}, title='Average Delivery Time vs. Promised Time')
fig.update_traces(marker_color=['#FF5722', '#8BC34A'], text=average_times.values.round(2), textposition='auto')
fig.show()

# Calculate top vendors by average order value
top_vendors_by_aov = data.groupby('vendor_id')['gmv_amount_lc'].mean().nlargest(10)

# Plotting Top Vendors by Average Order Value
fig = px.bar(x=top_vendors_by_aov.values, y=top_vendors_by_aov.index, orientation='h',
             labels={'x': 'Average Order Value (LC)', 'y': 'Vendor ID'}, title='Top 10 Vendors by Average Order Value')
fig.update_traces(marker_color='#00BCD4', text=top_vendors_by_aov.values.round(2), textposition='auto')
fig.show()

# Calculate success rate by city and platform
success_rate_city_platform = data.groupby(['City', 'platform'])['is_successful'].mean().unstack() * 100

# Plotting Success Rate by City and Platform
fig = px.imshow(success_rate_city_platform,
                labels=dict(x="Platform", y="City", color="Success Rate (%)"),
                title="Order Success Rate by City and Platform")
fig.show()

# Calculate average delivery time per city
average_delivery_time_by_city = data.groupby('City')['actual_delivery_time'].mean().nlargest(10)

# Plotting Average Delivery Time by City
fig = px.bar(x=average_delivery_time_by_city.values, y=average_delivery_time_by_city.index, orientation='h',
             labels={'x': 'Average Delivery Time (mins)', 'y': 'City'}, title='Top 10 Cities by Average Delivery Time')
fig.update_traces(marker_color='#FF5722', text=average_delivery_time_by_city.values.round(2), textposition='auto')
fig.show()

# Scatter plot of delivery fee vs. basket amount
fig = px.scatter(data, x='basket_amount_lc', y='delivery_fee_amount_lc',
                 labels={'x': 'Basket Amount (LC)', 'y': 'Delivery Fee (LC)'},
                 title='Delivery Fee vs. Basket Amount')
fig.show()

# Calculate successful orders by zone
successful_orders_by_zone = data[data['is_successful'] == 1]['Zone'].value_counts().nlargest(10)

# Plotting Top Zones by Successful Orders
fig = px.bar(x=successful_orders_by_zone.values, y=successful_orders_by_zone.index, orientation='h',
             labels={'x': 'Successful Orders', 'y': 'Zone'}, title='Top 10 Zones by Successful Orders')
fig.update_traces(marker_color='#8BC34A', text=successful_orders_by_zone.values, textposition='auto')
fig.show()

# Calculate acquisition rate per city
acquisition_rate_by_city = data.groupby('City')['is_acquisition'].mean().nlargest(10) * 100

# Plotting Customer Acquisition Rate by City
fig = px.bar(x=acquisition_rate_by_city.values, y=acquisition_rate_by_city.index, orientation='h',
             labels={'x': 'Acquisition Rate (%)', 'y': 'City'}, title='Top 10 Cities by Customer Acquisition Rate')
fig.update_traces(marker_color='#FFC107', text=acquisition_rate_by_city.values.round(2), textposition='auto')
fig.show()

# Calculate average order delay by platform
avg_order_delay_by_platform = data.groupby('platform')['order_delay'].mean()

# Plotting Average Order Delay by Platform
fig = px.bar(x=avg_order_delay_by_platform.index, y=avg_order_delay_by_platform.values,
             labels={'x': 'Platform', 'y': 'Average Order Delay (mins)'},
             title='Average Order Delay by Platform')
fig.update_traces(marker_color='#FF5722', text=avg_order_delay_by_platform.values.round(2), textposition='auto')
fig.show()

# Group data by vendor and calculate successful and total orders
vendor_success = data.groupby('vendor_id')['is_successful'].agg(['sum', 'count'])
vendor_success.columns = ['Successful Orders', 'Total Orders']
vendor_success['Success Rate'] = (vendor_success['Successful Orders'] / vendor_success['Total Orders']) * 100

# Plotting Successful vs. Total Orders by Vendor
fig = px.bar(vendor_success.nlargest(10, 'Total Orders'),
             y=vendor_success.nlargest(10, 'Total Orders').index,
             x=['Successful Orders', 'Total Orders'],
             labels={'value': 'Order Count', 'index': 'Vendor ID'},
             title='Successful Orders vs. Total Orders by Vendor')
fig.show()

# Calculate delivery fee as a percentage of basket amount
data['delivery_fee_percentage'] = (data['delivery_fee_amount_lc'] / data['basket_amount_lc']) * 100

# Plot delivery fee percentage distribution
fig = px.histogram(data, x='delivery_fee_percentage', nbins=50,
                   labels={'delivery_fee_percentage': 'Delivery Fee as % of Basket Amount'},
                   title='Distribution of Delivery Fee as Percentage of Basket Amount')
fig.show()

# Group data by order count and delay
order_count_delay = data.groupby('order_id')['order_delay'].agg(['count', 'mean'])

# Plot Order Count vs. Average Delay
fig = px.scatter(order_count_delay, x='count', y='mean',
                 labels={'count': 'Number of Orders', 'mean': 'Average Delay (mins)'},
                 title='Order Count vs. Average Delay')
fig.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import ipywidgets as widgets
from IPython.display import display

# Load your data
data = pd.read_csv('data_cleaned_final (1).csv')

# Convert order_time to datetime
data['order_time'] = pd.to_datetime(data['order_time'])



# Create interactive widgets
recency_slider = widgets.IntSlider(value=100, min=0, max=300, step=1, description='Max Recency:')
frequency_slider = widgets.IntSlider(value=1, min=0, max=10, step=1, description='Min Frequency:')
monetary_slider = widgets.FloatSlider(value=100, min=0, max=1000, step=1, description='Min Monetary:')

# Use interact to link the function to the widgets
widgets.interactive(filter_customers,
                    recency=recency_slider,
                    frequency=frequency_slider,
                    monetary=monetary_slider)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import ipywidgets as widgets
from IPython.display import display

# Load your data
data = pd.read_csv('data_cleaned_final (1).csv')

# Convert order_time to datetime
data['order_time'] = pd.to_datetime(data['order_time'])

# Create interactive widgets
recency_slider = widgets.IntSlider(value=100, min=0, max=300, step=1, description='Max Recency:')
frequency_slider = widgets.IntSlider(value=1, min=0, max=10, step=1, description='Min Frequency:')
monetary_slider = widgets.FloatSlider(value=100, min=0, max=1000, step=1, description='Min Monetary:')

# Define the filter_customers function
def filter_customers(recency, frequency, monetary):
    # Add your filtering logic here based on recency, frequency, and monetary values
    # This is just an example, replace with your actual filtering criteria
    filtered_data = data[(data['order_time'].max() - data['order_time']).dt.days <= recency]
    # ... further filtering based on frequency and monetary ...

    # Display the filtered data or perform further analysis
    print(filtered_data.head())

# Use interact to link the function to the widgets
widgets.interactive(filter_customers,
                    recency=recency_slider,
                    frequency=frequency_slider,
                    monetary=monetary_slider)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import ipywidgets as widgets
from IPython.display import display
import os

# Set the environment variable to prevent memory leak
os.environ['OMP_NUM_THREADS'] = '1'

# Load your data
data = pd.read_csv('data_cleaned_final (1).csv')

# Convert order_time to datetime
data['order_time'] = pd.to_datetime(data['order_time'])

# Calculate RFM metrics
current_date = data['order_time'].max()
rfm_df = data.groupby('analytical_customer_id').agg({
    'order_time': lambda x: (current_date - x.max()).days,  # Recency
    'order_id': 'count',                                   # Frequency
    'gmv_amount_lc': 'sum'                                 # Monetary
}).rename(columns={'order_time': 'recency',
                   'order_id': 'frequency',
                   'gmv_amount_lc': 'monetary'})

# Normalize the RFM data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_df)

# Fit MiniBatch K-Means with the chosen number of clusters
optimal_k = 4  # Replace with the optimal k you choose
kmeans = MiniBatchKMeans(n_clusters=optimal_k, random_state=42, n_init=10, batch_size=1024)  # Increased batch_size
rfm_df['segment'] = kmeans.fit_predict(rfm_scaled)

# Function to visualize RFM data
def plot_rfm_data(rfm_df, filtered=False):
    plt.figure(figsize=(12, 6))

    # Plot Recency, Frequency, and Monetary
    plt.subplot(1, 3, 1)
    plt.bar(rfm_df.index, rfm_df['recency'], color='blue')
    plt.title('Recency')
    plt.xticks(rotation=90)

    plt.subplot(1, 3, 2)
    plt.bar(rfm_df.index, rfm_df['frequency'], color='orange')
    plt.title('Frequency')
    plt.xticks(rotation=90)

    plt.subplot(1, 3, 3)
    plt.bar(rfm_df.index, rfm_df['monetary'], color='green')
    plt.title('Monetary')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()

# Define the interactive filtering function
def filter_customers(recency, frequency, monetary):
    filtered_df = rfm_df[(rfm_df['recency'] <= recency) &
                          (rfm_df['frequency'] >= frequency) &
                          (rfm_df['monetary'] >= monetary)]

    print(f"Filtered Customers (Total: {filtered_df.shape[0]}):")
    display(filtered_df)

    # Plot the filtered data
    plot_rfm_data(filtered_df)

# Create interactive widgets
recency_slider = widgets.IntSlider(value=100, min=0, max=300, step=1, description='Max Recency:')
frequency_slider = widgets.IntSlider(value=1, min=0, max=10, step=1, description='Min Frequency:')
monetary_slider = widgets.FloatSlider(value=100, min=0, max=1000, step=1, description='Min Monetary:')

# Use interact to link the function to the widgets
widgets.interactive(filter_customers,
                    recency=recency_slider,
                    frequency=frequency_slider,
                    monetary=monetary_slider)

# Initial Plot for Full Data
plot_rfm_data(rfm_df)

# Extract hour of order placement
data['order_hour'] = pd.to_datetime(data['order_time']).dt.hour

# Create time of day categories: Morning, Afternoon, Evening, Night
conditions = [
    (data['order_hour'] >= 6) & (data['order_hour'] < 12),
    (data['order_hour'] >= 12) & (data['order_hour'] < 18),
    (data['order_hour'] >= 18) & (data['order_hour'] < 24),
    (data['order_hour'] >= 0) & (data['order_hour'] < 6)
]
choices = ['Morning', 'Afternoon', 'Evening', 'Night']

#Import numpy
import numpy as np
data['time_of_day'] = np.select(conditions, choices, default='Unknown')

# Calculate success rate by time of day and platform
success_rate_time_platform = data.groupby(['time_of_day', 'platform'])['is_successful'].mean().unstack() * 100

# Plotting Success Rate by Time of Day and Platform
fig = px.imshow(success_rate_time_platform,
                labels=dict(x="Platform", y="Time of Day", color="Success Rate (%)"),
                title="Order Success Rate by Time of Day and Platform")
fig.show()