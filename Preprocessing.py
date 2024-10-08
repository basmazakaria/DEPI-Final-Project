# -*- coding: utf-8 -*-
"""DEPI.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oKlQQm0k8WmiH_tSn9IEwxfkD7v3RxYB

#import
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""#Read and Clean Data

"""

dim_vendor_df =pd.read_excel('dim_vendor.xlsx')
dim_vendor_df.head()

fct_order_df =pd.read_excel('fct_order.xlsx')
fct_order_df.head()

# Merging the 'dim_vendor' and 'fct_order' DataFrames on the 'vendor_id' column
merged_df = pd.merge(fct_order_df, dim_vendor_df, on='vendor_id')

# Display the first few rows of the merged DataFrame
merged_df.head()

merged_df.to_excel('Talabat Dataset after merged.xlsx', index=False)

merged_df.head()

merged_df.isnull().sum()

# Handling missing values with a mixed approach

# 1. Fill missing categorical columns with "Unknown" or similar placeholders
merged_df['reason'].fillna('Successful order', inplace=True)
merged_df['sub_reason'].fillna('Successful order', inplace=True)
merged_df['owner'].fillna('Successful order', inplace=True)

# 2. For numerical columns, we will fill missing values with the median
merged_df['affordability_amt_total'].fillna('0', inplace=True)


# Verify the cleaned dataset
merged_df.isnull().sum()

data_cleaned=merged_df.copy()
# If both 'actual_delivery_time' and 'promised_delivery_time' are null, fill them with zeros along with 'order_delay'.

# Check if both 'actual_delivery_time' and 'promised_delivery_time' are null, then fill those and 'order_delay' with zeros.
data_cleaned.loc[
    (data_cleaned['actual_delivery_time'].isnull()) & (data_cleaned['promised_delivery_time'].isnull()),
    ['actual_delivery_time', 'promised_delivery_time', 'order_delay', 'dropoff_distance_manhattan']
] = 0

# Display the updated dataset to the user
data_cleaned.isnull().sum()

# Apply the condition to check for null 'actual_delivery_time' and update 'order_delay' to zero if true
data_cleaned.loc[
    data_cleaned['actual_delivery_time'].isnull(), 'order_delay'
] = 0
data_cleaned.isnull().sum()

data_cleaned.to_csv('data_cleaned.csv', index=False)

# Fill missing (NaN) values in 'order_delay' with 0
data_cleaned['order_delay'].fillna(0, inplace=True)

data_cleaned.isnull().sum()

data_cleaned =pd.read_csv('data_cleaned_final.csv')

# Applying the new condition as requested
data_cleaned.loc[
    data_cleaned['City'].isin(["Shebeen El Koom", "Al Mahallah Al Kubra"]), 'City Cluster'
] = "Delta"

data_cleaned.loc[
    data_cleaned['City'].isin(["El Gouna", "Suez"]), 'City Cluster'
] = "ESM"

data_cleaned.isnull().sum()

# Apply the condition to check if 'actual_delivery_time' is greater than 0 and calculate 'order_delay'
data_cleaned.loc[
    data_cleaned['actual_delivery_time'] > 0, 'order_delay'
] = (data_cleaned['actual_delivery_time'] - data_cleaned['promised_delivery_time'])

# Drop rows with any null values and assign the result to a new variable
data_cleaned_final = data_cleaned.dropna().copy()

# Verify that there are no more null values
data_cleaned_final.isnull().sum()

data_cleaned_final.info()

data_cleaned_final.isnull().sum()

data_cleaned_final.to_csv('data_cleaned_final.csv', index=False)

data_cleaned.isnull().sum()

data_cleaned.info()

# Generate summary statistics of the numerical columns
summary_statistics = data_cleaned.describe()

# Display the result
print(summary_statistics)

# Proportion of successful and unsuccessful orders
proportion_success = data_cleaned['is_successful'].value_counts(normalize=True) * 100

# Display the proportion
print(proportion_success)

# Select only the numeric columns from the dataset
numeric_data = data_cleaned.select_dtypes(include=['number'])

# Calculate correlations between numerical columns
correlation_matrix = numeric_data.corr()

data_cleaned.hist()

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Ensure plots display inline
plt.rcParams["figure.figsize"] = (10, 6)

# 1. Proportion of successful and unsuccessful orders
proportion_success = data_cleaned['is_successful'].value_counts(normalize=True) * 100
plt.figure(figsize=(8, 6))
sns.barplot(x=proportion_success.index, y=proportion_success.values, palette="Blues_d")
plt.title('Proportion of Successful vs Unsuccessful Orders')
plt.ylabel('Percentage')
plt.xlabel('Order Success Status')
plt.show()

# 2. Segmentation Overview (Treemap)
fig_treemap = px.treemap(data_cleaned, path=['City', 'vendor_id'], values='order_id',
                         title="Segmentation Overview of Orders by City and Vendor",
                         color='gmv_amount_lc')
fig_treemap.show()

# 3. Distribution of GMV Amount
plt.figure(figsize=(10, 6))
plt.hist(data_cleaned['gmv_amount_lc'].dropna(), bins=30, edgecolor='black', color='skyblue')
plt.title('Distribution of GMV Amount')
plt.xlabel('GMV Amount')
plt.ylabel('Frequency')
plt.show()

data_cleaned.dtypes