#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Let's first inspect the sheet provided to understand its structure and the data it contains.
import pandas as pd

# Load the CSV file

data = pd.read_csv('last_cleaned.csv')

# Display the first few rows and basic information to understand the data structure
data_info = data.info()
data_head = data.head()

data_info, data_head


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Preprocessing
# Convert 'order_date' and 'order_time' to datetime
data['order_datetime'] = pd.to_datetime(data['order_date'] + ' ' + data['order_time'])

# Extracting useful features from the datetime
data['order_day_of_week'] = data['order_datetime'].dt.dayofweek  # Monday=0, Sunday=6
data['order_hour'] = data['order_datetime'].dt.hour

# Drop the original 'order_date', 'order_time', and 'order_datetime' columns
data = data.drop(columns=['order_date', 'order_time', 'order_datetime', 'is_successful'])

# Label encoding for 'vendor_zone' and 'customer_zone'
label_encoder = LabelEncoder()
data['vendor_zone_encoded'] = label_encoder.fit_transform(data['vendor_zone'])
data['customer_zone_encoded'] = label_encoder.fit_transform(data['customer_zone'])

# Drop original categorical columns
data = data.drop(columns=['vendor_zone', 'customer_zone'])

# Define features (X) and target (y)
X = data[['vendor_zone_encoded', 'customer_zone_encoded', 'dropoff_distance_manhattan', 
          'promised_delivery_time', 'order_delay', 'order_day_of_week', 'order_hour']]
y = data['actual_delivery_time']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

rmse, model.coef_, model.intercept_


# In[31]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Assuming your data is already cleaned and processed

# Features and target
X = data[['vendor_zone_encoded','dropoff_distance_manhattan', 
          'promised_delivery_time','order_day_of_week', 'order_hour']]
y = data[ 'actual_delivery_time']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)  # حساب R²

print(f"RMSE: {rmse}")
print(f"R²: {r2}")  # طباعة R²


# In[32]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Assuming your data is already cleaned and processed

# Features and target
X = data[['vendor_zone_encoded', 'dropoff_distance_manhattan', 
          'promised_delivery_time', 'order_day_of_week', 'order_hour']]
y = data['actual_delivery_time']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # يمكنك ضبط n_estimators حسب الحاجة
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)  # حساب R²

print(f"RMSE: {rmse}")
print(f"R²: {r2}")  # طباعة R²


# In[36]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# إعداد نموذج Random Forest
model = RandomForestRegressor(random_state=42)

# إعداد param distributions
param_distributions = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}

# ضبط المعلمات باستخدام Randomized Search
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, 
                                   n_iter=5, cv=2, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)

# أفضل نموذج
best_model_random = random_search.best_estimator_

# التنبؤ باستخدام النموذج الأفضل
y_pred_best_random = best_model_random.predict(X_test)

# تقييم النموذج
mse_best_random = mean_squared_error(y_test, y_pred_best_random)
rmse_best_random = np.sqrt(mse_best_random)
r2_best_random = r2_score(y_test, y_pred_best_random)

print(f"Best RMSE (Randomized Search): {rmse_best_random}")
print(f"Best R² (Randomized Search): {r2_best_random}")


# In[39]:


from sklearn.ensemble import RandomForestRegressor

# Train the model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest RMSE: {rmse_rf}")
print(f"Random Forest R²: {r2_rf}")


# In[8]:


from sklearn.ensemble import RandomForestRegressor

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Random Forest Regressor:")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")


# In[4]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# حساب مصفوفة الارتباط للأعمدة العددية فقط
correlation_matrix = data.select_dtypes(include=['float64', 'int64']).corr()

# رسم مصفوفة الارتباط
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix")
plt.show()


# In[ ]:





# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Assuming your data is already cleaned and processed
# Features and target
X = data[['dropoff_distance_manhattan', 'promised_delivery_time', 'order_delay']]
y = data['actual_delivery_time']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")


# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# لنفترض أن لديك البيانات في DataFrame اسمه data

# تحويل الأعمدة إلى نوع بيانات التاريخ والوقت
data['order_date'] = pd.to_datetime(data['order_date'])
data['order_time'] = pd.to_datetime(data['order_time'])

# استخراج ميزات جديدة
data['order_day_of_week'] = data['order_date'].dt.dayofweek  # من 0 إلى 6
data['order_hour'] = data['order_time'].dt.hour  # من 0 إلى 23

# الميزات المستندة إلى البيانات العددية
X = data[['dropoff_distance_manhattan', 'promised_delivery_time', 'order_delay', 
           'order_day_of_week', 'order_hour']]
y = data['actual_delivery_time']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# التنبؤات
y_pred = model.predict(X_test)

# التقييم
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")


# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# إعداد البيانات (تأكد من أن البيانات الخاصة بك نظيفة ومعالجة)
# يمكنك تعديل X و y وفقًا للاحتياجات
X = data[['dropoff_distance_manhattan', 'promised_delivery_time', 'order_delay']]
y = data['actual_delivery_time']

# تقسيم البيانات إلى مجموعات التدريب والاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# إنشاء نموذج الانحدار الخطي
model = LinearRegression()

# تدريب النموذج
model.fit(X_train, y_train)

# إجراء التنبؤات
y_pred = model.predict(X_test)

# تقييم النموذج
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Linear Regression:")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")


# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# إعداد البيانات (تأكد من أن البيانات الخاصة بك نظيفة ومعالجة)
X = data[['dropoff_distance_manhattan', 'promised_delivery_time', 'order_delay']]
y = data['actual_delivery_time']

# تقسيم البيانات إلى مجموعات التدريب والاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# إنشاء نموذج الانحدار الخطي
model = LinearRegression()

# تدريب النموذج
model.fit(X_train, y_train)

# إجراء التنبؤات
y_pred = model.predict(X_test)

# تقييم النموذج
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Linear Regression:")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# إدخال بيانات المستخدم
dropoff_distance = float(input("Enter the dropoff distance (in meters): "))
promised_time = float(input("Enter the promised delivery time (in seconds): "))
order_delay = float(input("Enter the order delay (in seconds): "))

# إنشاء DataFrame للبيانات المدخلة
user_input = pd.DataFrame({
    'dropoff_distance_manhattan': [dropoff_distance],
    'promised_delivery_time': [promised_time],
    'order_delay': [order_delay]
})

# إجراء التنبؤ باستخدام المدخلات
user_prediction = model.predict(user_input)

print(f"Predicted actual delivery time: {user_prediction[0]} seconds")


# In[22]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Assume your data is already cleaned and processed
# Example training data (you can replace it with your actual data)
data = pd.DataFrame({
    'vendor_zone': ['mohandssen', 'nasr_city', 'zamalek', 'mohandssen', 'nasr_city'],
    'dropoff_distance_manhattan': [5454, 1234, 6789, 5432, 1231],
    'promised_delivery_time': [2211, 1340, 1789, 2230, 1455],
    'order_day_of_week': [1, 3, 5, 0, 2],
    'order_hour': [12, 9, 17, 11, 14],
    'actual_delivery_time': [30, 45, 50, 35, 40]
})

# Convert vendor zone names to numbers using LabelEncoder
label_encoder = LabelEncoder()
data['vendor_zone_encoded'] = label_encoder.fit_transform(data['vendor_zone'])

# Features and target
X = data[['vendor_zone_encoded', 'dropoff_distance_manhattan', 
          'promised_delivery_time', 'order_day_of_week', 'order_hour']]
y = data['actual_delivery_time']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# User interaction to input data
print("Please enter the following data:")

# User input
vendor_zone = input("Enter the vendor zone (e.g., mohandssen): ")
dropoff_distance_manhattan = float(input("Enter the dropoff distance (in meters): "))
promised_delivery_time = int(input("Enter the promised delivery time (in minutes): "))
order_day_of_week = int(input("Enter the order day of the week (0 for Monday, 6 for Sunday): "))
order_hour = int(input("Enter the order hour (0 to 23): "))

# Convert the vendor zone name to a numerical value using LabelEncoder
vendor_zone_encoded = label_encoder.transform([vendor_zone])[0]

# Combine the input data into a DataFrame
input_data = pd.DataFrame({
    'vendor_zone_encoded': [vendor_zone_encoded],
    'dropoff_distance_manhattan': [dropoff_distance_manhattan],
    'promised_delivery_time': [promised_delivery_time],
    'order_day_of_week': [order_day_of_week],
    'order_hour': [order_hour]
})

# Make predictions using the model
y_pred = model.predict(input_data)

# Display the result
print(f"\nEstimated delivery time: {y_pred[0]:.2f} minutes")


# In[ ]:





# In[ ]:





# In[ ]:




