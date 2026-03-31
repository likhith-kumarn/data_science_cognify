import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("/content/Dataset .csv")
df.head()

df.shape
df.shape[0]
df.shape[1]

df.isnull().sum()

#if null exists
num_col=df.select_dtypes(include=np.number).columns
df[num_col]=df[num_col].fillna(df[num_col].median)

cat_col=df.select_dtypes(include=object).columns
for col in cat_col:
  df[col].fillna(df[col].mode()[0], inplace=True)

df['Aggregate rating'].value_counts()

plt.figure(figsize=(8,5))
sns.histplot(df['Aggregate rating'], bins=10, kde=True)
plt.title("Distribution of Aggregate Rating")
plt.xlabel("Aggregate Rating")
plt.ylabel("Count")
plt.show()

rating_counts = df['Aggregate rating'].value_counts(normalize=True)*100
rating_counts

df.describe()

df['Country Code'].value_counts()
df['City'].value_counts()
df['Cuisines'].value_counts()

df["City"].value_counts().head(10)
top_cuisine=df["Cuisines"].value_counts().head(10)

# Top Cities Plot
plt.figure(figsize=(10,5))
top_cities = df["City"].value_counts().head(10)
sns.barplot(x=top_cities.values, y=top_cities.index)
plt.title("Top Cities with Most Restaurants")
plt.xlabel("Number of Restaurants")
plt.ylabel("City")
plt.show()

# Top Cuisines Plot
plt.figure(figsize=(10,5))
sns.barplot(x=top_cuisine.values, y=top_cuisine.index)
plt.title("Top Cuisines")
plt.xlabel("Number of Restaurants")
plt.ylabel("Cuisines")
plt.show()

import folium
map_center=[df['Longitude'].mean(),df['Latitude'].mean()]

restaurant_map = folium.Map(location=map_center, zoom_start=5)
for i in range(len(df)):
    folium.Marker(
        location=[df.iloc[i]['Latitude'], df.iloc[i]['Longitude']],
        popup=df.iloc[i]['Restaurant Name']
    ).add_to(restaurant_map)
restaurant_map


country_dist = df['Country Code'].value_counts()
plt.figure(figsize=(8,5))
sns.barplot(x=country_dist.index, y=country_dist.values)
plt.title("Restaurants by Country Code")
plt.xlabel("Country Code")
plt.ylabel("Number of Restaurants")
plt.show()

correlation = df[['Latitude','Longitude','Aggregate rating']].corr()
correlation

plt.figure(figsize=(6,4))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Between Location and Rating")
plt.show()

table_booking_percent = df['Has Table booking'].value_counts(normalize=True) * 100
online_delivery_percent = df['Has Online delivery'].value_counts(normalize=True) * 100
table_booking_percent

online_delivery_percent

avg_rating_booking = df.groupby('Has Table booking')['Aggregate rating'].mean()
avg_rating_booking

plt.figure(figsize=(6,4))
sns.barplot(x=avg_rating_booking.index, y=avg_rating_booking.values)
plt.title("Average Ratings: Table Booking vs No Booking")
plt.xlabel("Table Booking")
plt.ylabel("Average Rating")
plt.show()

delivery_price = pd.crosstab(df['Price range'], df['Has Online delivery'])
delivery_price

plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Price range', hue='Has Online delivery')
plt.title("Online Delivery Availability Across Price Ranges")
plt.xlabel("Price Range")
plt.ylabel("Number of Restaurants")
plt.show()

price_range_counts = df['Price range'].value_counts()
price_range_counts

most_common_price = price_range_counts.idxmax()
most_common_price

avg_rating_price = df.groupby('Price range')['Aggregate rating'].mean()
avg_rating_price

plt.figure(figsize=(6,4))
sns.barplot(x=avg_rating_price.index, y=avg_rating_price.values)
plt.title("Average Rating by Price Range")
plt.xlabel("Price Range")
plt.ylabel("Average Rating")
plt.show()

rating_color = df.groupby('Rating color')['Aggregate rating'].mean()
rating_color

df['Restaurant_Name_Length'] = df['Restaurant Name'].apply(len)
df['Restaurant_Name_Length']

print(df[['Restaurant Name','Restaurant_Name_Length']].head())

df['Address_Length'] = df['Address'].apply(len)
df['Address_Length']
print(df[['Address','Address_Length']].head())

df['Has_Table_Booking'] = df['Has Table booking'].map({'Yes':1,'No':0})
print(df[['Has Table booking','Has_Table_Booking']].head())

df['Has_Online_Delivery'] = df['Has Online delivery'].map({'Yes':1,'No':0})
print(df[['Has Online delivery','Has_Online_Delivery']].head())

df['Cuisine_Count'] = df['Cuisines'].apply(lambda x: len(str(x).split(',')))
print(df[['Cuisines','Cuisine_Count']].head)

df.head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

X = df.drop(columns=['Aggregate rating','Restaurant Name','Address','Cuisines'])
y = df['Aggregate rating']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

mse_lr = mean_squared_error(y_test, pred_lr)
r2_lr = r2_score(y_test, pred_lr)
print("Linear Regression Results")
print("MSE:", mse_lr)
print("R2 Score:", r2_lr)

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)
mse_dt = mean_squared_error(y_test, pred_dt)
r2_dt = r2_score(y_test, pred_dt)

print("\nDecision Tree Results")
print("MSE:", mse_dt)
print("R2 Score:", r2_dt)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, pred_rf)
r2_rf = r2_score(y_test, pred_rf)

print("\nRandom Forest Results")
print("MSE:", mse_rf)
print("R2 Score:", r2_rf)

results = pd.DataFrame({
    "Model": ["Linear Regression","Decision Tree","Random Forest"],
    "MSE":[mse_lr,mse_dt,mse_rf],
    "R2 Score":[r2_lr,r2_dt,r2_rf]
})
print(results)

df['Cuisines'] = df['Cuisines'].astype(str)
df_cuisine = df.assign(Cuisines=df['Cuisines'].str.split(', ')).explode('Cuisines')
cuisine_rating = df_cuisine.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False)
cuisine_rating

plt.figure(figsize=(10,6))
sns.barplot(x=cuisine_rating.head(10).values, y=cuisine_rating.head(10).index)
plt.title("Top Cuisines with Highest Ratings")
plt.xlabel("Average Rating")
plt.ylabel("Cuisine")
plt.show()

cuisine_votes = df_cuisine.groupby('Cuisines')['Votes'].sum().sort_values(ascending=False)
cuisine_votes

plt.figure(figsize=(10,6))
sns.barplot(x=cuisine_votes.head(10).values, y=cuisine_votes.head(10).index)
plt.title("Most Popular Cuisines (Based on Votes)")
plt.xlabel("Total Votes")
plt.ylabel("Cuisine")
plt.show()

top_rated_cuisines = df_cuisine.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False)
top_rated_cuisines

sns.set(style="whitegrid")
plt.figure(figsize=(8,5))
sns.histplot(df['Aggregate rating'], bins=20, kde=True)
plt.title("Distribution of Restaurant Ratings")
plt.xlabel("Aggregate Rating")
plt.ylabel("Number of Restaurants")
plt.show()

rating_counts = df['Aggregate rating'].value_counts().sort_index()

plt.figure(figsize=(10,5))
sns.barplot(x=rating_counts.index, y=rating_counts.values)
plt.title("Rating Frequency Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

df['Cuisines'] = df['Cuisines'].astype(str)
df_cuisine = df.assign(Cuisines=df['Cuisines'].str.split(', ')).explode('Cuisines')
top_cuisines = df_cuisine['Cuisines'].value_counts().head(10).index
cuisine_rating = df_cuisine[df_cuisine['Cuisines'].isin(top_cuisines)].groupby('Cuisines')['Aggregate rating'].mean()

plt.figure(figsize=(10,6))
sns.barplot(x=cuisine_rating.values, y=cuisine_rating.index)
plt.title("Average Rating by Cuisine")
plt.xlabel("Average Rating")
plt.ylabel("Cuisine")
plt.show()

plt.figure(figsize=(8,6))
corr = df[['Aggregate rating','Votes','Price range','Latitude','Longitude']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Between Features")
plt.show()
