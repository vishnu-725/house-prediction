import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================
# 1. Load cleaned DataFrame
# ==========================
with open("cleaned_df.pkl", "rb") as f:
    df = pickle.load(f)

# ==========================
# 2. Distribution of House Prices
# ==========================
plt.figure(figsize=(8,5))
plt.hist(df['price'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel("Price (Lakhs)")
plt.ylabel("Count")
plt.title("Distribution of House Prices")
plt.show()

# ==========================
# 3. Distribution of BHK
# ==========================
plt.figure(figsize=(8,5))
df['BHK'].value_counts().sort_index().plot(kind='bar', color='orange')
plt.xlabel("Number of Bedrooms (BHK)")
plt.ylabel("Count")
plt.title("Distribution of BHK")
plt.show()

# ==========================
# 4. Bathrooms vs BHK (Boxplot)
# ==========================
plt.figure(figsize=(8,5))
sns.boxplot(x="BHK", y="bath", data=df)
plt.title("Bathrooms vs BHK")
plt.show()

# ==========================
# 5. Top 10 Locations with Most Listings
# ==========================
top_locations = df['original_location'].value_counts().head(10)
plt.figure(figsize=(10,5))
top_locations.plot(kind='bar', color='green')
plt.xlabel("Location")
plt.ylabel("Number of Houses")
plt.title("Top 10 Locations with Most Listings")
plt.show()

# ==========================
# 6. Price vs Total Sqft (Scatterplot)
# ==========================
plt.figure(figsize=(8,5))
plt.scatter(df['total_sqft'], df['price'], alpha=0.5, color='purple')
plt.xlabel("Total Sqft")
plt.ylabel("Price (Lakhs)")
plt.title("Price vs Total Sqft")
plt.show()

# ==========================
# 7. Price per Sqft Distribution
# ==========================
plt.figure(figsize=(8,5))
plt.hist(df['price']*100000 / df['total_sqft'], bins=50, color='red', edgecolor='black')
plt.xlabel("Price per Sqft")
plt.ylabel("Count")
plt.title("Distribution of Price per Sqft")
plt.show()

# ==========================
# 8. Average Price by BHK
# ==========================
avg_price_bhk = df.groupby('BHK')['price'].mean()
plt.figure(figsize=(8,5))
avg_price_bhk.plot(kind='bar', color='blue')
plt.xlabel("BHK")
plt.ylabel("Average Price (Lakhs)")
plt.title("Average Price by BHK")
plt.show()

# ==========================
# 9. Average Price by Location (Top 10 Expensive)
# ==========================
top_avg_loc = df.groupby('original_location')['price'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,5))
top_avg_loc.plot(kind='bar', color='darkred')
plt.xlabel("Location")
plt.ylabel("Average Price (Lakhs)")
plt.title("Top 10 Expensive Locations")
plt.show()

