import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.interchange.dataframe_protocol import DataFrame
from sympy import false

# upload dataset
df=pd.read_csv(r"C:\Users\vishn\Downloads\Bengaluru_House_Data.csv")

# we don't need area-type , society , balcony , availability to predict house price . Because these are not  decide the house price , house price is mainly based on location and size
df1=df.drop(['area_type' , 'society' , 'balcony' , 'availability'] , axis='columns')

# here we are removing null values
df2=df1.dropna()

# in the dataset "size" having different type of data which is difficult to predict house price , so we are taking how many bedrooms that are present in "size"
df2 = df1.dropna().copy()
df2['BHK'] = df2['size'].apply(lambda x :int(x.split(' ')[0]))
#pd.set_option("display.max_rows", 100)
#print(df2.head(100))

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
#print(df2[~df2['total_sqft'].apply(is_float)])

# so when two two values are present in "total_sqt" we are taking average of those two values
def average(x):
    tokens=x.split('-')
    if len(tokens)==2:
        return(float(tokens[0]) + float(tokens[0])) / 2
    try:
        return float(x)
    except:
        return None


df3=df2.copy()
df3['total_sqft']=df3['total_sqft'].apply(average)

# and here we have clean data








df4=df3.copy()
df4['price_per_sqt']=(df4['price']*100000) / df4['total_sqft']
total_locations=df4.groupby('location')['location'].agg('count').sort_values(ascending=False)

counts = df4['location'].value_counts()
df4 = df4[df4['location'].isin(counts[counts >= 10].index)]
print('len of df4' ,len(df4))

# usually per 1 bedroom requires 300 sqft but in the dataset some data points that having less that 300 sqft per 1 bedroom , so we have to remove those data
df5=df4[~(df4.total_sqft/df4.BHK<300)]
print('len of df5' ,len(df5))

def remove_outliers(df):
    df_out=pd.DataFrame()
    for key , subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqt)
        st=np.std(subdf.price_per_sqt)
        reduced_df=subdf[(subdf.price_per_sqt>(m-st)) & (subdf.price_per_sqt<=(m+st))]
        df_out=pd.concat([df_out,reduced_df] , ignore_index=True)
    return df_out
df6=remove_outliers(df5)
print('len of df6',len(df6))





def remove_bhk_outliers (df):
    exclude_indicies=np.array([])
    for location ,location_df in df.groupby('location'):
        bhk_stats= {}
        for bhk , bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean' : np.mean(bhk_df.price_per_sqt),
                'std' : np.std(bhk_df.price_per_sqt),
                'count' : bhk_df.shape[0]
            }
        for bhk , bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indicies=np.append(exclude_indicies , bhk_df[bhk_df.price_per_sqt<(stats['mean'])].index.values)
    return df.drop(exclude_indicies , axis='index')

df7=remove_bhk_outliers(df6)
print(df7.shape)

#it is not possible that no.of bedrooms < no.of bathrooms , so now we have to remove those bathrooms
df8=df7[df7.bath<df7.BHK+2]
print(len(df8))











# now we have location data as categorical ml models don't understand categorical data , so we have to convert categorical data into numeric data by using one hotencoder or get_dummies
categoric_to_numeric=pd.get_dummies(df8.location)


# so now we have to concat this data with dataframe
df9=pd.concat([df8,categoric_to_numeric],axis="columns")
print(df9.shape)

# so now we don't want size and price_per_sqft
df10=df9.drop(['size','price_per_sqt','location'],axis='columns')
print(df10.head(2))

# we need to separate the independent & dependent variable from dataset
# here x is independent variable
x=df10.drop('price',axis="columns")
# here y is dependent variable
y=df10.price

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# 1. Split dataset into training & testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 2. Create Linear Regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# 3. Evaluate model
y_pred = lr.predict(x_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


# 4. Prediction function
def predict_price(location, sqft, bath, bhk):
    # Create zero vector
    x_pred = np.zeros(len(x.columns))

    # Fill numeric features
    x_pred[x.columns.get_loc('total_sqft')] = sqft
    x_pred[x.columns.get_loc('bath')] = bath
    x_pred[x.columns.get_loc('BHK')] = bhk

    # Handle location (one-hot encoded)
    if location in x.columns:
        loc_index = x.columns.get_loc(location)
        x_pred[loc_index] = 1

    # Convert to DataFrame with same column names as training set
    x_pred_df = pd.DataFrame([x_pred], columns=x.columns)

    return lr.predict(x_pred_df)[0]

#predicted price of specific zone with sqft , bedroom & bathrooms
price = predict_price('1st Phase JP Nagar', 1000, 2, 2)
print("Predicted Price (in lakhs):", price)




