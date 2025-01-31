import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("used_cars.csv")

# Cleaning Data

data.head()
data.tail()
data.info()
data.nunique()
data.isnull().sum()
(data.isnull().sum()/(len(data))) * 100


# Removing the S.No column
data = data.drop(['S.No.'], axis=1)
data.info()


# Creating Features
from datetime import date

date.today().year
data['Car_Age'] = date.today().year - data['Year']
data.head()

data['Brand'] = data.Name.str.split().str.get(0)
data['Model'] = data.Name.str.split().str.get(1) + data.Name.str.split().str.get(2)
data[['Name', 'Brand', 'Model']]



# Data Cleaning
data.Brand.unique()
data.Brand.nunique()


searchfor = ['Isuzu', 'ISUZU', 'Mini', 'Land']
data[data.Brand.str.contains('|'.join(searchfor))].head(5)
data["Brand"].replace({"ISUZU": "Isuzu", "Mini": "Mini Cooper", "Land": "Land Rover"}, inplace=True)



# EDA - Exploratory Data Analysis


# Statistics Summary
data.describe().T
data.describe(include='all').T


cat_cols = data.select_dtypes(include=['object']).columns
num_cols = data.select_dtypes(include=np.number).columns.tolist()
print("Categorical Variables")
print(cat_cols)
print("Numerical Variable")
print(num_cols)


# EDA Univariate Analysis
for col in num_cols:
    print(col)
    print("Skew :", round(data[col].skew(), 2))
    plt.figure(figsize= (15, 4))
    plt.subplot(1, 2, 1)
    data[col].hist(grid=False)
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    sns.boxplot(x= data[col])
    plt.show()

fig, axes = plt.subplots(3, 2, figsize = (18, 18))
fig.suptitle('Bar plot for all Categorical variables in the dataset')
sns.countplot(ax = axes[0, 0], x='Fuel_Type', data = data, color = 'blue', order = data['Fuel_Type'].value_counts().index);
sns.countplot(ax = axes[0, 1], x='Transmission', data = data, color = 'blue', order = data['Transmission'].value_counts().index);
sns.countplot(ax = axes[1, 0], x = 'Owner_Type', data = data, color = 'blue', order = data['Owner_Type'].value_counts().index)
sns.countplot(ax = axes[1, 1], x = 'Location', data = data, color = 'blue', order = data['Location'].value_counts().index)
sns.countplot(ax = axes[2, 0], x = 'Brand', data = data, color = 'blue', order = data['Brand'].head(20).value_counts().index)
sns.countplot(ax = axes[2, 1], x = 'Model', data = data, color = 'blue', order = data['Model'].head(20).value_counts().index)

axes[1][1].tick_params(labelrotation=45);
axes[2][0].tick_params(labelrotation=90);
axes[2][1].tick_params(labelrotation=90);



# Data Transformation

# function for log transform of column 
def log_transform(data, col):
    for colname in col:
        if (data[colname] == 1.0).all():
            data[colname + '_log'] = np.log(data[colname] + 1)
        else:
            data[colname + '_log'] = np.log(data[colname])
    data.info()

log_transform(data, ['Kilometers_Driven', 'Price'])

#Log transformation of the feature 'Kilometers_Driven'
sns.distplot(data["Kilometers_Driven_log"], axlabel="Kilometers_Driven_log");


# EDA Bivariate Analysis
plt.figure(figsize=(13, 17))
sns.pairplot(data=data.drop(['Kilometers_Driven', 'Price'], axis=1))
plt.show()



#  bar plot
fig, axarr = plt.subplots(4, 2, figsize=(12, 18))
data.groupby('Location')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[0][0], fontsize=12)

axarr[0][0].set_title("Location Vs Price", fontsize=18)
data.groupby('Transmission')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[0][1], fontsize=12)
axarr[0][1].set_title("Transmission Vs Price", fontsize=18)
data.groupby('Fuel_Type')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[1][0], fontsize=12)
axarr[1][0].set_title("Fuel_Type Vs Price", fontsize=18)
data.groupby('Owner_Type')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[1][1], fontsize=12)
axarr[1][1].set_title("Owner_Type Vs Price", fontsize=18)
data.groupby('Brand')['Price_log'].mean().sort_values(ascending=False).head(10).plot.bar(ax=axarr[2][0], fontsize=12)
axarr[2][0].set_title("Brand Vs Price", fontsize=18)
data.groupby('Model')['Price_log'].mean().sort_values(ascending=False).head(10).plot.bar(ax=axarr[2][1], fontsize=12)
axarr[2][1].set_title("Model Vs Price", fontsize=18)
data.groupby('Seats')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[3][0], fontsize=12)
axarr[3][0].set_title("Seats Vs Price", fontsize=18)
data.groupby('Car_Age')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[3][1], fontsize=12)
axarr[3][1].set_title("Car_Age Vs Price", fontsize=18)
plt.subplots_adjust(hspace=1.0)
plt.subplots_adjust(wspace=.5)
sns.despine()



# EDA Multivariate Analysis
plt.figure(figsize=(12, 7))
sns.heatmap(data.drop(['Kilometers_Driven','Price'],axis=1).select_dtypes(include=['number']).corr(), annot = True, vmin = -1, vmax = 1)
plt.show()



# Impute Missing values
import pandas as pd
import numpy as np

# Ensure 'Mileage' column is numeric
data.loc[data["Mileage"] == 0.0, 'Mileage'] = np.nan
data['Mileage'] = pd.to_numeric(data['Mileage'], errors='coerce')
data['Mileage'].fillna(value=data['Mileage'].mean(), inplace=True)

# Handle missing 'Seats'
data['Seats'] = pd.to_numeric(data['Seats'], errors='coerce')
data['Seats'] = data.groupby(['Model', 'Brand'])['Seats'].transform(lambda x: x.fillna(x.median()))

# Handle missing 'Engine'
data['Engine'] = pd.to_numeric(data['Engine'], errors='coerce')
data['Engine'] = data.groupby(['Brand', 'Model'])['Engine'].transform(lambda x: x.fillna(x.median()))

# Handle missing 'Power'
data['Power'] = pd.to_numeric(data['Power'], errors='coerce')
data['Power'] = data.groupby(['Brand', 'Model'])['Power'].transform(lambda x: x.fillna(x.median()))

# Check missing values to confirm
print("Missing values after cleanup:")
print(data.isnull().sum())


from pandas_profiling import ProfileReport

# Generate the report

profile = ProfileReport(data, title="EDA Report", explorative=True)



# Save the report as HTML
profile.to_file("eda_report.html")