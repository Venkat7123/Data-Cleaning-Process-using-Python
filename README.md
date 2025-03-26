# Data-Cleaning-Process-using-Python
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
```
import pandas as pd

df = pd.read_csv('/content/SAMPLEIDS (1).csv')

df

df.shape

df.describe()

df.info()

df.head(10)

df.tail(10)

df.isna().sum()

df.dropna(how='any').shape

df.shape

x = df.dropna(how='any')
x

mn = df.TOTAL.mean()
mn

df.TOTAL.fillna(mn,inplace = True)
df

df.isnull().sum()

df.M1.fillna(method = 'ffill', inplace = True)
df

df.isnull().sum()

df.M2.fillna(method = 'ffill', inplace = True)
df

df.isna().sum()

df.M3.fillna(method = 'ffill', inplace = True)
df

df.isnull().sum()

df.duplicated()

df.drop_duplicates(inplace = True)
df

df.duplicated()

df['DOB']

import seaborn as sns
sns.heatmap(df.isnull() , yticklabels = False , annot = True)

df.dropna(inplace = True)
sns.heatmap(df.isnull() , yticklabels = False , annot = True)

import pandas as pd
age = [1,3,28,27,25,92,30,39,40,50,26,24,29,94]
af = pd.DataFrame(age)
af

import seaborn as sns
sns.boxplot(data = af)

sns.scatterplot(data = af)

q1 = af.quantile(0.25)
q2 = af.quantile(0.5)
q3 = af.quantile(0.75)
iqr = q3-q1
iqr

import numpy as np
Q1 = np.percentile(af,25)
Q3 = np.percentile(af,75)
IQR = Q3-Q1
IQR

lower_bound = Q1-1.5*IQR
upper_bound = Q3+1.5*IQR

lower_bound

upper_bound

outliers = [x for x in age if x < lower_bound or x > upper_bound]
print("Q1:",Q1)
print("Q3:",Q3)
print("IQR:",IQR)
print("Lower_bound:",lower_bound)
print("Upper_bound:",upper_bound)
print("Outliers:",outliers)

df = df[((df>=lower_bound) & (df<=upper_bound))]
df

df=df.dropna()
df

sns.boxplot(data = df)

sns.scatterplot(data = df)

data = [1,2,2,2,3,1,1,15,2,2,2,3,1,1,2]
mean = np.mean(data)
std = np.std(data)
print("Mean of the dataset is",mean)
print("Standard deviation of the dataset is",std)

threshold = 3
outlier = []
for i in data:
  z = (i-mean)/std
  if z > threshold:
    outlier.append(i)
print("Outlier of the dataset is",outlier)

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
data = {
    'weight' : [12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 202, 72, 75, 78, 81, 84, 232, 87, 90, 93, 96, 99,258]
}
df=pd.DataFrame(data)
df

z = np.abs(stats.zscore(df))
print(df[z['weight']>3])
```
# Result
Data cleaning was successfully performed using Python libraries like NumPy, Pandas, and Seaborn, and the results were verified.
