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
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223052.png>)
```
df.shape
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 224803.png>)
```
df.describe()
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223059.png>)
```
df.info()
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223105.png>)
```
df.head(10)
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223105.png>)
```
df.tail(10)
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223115.png>)
```
df.isna().sum()
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223120.png>)
```
df.dropna(how='any').shape
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223126.png>)
```
df.shape
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223131.png>)
```
x = df.dropna(how='any')
x
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223137.png>)
```
mn = df.TOTAL.mean()
mn
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223141.png>)
```
df.TOTAL.fillna(mn,inplace = True)
df
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223209.png>)
```
df.isnull().sum()
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223217.png>)
```
df.M1.fillna(method = 'ffill', inplace = True)
df
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223226.png>)
```
df.isnull().sum()
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223231.png>)
```
df.M2.fillna(method = 'ffill', inplace = True)
df
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223249.png>)
```
df.isna().sum()
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223256.png>)
```
df.M3.fillna(method = 'ffill', inplace = True)
df
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223311.png>)
```
df.isnull().sum()
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223318.png>)
```
df.duplicated()
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223324.png>)
```
df.drop_duplicates(inplace = True)
df
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223331.png>)
```
df.duplicated()
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223337.png>)
```
df['DOB']
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223344.png>)
```
import seaborn as sns
sns.heatmap(df.isnull() , yticklabels = False , annot = True)
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223350.png>)
```
df.dropna(inplace = True)
sns.heatmap(df.isnull() , yticklabels = False , annot = True)
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223355.png>)
```
import pandas as pd
age = [1,3,28,27,25,92,30,39,40,50,26,24,29,94]
af = pd.DataFrame(age)
af
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223400.png>)
```
import seaborn as sns
sns.boxplot(data = af)
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223404.png>)
```
sns.scatterplot(data = af)
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223414.png>)
```
q1 = af.quantile(0.25)
q2 = af.quantile(0.5)
q3 = af.quantile(0.75)
iqr = q3-q1
iqr
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223420.png>)
```
import numpy as np
Q1 = np.percentile(af,25)
Q3 = np.percentile(af,75)
IQR = Q3-Q1
IQR
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223423.png>)
```
lower_bound = Q1-1.5*IQR
upper_bound = Q3+1.5*IQR
lower_bound
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223428.png>)
```
upper_bound
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223433.png>)
```
outliers = [x for x in age if x < lower_bound or x > upper_bound]
print("Q1:",Q1)
print("Q3:",Q3)
print("IQR:",IQR)
print("Lower_bound:",lower_bound)
print("Upper_bound:",upper_bound)
print("Outliers:",outliers)
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223440.png>)
```
df = df[((df>=lower_bound) & (df<=upper_bound))]
df
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223445.png>)
```
df=df.dropna()
df
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223451.png>)
```
sns.boxplot(data = df)
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223454.png>)
```
sns.scatterplot(data = df)
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223459.png>)
```
data = [1,2,2,2,3,1,1,15,2,2,2,3,1,1,2]
mean = np.mean(data)
std = np.std(data)
print("Mean of the dataset is",mean)
print("Standard deviation of the dataset is",std)
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223503.png>)
```
threshold = 3
outlier = []
for i in data:
  z = (i-mean)/std
  if z > threshold:
    outlier.append(i)
print("Outlier of the dataset is",outlier)
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223507.png>)
```
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
data = {
    'weight' : [12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 202, 72, 75, 78, 81, 84, 232, 87, 90, 93, 96, 99,258]
}
df=pd.DataFrame(data)
df
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223518.png>)
```
z = np.abs(stats.zscore(df))
print(df[z['weight']>3])
```
![alt text](<Output Screenshots/Screenshot 2025-04-23 223523.png>)
# Result
Data cleaning was successfully performed using Python libraries like NumPy, Pandas, and Seaborn, and the results were verified.
