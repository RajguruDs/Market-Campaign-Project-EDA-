# Market Campaign Project


#### Data Inspection


```python
import pandas as pd 
import numpy as np 
# Importing the marketing dataset 
data=pd.read_csv('marketing_data.csv')
# Creating a DataFrame
df=pd.DataFrame(data)
# Inspecting the dataset
print(df.head())
print(df.tail())
print(df.info())
```

          ID  Year_Birth   Education Marital_Status      Income   Kidhome  \
    0   1826        1970  Graduation       Divorced  $84,835.00         0   
    1      1        1961  Graduation         Single  $57,091.00         0   
    2  10476        1958  Graduation        Married  $67,267.00         0   
    3   1386        1967  Graduation       Together  $32,474.00         1   
    4   5371        1989  Graduation         Single  $21,474.00         1   
    
       Teenhome Dt_Customer  Recency  MntWines  ...  NumStorePurchases  \
    0         0     6/16/14        0       189  ...                  6   
    1         0     6/15/14        0       464  ...                  7   
    2         1     5/13/14        0       134  ...                  5   
    3         1     5/11/14        0        10  ...                  2   
    4         0      4/8/14        0         6  ...                  2   
    
       NumWebVisitsMonth  AcceptedCmp3  AcceptedCmp4  AcceptedCmp5  AcceptedCmp1  \
    0                  1             0             0             0             0   
    1                  5             0             0             0             0   
    2                  2             0             0             0             0   
    3                  7             0             0             0             0   
    4                  7             1             0             0             0   
    
       AcceptedCmp2  Response  Complain  Country  
    0             0         1         0       SP  
    1             1         1         0       CA  
    2             0         0         0       US  
    3             0         0         0      AUS  
    4             0         1         0       SP  
    
    [5 rows x 28 columns]
             ID  Year_Birth   Education Marital_Status      Income   Kidhome  \
    2235  10142        1976         PhD       Divorced  $66,476.00         0   
    2236   5263        1977    2n Cycle        Married  $31,056.00         1   
    2237     22        1976  Graduation       Divorced  $46,310.00         1   
    2238    528        1978  Graduation        Married  $65,819.00         0   
    2239   4070        1969         PhD        Married  $94,871.00         0   
    
          Teenhome Dt_Customer  Recency  MntWines  ...  NumStorePurchases  \
    2235         1      3/7/13       99       372  ...                 11   
    2236         0     1/22/13       99         5  ...                  3   
    2237         0     12/3/12       99       185  ...                  5   
    2238         0    11/29/12       99       267  ...                 10   
    2239         2      9/1/12       99       169  ...                  4   
    
          NumWebVisitsMonth  AcceptedCmp3  AcceptedCmp4  AcceptedCmp5  \
    2235                  4             0             0             0   
    2236                  8             0             0             0   
    2237                  8             0             0             0   
    2238                  3             0             0             0   
    2239                  7             0             1             1   
    
          AcceptedCmp1  AcceptedCmp2  Response  Complain  Country  
    2235             0             0         0         0       US  
    2236             0             0         0         0       SP  
    2237             0             0         0         0       SP  
    2238             0             0         0         0      IND  
    2239             0             0         1         0       CA  
    
    [5 rows x 28 columns]
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2240 entries, 0 to 2239
    Data columns (total 28 columns):
     #   Column               Non-Null Count  Dtype 
    ---  ------               --------------  ----- 
     0   ID                   2240 non-null   int64 
     1   Year_Birth           2240 non-null   int64 
     2   Education            2240 non-null   object
     3   Marital_Status       2240 non-null   object
     4    Income              2216 non-null   object
     5   Kidhome              2240 non-null   int64 
     6   Teenhome             2240 non-null   int64 
     7   Dt_Customer          2240 non-null   object
     8   Recency              2240 non-null   int64 
     9   MntWines             2240 non-null   int64 
     10  MntFruits            2240 non-null   int64 
     11  MntMeatProducts      2240 non-null   int64 
     12  MntFishProducts      2240 non-null   int64 
     13  MntSweetProducts     2240 non-null   int64 
     14  MntGoldProds         2240 non-null   int64 
     15  NumDealsPurchases    2240 non-null   int64 
     16  NumWebPurchases      2240 non-null   int64 
     17  NumCatalogPurchases  2240 non-null   int64 
     18  NumStorePurchases    2240 non-null   int64 
     19  NumWebVisitsMonth    2240 non-null   int64 
     20  AcceptedCmp3         2240 non-null   int64 
     21  AcceptedCmp4         2240 non-null   int64 
     22  AcceptedCmp5         2240 non-null   int64 
     23  AcceptedCmp1         2240 non-null   int64 
     24  AcceptedCmp2         2240 non-null   int64 
     25  Response             2240 non-null   int64 
     26  Complain             2240 non-null   int64 
     27  Country              2240 non-null   object
    dtypes: int64(23), object(5)
    memory usage: 490.1+ KB
    None

Inferences from above dataset:
1: By insepecting the above dataset, The Income column has dirty values and name of the particular column contain unwanted     spaces.
2: The Date are not properly inserted in Dt_Customer 
3: Data description 
The variables such as birth year, education, income, and others pertain to the first
'P' or 'People' in the tabular data presented to the user. The expenditures on items
like wine, fruits, and gold, are associated with ‘Product’. Information relevant to
sales channels, such as websites and stores, is connected to ‘Place’, and the fields
discussing promotions and the outcomes of various campaigns are linked to
‘Promotion’.
#### Data Cleaning 



```python
# Converting the Dt_Customer to Datetime Object
df['Dt_Customer']=pd.to_datetime(df['Dt_Customer'])
# Renaming the Income column to Income without space
df.rename(columns={' Income ': 'Income'}, inplace=True)
# Clean 'Income' column
df['Income'] = df['Income'].str.replace(r'[$,]', '', regex=True).astype(float)
df['Income'].fillna(df['Income'].median(), inplace=True)
# Cleaning the Education and Martial Status Columns
df['Education'] = df['Education'].replace({
    '2n Cycle': 'Graduation', 
    'Basic': 'Basic',
    'Graduation': 'Graduation',
    'Master': 'Master',
    'PhD': 'PhD'
})
df['Marital_Status'] = df['Marital_Status'].replace({
    'Alone': 'Single',
    'Absurd': 'Single',
    'YOLO': 'Single',
    'Widow': 'Single'  
})
# Creating a variables to represent the total number of children, age, and total spending
df[['Kidhome','Teenhome']].value_counts()
df['Total_Children'] = df['Kidhome'] + df['Teenhome']
df['Age']=2025-df['Year_Birth']
spending_columns = ['MntWines', 'MntFruits', 'MntMeatProducts',
                    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df['Total_Spending'] = df[spending_columns].sum(axis=1)
df['Total_Purchases'] = (
    df['NumWebPurchases'] +
    df['NumCatalogPurchases'] +
    df['NumStorePurchases']
)
print(df['Income'])
print(df['Total_Children'])
print(df['Age'])
print(df['Total_Spending'])
print(df['Total_Purchases'])
```

    0       84835.0
    1       57091.0
    2       67267.0
    3       32474.0
    4       21474.0
             ...   
    2235    66476.0
    2236    31056.0
    2237    46310.0
    2238    65819.0
    2239    94871.0
    Name: Income, Length: 2240, dtype: float64
    0       0
    1       0
    2       1
    3       2
    4       1
           ..
    2235    1
    2236    1
    2237    1
    2238    0
    2239    2
    Name: Total_Children, Length: 2240, dtype: int64
    0       55
    1       64
    2       67
    3       58
    4       36
            ..
    2235    49
    2236    48
    2237    49
    2238    47
    2239    56
    Name: Age, Length: 2240, dtype: int64
    0       1190
    1        577
    2        251
    3         11
    4         91
            ... 
    2235     689
    2236      55
    2237     309
    2238    1383
    2239    1078
    Name: Total_Spending, Length: 2240, dtype: int64
    0       14
    1       17
    2       10
    3        3
    4        6
            ..
    2235    18
    2236     4
    2237    12
    2238    19
    2239    17
    Name: Total_Purchases, Length: 2240, dtype: int64


    C:\Users\karan\AppData\Local\Temp\ipykernel_7316\2388572470.py:2: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      df['Dt_Customer']=pd.to_datetime(df['Dt_Customer'])
    C:\Users\karan\AppData\Local\Temp\ipykernel_7316\2388572470.py:7: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      df['Income'].fillna(df['Income'].median(), inplace=True)

Inferences from Data Cleaning
1: Converting the column Dt_Customer to Datetime Object and Renaming the Income column to Income without space.
2: There are missing values in Income column so we are filling the null values with the median value and replacing the $ sign with blank space.
3: We are cleaning the Education and Martial Status Columns for no further complicaton
4: Creating a variables to represent the total number of children, age, and total spending for further data analysis.
#### Data Analysis


```python
import matplotlib.pyplot as plt
import seaborn as sns
# Handling outliers by IQR
def treat_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
cols = ['Income', 'Age', 'Total_Spending', 'Total_Purchases']
for col in cols:
    df = treat_outliers_iqr(df, col)
# Handling outliers by winsorizing
# from scipy.stats.mstats import winsorize
# for col in cols:
#     df[col] = winsorize(df[col], limits=[0.05, 0.05])
for col in cols:
    plt.figure(figsize=(12, 5))
    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Histogram of {col}')
    # Box plot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot of {col}')
    plt.tight_layout()
    plt.show() 

```

    C:\Users\karan\OneDrive\Desktop\Data Science\Module 3 - Data Science using python\.venv\Lib\site-packages\numpy\lib\_function_base_impl.py:4842: UserWarning: Warning: 'partition' will ignore the 'mask' of the MaskedArray.
      arr.partition(
    C:\Users\karan\OneDrive\Desktop\Data Science\Module 3 - Data Science using python\.venv\Lib\site-packages\numpy\lib\_function_base_impl.py:4842: UserWarning: Warning: 'partition' will ignore the 'mask' of the MaskedArray.
      arr.partition(
    C:\Users\karan\OneDrive\Desktop\Data Science\Module 3 - Data Science using python\.venv\Lib\site-packages\numpy\lib\_function_base_impl.py:4842: UserWarning: Warning: 'partition' will ignore the 'mask' of the MaskedArray.
      arr.partition(
    C:\Users\karan\OneDrive\Desktop\Data Science\Module 3 - Data Science using python\.venv\Lib\site-packages\numpy\lib\_function_base_impl.py:4842: UserWarning: Warning: 'partition' will ignore the 'mask' of the MaskedArray.
      arr.partition(
    C:\Users\karan\OneDrive\Desktop\Data Science\Module 3 - Data Science using python\.venv\Lib\site-packages\numpy\lib\_function_base_impl.py:4842: UserWarning: Warning: 'partition' will ignore the 'mask' of the MaskedArray.
      arr.partition(
    C:\Users\karan\OneDrive\Desktop\Data Science\Module 3 - Data Science using python\.venv\Lib\site-packages\numpy\lib\_function_base_impl.py:4842: UserWarning: Warning: 'partition' will ignore the 'mask' of the MaskedArray.
      arr.partition(
    C:\Users\karan\OneDrive\Desktop\Data Science\Module 3 - Data Science using python\.venv\Lib\site-packages\numpy\lib\_function_base_impl.py:4842: UserWarning: Warning: 'partition' will ignore the 'mask' of the MaskedArray.
      arr.partition(
    C:\Users\karan\OneDrive\Desktop\Data Science\Module 3 - Data Science using python\.venv\Lib\site-packages\numpy\lib\_function_base_impl.py:4842: UserWarning: Warning: 'partition' will ignore the 'mask' of the MaskedArray.
      arr.partition(



    
![png](output_8_1.png)
    



    
![png](output_8_2.png)
    



    
![png](output_8_3.png)
    



    
![png](output_8_4.png)
    

Inferences from Data Analysis 
1: Creating the function name with treat_outliers_iqr for handling the ouliers by IQR.
2: Generating a box plots and histograms to gain insights into the distributions.
3: By analyzing the above graphs there were some outliers before applying outlier treatment and the distributions was not symmetric.
4: After handling outliers some elements are normally distributed.

```python
%pip install scikit-learn
```

#### Data Preparation 


```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# Ordinal encoding for 'Education'
label_encoder = LabelEncoder()
df['Education_Encod'] = label_encoder.fit_transform(df['Education'])
print(df['Education_Encod'])
# One-hot encoding for 'Marital_Status' and 'Country'
df_encoded = pd.get_dummies(df[['Education_Encod', 'Marital_Status', 'Country']], drop_first=True)
print(df_encoded)
# Compute correlation matrix
corr_matrix = df_encoded.corr()
# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation: Education, Marital Status, and Country")
plt.tight_layout()
plt.show()
```

    0       1
    1       1
    2       1
    3       1
    4       1
           ..
    2235    3
    2236    1
    2237    1
    2238    1
    2239    3
    Name: Education_Encod, Length: 2226, dtype: int64
          Education_Encod  Marital_Status_Married  Marital_Status_Single  \
    0                   1                   False                  False   
    1                   1                   False                   True   
    2                   1                    True                  False   
    3                   1                   False                  False   
    4                   1                   False                   True   
    ...               ...                     ...                    ...   
    2235                3                   False                  False   
    2236                1                    True                  False   
    2237                1                   False                  False   
    2238                1                    True                  False   
    2239                3                    True                  False   
    
          Marital_Status_Together  Country_CA  Country_GER  Country_IND  \
    0                       False       False        False        False   
    1                       False        True        False        False   
    2                       False       False        False        False   
    3                        True       False        False        False   
    4                       False       False        False        False   
    ...                       ...         ...          ...          ...   
    2235                    False       False        False        False   
    2236                    False       False        False        False   
    2237                    False       False        False        False   
    2238                    False       False        False         True   
    2239                    False        True        False        False   
    
          Country_ME  Country_SA  Country_SP  Country_US  
    0          False       False        True       False  
    1          False       False       False       False  
    2          False       False       False        True  
    3          False       False       False       False  
    4          False       False        True       False  
    ...          ...         ...         ...         ...  
    2235       False       False       False        True  
    2236       False       False        True       False  
    2237       False       False        True       False  
    2238       False       False       False       False  
    2239       False       False       False       False  
    
    [2226 rows x 11 columns]



    
![png](output_12_1.png)
    

Inferences from Data Preparation
1: Applying Ordinal encoding for 'Education' and One-hot encoding for 'Marital_Status' and 'Country'
2: Generating a heatmap to illustrate the correlation between different pairs of variables.
3: This will be helpful for further to train models in machine learningl.
#### Testing hypothesis
1: Older individuals may not possess the same level of technological proficiency and may, therefore, lean toward traditional in-store shopping preferences.
Hypothesis Test: In-Store Purchases by Age Group
Null Hypothesis (Ho): There is no difference in the mean number of store purchases between older (60+) and younger (<60) individuals.
Alternative Hypothesis (H1): There is a difference in the mean number of store purchases between the two age groups.

```python
import scipy.stats as stats
older_web_group=df[df['Age']>=60]['NumWebPurchases']
younger_web_group=df[df['Age']<60]['NumWebPurchases']
print(older_web_group,younger_web_group)
t_stat_web, p_value_web = stats.ttest_ind(older_web_group, younger_web_group, equal_var=False)
print("T-statistic (Web Purchases):", t_stat_web)
print("P-value:", p_value_web)
alpha = 0.05
if p_value_web < alpha:
    print("Result: Statistically significant difference in web purchases between older and younger individuals.")
else:
    print("Result: No statistically significant difference in web purchases between the two age groups.")
```

    1        7
    2        3
    5        4
    6       10
    8        6
            ..
    2201    10
    2202     2
    2216     4
    2217     4
    2227     5
    Name: NumWebPurchases, Length: 856, dtype: int64 0       4
    3       1
    4       3
    7       2
    11      3
           ..
    2235    5
    2236    1
    2237    6
    2238    5
    2239    8
    Name: NumWebPurchases, Length: 1370, dtype: int64
    T-statistic (Web Purchases): 5.797758167762514
    P-value: 7.918810798495903e-09
    Result: Statistically significant difference in web purchases between older and younger individuals.

Inference from above hypothesis testing
1: Older Customers Prefer In-Store Shopping
2: There is a clear trend showing that older age groups make more in-store purchases.
3: This supports the idea that older individuals may be less tech-savvy and prefer traditional shopping.2: Customers with children likely experience time constraints, making online
shopping a more convenient option
Hypothesis to Test:
Null Hypothesis (Ho): There is no difference in online purchases between customers with and without children.
Alternative Hypothesis (H1): Customers with children make more online purchases on average.

```python
df['Has_Children'] = df['Total_Children'] > 0
print(df['Has_Children'])
web_purchases_with_kids = df[df['Has_Children'] == True]['NumWebPurchases']
web_purchases_without_kids = df[df['Has_Children'] == False]['NumWebPurchases']
print(web_purchases_with_kids,web_purchases_without_kids)
t_stat_web, p_value_web = stats.ttest_ind(web_purchases_with_kids, web_purchases_without_kids, equal_var=False)
print("T-statistic (Web Purchases):", t_stat_web)
print("P-value:", p_value_web)
alpha = 0.05
if p_value_web < alpha:
    print("Result: Statistically significant difference - Customers with children make more online purchases on average..")
else:
    print("Result: No statistically significant difference - having children does not significantly affect online shopping frequency")
```

    0       False
    1       False
    2        True
    3        True
    4        True
            ...  
    2235     True
    2236     True
    2237     True
    2238    False
    2239     True
    Name: Has_Children, Length: 2226, dtype: bool
    2       3
    3       1
    4       3
    7       2
    8       6
           ..
    2234    9
    2235    5
    2236    1
    2237    6
    2239    8
    Name: NumWebPurchases, Length: 1596, dtype: int64 0        4
    1        7
    5        4
    6       10
    10       5
            ..
    2220     6
    2226     2
    2229     1
    2230     1
    2238     5
    Name: NumWebPurchases, Length: 630, dtype: int64
    T-statistic (Web Purchases): -3.6869215865310174
    P-value: 0.00023609997153243851
    Result: Statistically significant difference - Customers with children make more online purchases on average..

Inference from above hypothesis testing
1: Customers with children makes more online payments on average.
2: Contrary to expectations, customers with children make more online purchases than those with kids.
3: This suggests time constraints may be the dominant factor driving online shopping.3: Sales at physical stores may face the risk of cannibalization by alternative
distribution channels.
Hypothesis to Test:
Null Hypothesis (Ho): There is no negative relationship between online purchases and store purchases.
Alternative Hypothesis (H1): There is a negative relationship

```python
correlation,p_value=stats.pearsonr(df['NumWebPurchases'],df['NumStorePurchases'])
print("Pearson correlation coefficient:", correlation)
print("P-value:", p_value)
alpha=0.05
if p_value<=alpha:
    if correlation<0:
        print("Result: Statistically significant negative correlation. Online purchases may be cannibalizing in-store sales.")
    else:
        print("Result: Statistically significant positive correlation. No evidence of cannibalization.")
else:
    print("Result: No statistically significant relationship between Online purchases and in store")
```

    Pearson correlation coefficient: 0.4991537684627939
    P-value: 1.3859926786824851e-140
    Result: Statistically significant positive correlation. No evidence of cannibalization.

Inference from above hypothesis testing
1: Moderate positive correlations exist between store, web.
2: Customers who shop more in one channel often shop more in others too—indicating multi-channel engagement rather than competition.4: Does the United States significantly outperform the rest of the world in
total purchase volumes?
Hypothesis Setup:
Ho (Null Hypothesis): There is no significant difference in total purchase volumes between the U.S. and the rest of the world.
H1 (Alternative Hypothesis): The U.S. has significantly higher total purchase volumes than the rest of the world.

```python
Us_purchase=df[df['Country']=='US']['Total_Purchases']
Other_purchase=df[df['Country']!='US']['Total_Purchases']
t_stat, p_value = stats.ttest_ind(Us_purchase, Other_purchase, equal_var=False)
print("T-statistic:", t_stat)
print("P-value:", p_value)
print(Us_purchase.mean())
print(Other_purchase.mean())
if p_value < alpha:
    print("Result: Statistically significant difference — U.S. customers have different total purchase volumes.")
    if Us_purchase.mean() > Other_purchase.mean():
        print("U.S. outperforms the rest of the world in purchase volume.")
    else:
        print("U.S. underperforms compared to the rest of the world.")
else:
    print("Result: No statistically significant difference in purchase volumes between U.S. and other countries.")
```

    T-statistic: 1.2667417917830992
    P-value: 0.20772490174808025
    13.37037037037037
    12.495750708215297
    Result: No statistically significant difference in purchase volumes between U.S. and other countries.

Inference from above hypothesis testing
1: United States significantly does not outperform the rest of the world in total purchase volumes.
2: The purchases between different countries has no statistically significant difference in purchase volumes.
#### Data Visualization
a. Identify the top-performing products and those with the lowest revenue.

```python
import matplotlib.pyplot as plt
import seaborn as sns
product_revenue = {
    'Wines': df['MntWines'].sum(),
    'Fruits': df['MntFruits'].sum(),
    'Meat': df['MntMeatProducts'].sum(),
    'Fish': df['MntFishProducts'].sum(),
    'Sweets': df['MntSweetProducts'].sum(),
    'Gold': df['MntGoldProds'].sum(),
}
revenue_df = pd.DataFrame(product_revenue.items(), columns=['Product', 'TotalRevenue'])
revenue_df = revenue_df.sort_values(by='TotalRevenue', ascending=False)
print(revenue_df)
plt.figure(figsize=(10, 6))
sns.barplot(data=revenue_df, x='Product', y='TotalRevenue')
plt.title('Revenue by Product Category')
plt.ylabel('Total Revenue')
plt.xlabel('Product')
plt.show()
```

      Product  TotalRevenue
    0   Wines        676255
    2    Meat        365773
    5    Gold         98103
    3    Fish         83620
    4  Sweets         60107
    1  Fruits         58319



    
![png](output_29_1.png)
    

Inference from above Visualization
1: Wines are mostly purchased and has highest revenue on the other hand fruits are least selling product with lower revenue.b. Examine if there is a correlation between customers' age and the
acceptance rate of the last campaign.

```python
Relationship=df[['Age','Response']]
corr_matrix=Relationship.corr()
print(corr_matrix)
sns.heatmap(corr_matrix,annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title('correlation between customers age and theacceptance rate of the last campaign. ')
plt.show()
```

                  Age  Response
    Age       1.00000  -0.01558
    Response -0.01558   1.00000



    
![png](output_32_1.png)
    


Inference from above Visualization
1: There is no correlation between Age and Response
c. Determine the country with the highest number of customers who
accepted the last campaign.

```python
NofCustomer=df.groupby('Country')['Response'].sum()
Highest_customer=pd.DataFrame(NofCustomer,columns=['Response'])
Highest_customer = Highest_customer.sort_values(by='Response', ascending=False)
print(Highest_customer)
plt.figure(figsize=(10, 6))
sns.barplot(data=Highest_customer, x='Country', y='Response')
plt.title('Highest number of customers who accepted the last campaign')
plt.ylabel('Response')
plt.xlabel('Country')
plt.show()
```

             Response
    Country          
    SP            176
    SA             51
    CA             37
    AUS            23
    GER            17
    IND            13
    US             13
    ME              2



    
![png](output_35_1.png)
    

Inference from above Visualization
1: Country spain has the highest number of response who accepted the last campaign and the Country Montenegro has the least number of response who accepted the last campaign.d. Investigate if there is a discernible pattern in the number of children at
home and the total expenditure.

```python
grouped=df.groupby('Total_Children')['Total_Spending'].sum()
grouped=pd.DataFrame(grouped,columns=['Total_Spending'])
print(grouped)
plt.figure(figsize=(8, 5))
sns.barplot(data=grouped, x='Total_Children', y='Total_Spending', palette='coolwarm')
plt.title('Average Total Expenditure vs. Number of Children at Home')
plt.xlabel('Number of Children at Home')
plt.ylabel('Average Total Expenditure')
plt.show()
```

                    Total_Spending
    Total_Children                
    0                       692759
    1                       531427
    2                       103437
    3                        14554


    C:\Users\karan\AppData\Local\Temp\ipykernel_16776\1756766859.py:5: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(data=grouped, x='Total_Children', y='Total_Spending', palette='coolwarm')



    
![png](output_38_2.png)
    

Inference from above Visualization
1: On average, at home with no childrens has highest total expenditure..e. Analyze the educational background of customers who lodged complaints
in the last two years.

```python
complaint=df.groupby('Education')['Complain'].sum()
complaint=pd.DataFrame(complaint)
complaint = complaint.sort_values(by='Complain', ascending=False)
print(complaint)
plt.figure(figsize=(8, 5))
sns.barplot(data=complaint, x='Education', y='Complain', palette='coolwarm')
plt.title('Educational background of customers who lodged complaints in the last two years')
plt.xlabel('Education')
plt.ylabel('Complain')
plt.show()
```

                Complain
    Education           
    Graduation        17
    Master             2
    PhD                1
    Basic              0


    C:\Users\karan\AppData\Local\Temp\ipykernel_15252\2160802928.py:6: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(data=complaint, x='Education', y='Complain', palette='coolwarm')



    
![png](output_41_2.png)
    

Inference from above Visualization
1: Customers whose education background is Graduation has highest complaints who lodged
in the last two years.

```python

```
