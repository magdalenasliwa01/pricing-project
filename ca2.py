#!/usr/bin/env python
# coding: utf-8

# In[204]:

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:46:59 2021

@author: MagdalenaSliwa
"""

#Importing libraries

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#https://appdividend.com/2020/01/31/python-unzip-how-to-extract-single-or-multiple-files/ author: Krunal
#Unzip the files
from zipfile import ZipFile
with ZipFile('CA2.zip', 'r') as zipObj:
# Extract all the contents of zip file in current directory
   zipObj.extractall()

# In[205]:


# Read pricing.xls into a dataframe
prices = pd.read_excel('Pricing.xlsx')


# In[206]:


#Overview of the dataset
prices.info()

prices.head()

prices.shape

#Checking is the data set is a DataFrame
df = pd.DataFrame(prices)
print(type(df))

# In[207]:
#Step 1. Data preprocessing

#DataFrame subsetting to see prices in GBP only
prices_uk = prices[prices['Country'] == 'UK']

# In[208]:
#Printing out the first 5 rows of data in my dataset
prices_uk.head()
#Checking number of rows and columns of the new dataset
prices_uk.shape

# In[209]:
#Filtering columns + removing missing values from it
new_prices_uk = prices_uk[['Product','LP USD', 'Discount %']].dropna()
print(type(new_prices_uk))

# In[210]:
#Overview of a substracted dataset
print(new_prices_uk.info())
new_prices_uk.head()
# In[211]:
#Removing decimal points to 2
#https://stackoverflow.com/questions/37084812/how-to-remove-decimal-points-in-pandas
pd.set_option('precision', 2)

# In[212]:
#Renaming some columns
new_prices_uk = new_prices_uk.rename(columns = {'LP USD': 'List_Price_UK', 'Discount %': 'Discount_%_UK'})
new_prices_uk.head()

# In[213]:
#Data Analysis
#Insight 1:

#Calculating weighted average ratings
print(new_prices_uk.describe())

# In[214]:

list_prices = new_prices_uk['List_Price_UK']
print(list_prices)

# In[215]:
#Calculating mean, mode, median and skewness of the list prices

print('Mean of the List prices is set to: ' + str(list_prices.mean()))
print('Mode of the List prices is set to: ' + str(list_prices.mode()))
print('Median of the List prices is set to: ' + str(list_prices.median()))
print('The skewness of the list prices is set to: ' + str(list_prices.skew()))

# In[216]:
# 1st insight: What is the average price discount to the clients? How many clients receive more or les than average discount rate?

#Calculating average price discount rate
discount_rate = new_prices_uk['Discount_%_UK']
print(discount_rate)

# In[217]:

#Converting a float to percentage value
discount_average = "{:.0%}".format(discount_rate.mean())
print("Average price discount is estimated to be " + str(discount_average) + " .")

# In[218]:

#Creating a function to convert a float to percentage value with none decinemal places
def percentage(value):
    if type(value) is float:
        val_to_int = round(value,2) * 100
        return int(val_to_int)
    
print(percentage(0.415211))
print("Average price discount is estimated to be " + str(percentage(discount_rate.mean)))

# In[219]:

#Calculating outliers for a Discount % column
#Step 1: Sorting the 'Discount %' column values in ascending order

discount_sorted = new_prices_uk.sort_values(["Discount_%_UK"], ascending = True)
print(discount_sorted)

# In[220]:

#Step 2: Calculating IQR value
discount = discount_sorted['Discount_%_UK']
q1 = discount.quantile(0.25)
q3 = discount.quantile(0.75)
iqr = q3 - q1
print(iqr)

#https://www.kite.com/python/answers/how-to-format-a-number-as-a-percentage-in-python
#Converting a float to percentage value
iqr_percentage = "{:.0%}".format(iqr)
print("The IQR value is equal to " + str(iqr_percentage) + " .")

# In[221]:

#https://medium.datadriveninvestor.com/finding-outliers-in-dataset-using-python-efc3fce6ce32 Renu Khandelwal
#Step 3: Calculating lower and upper bounds
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 +(1.5* iqr)

#Converting a float to percentage value
lower_bound_percentage = "{:.0%}".format(lower_bound)

upper_bound_percentage = "{:.0%}".format(upper_bound)
print("The range excluding outliers lies between " + str(lower_bound_percentage) +
      " and " + str(upper_bound_percentage) + " .")

#Step 4: Calculating discount outliers
discount_outliers = discount_sorted[(discount < q1 - 1.5 * iqr) | (discount > q3 + 1.5 * iqr)]
print(discount_outliers)

# In[222]:

#Step 5: Calculating number of outliers in the Discount % column
print("Outliers include all the values lower than " + str(lower_bound_percentage) +
      " and higher than  " + str(upper_bound_percentage) +" and the total number of outliers is equal to : "
      + str(discount_outliers.shape[0] + int(1)) + " .")

# In[223]:

#Drawing box plot displaying outliers 
discount_sorted[['Product', 'Discount_%_UK']].plot(kind = 'box',
                                                   title = "Discount statistics for the UK market",
                                                   color = '#00CED1',
                                                  )
plt.ylabel("Discount rate (%)")
plt.grid(axis = 'y', alpha = 0.75)
plt.show()

# In[224]:

#Drawing histogram to illustrate the frequency of discounts to the list prices in GBP
#https://realpython.com/python-histograms/ Brad Solomon
new_prices_uk['Discount_%_UK'].plot(kind = 'hist',
                                 bins = 15,
                                 rwidth = 0.9,
                                 color = '#00CED1',
                                 title = "Frequency of discounts for list prices in the UK ",
                                 yticks = [0, 1000, 2000, 3000, 4000, 5000, 6000],
                                 xticks = [0.15, 0.2, 0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.60,0.65,0.70,0.75,0.80],
                                 fontsize = 8,
                                )

plt.xlabel("Discount %")
plt.ylabel("Frequency")
plt.grid(axis = 'y', alpha = 0.75)
plt.show()

# In[225]:

#Insight 3: Is there any correlation between the discount rates for customers in the United Kingdom, 
#France or Germany and the prices of products?

#Defining a dictionary containing pricing data

variables = {
    'Countries' : ['UK', 'France', 'Germany'],
    'Product' : ['Product A', 'Product B', 'Product C', 
                 'Product D', 'Product E','Product F', 
                 'Product G','Product H','Product I',
                 'Product J','Product K','Product L',
                 'Product M','Product N','Product O','Product P'],
    'Currency' : ['EUR', 'GBP']
        }

# In[227]:

products = variables['Product']
print(type(products))


# In[228]:

#Calculating the number of elements in the 'Product' key
print(len(variables['Product']))

# In[229]:

#DataFrame subsetting to see the prices and discounts for each of 16 products for the UK (GBP)
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html

scatterplot_dataset_1 = new_prices_uk.drop_duplicates(subset=['Product'])
print(scatterplot_dataset_1)

# In[230]:

#Drawing scatterplot
#Preprocessing and filtering columns

scatterplot = scatterplot_dataset_1[['List_Price_UK', 'Discount_%_UK']]
print(scatterplot)

# In[231]:

scatterplot.plot(kind = 'scatter',
                 x = 'List_Price_UK',
                 y = 'Discount_%_UK', 
                 title = "List Prices ($) vs Discount rates for the UK customers",
                 color = '#14AAF5',
                 yticks = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                 xticks = [0,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400],
                 fontsize = 8,
                 edgecolor = 'black',
                 linewidth = 1,
                 alpha = 0.75
                )

plt.grid(axis = 'y', alpha = 0.75)
plt.show()

# In[232]:

#DataFrames preprocessing
#DataFrame subsetting to see prices for French customers
prices_france = prices[prices['Country'] == 'France']
print(prices_france)

# In[233]:

#Printing out the first 5 rows of data in my dataset
prices_france.head()
#Checking number of rows and columns of the new dataset
prices_france.shape  

# In[234]:

#Filtering columns + removing missing values from it
new_prices_france = prices_france[['Product','LP USD', 'Discount %']].dropna()
print(type(new_prices_france))
print(new_prices_france)

# In[235]:

#Renaming one column
new_prices_france = new_prices_france.rename(columns = {'LP USD': 'List_Price_$', 'Discount %': 'Discount_%'})
new_prices_france.head()

# In[236]:

#DataFrame subsetting to see prices for German customers
prices_germany = prices[prices['Country'] == 'Germany']
print(prices_germany)

# In[237]:

#Printing out the first 5 rows of data in my dataset
prices_germany.head()
#Checking number of rows and columns of the new dataset
prices_germany.shape 

# In[238]:

#Filtering columns
new_prices_germany = prices_germany[['Product','LP USD', 'Discount %']]
print(type(new_prices_germany))
print(new_prices_germany)

# In[239]:

#Renaming one column
new_prices_germany = new_prices_germany.rename(columns = {'List_Price_$ss' : 'List_Price_$', 'Discount %': 'Discount_%'})
new_prices_germany.head()

# In[240]:

#Merging two dataframes by rows into one indicating customers in Germany and France who pay in EUR currency
prices_eur = pd.concat([new_prices_france, new_prices_germany], axis = 0)
print(prices_eur)

# In[241]:

#DataFrame subsetting to see the prices and discounts for each of 16 products for the Germany and France (EUR)
scatterplot_dataset_2 = prices_eur.drop_duplicates(subset=['Product'])
print(scatterplot_dataset_2)

# In[242]:

#Renaming two columns
scatterplot_dataset_2 = scatterplot_dataset_2.rename(columns = {'List_Price_$': 'List_Price_EUR', 'Discount_%': 'Discount_%_EUR'})
scatterplot_dataset_2.head()

# In[243]:

#Drawing second scatterplot
#Preprocessing and filtering columns

scatterplot_2 = scatterplot_dataset_2[['List_Price_EUR', 'Discount_%_EUR']]
print(scatterplot_2)

# In[244]:

scatterplot_2.plot(kind = 'scatter',
                 x = 'List_Price_EUR',
                 y = 'Discount_%_EUR', 
                 title = "List Prices in $ vs Discount rates for the French and German customers",
                 color = '#FF9933',
                 yticks = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                 xticks = [0,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400],
                 fontsize = 8,
                 edgecolor = 'black',
                 linewidth = 1,
                 alpha = 0.75
                )

plt.grid(axis = 'y', alpha = 0.75)
plt.show()
# In[245]:
#Creating multiple scatterplot
#Creating lists from the DataFrames

# In[247]:

#Converting DataFrames into lists using NumPy
l_p_uk = scatterplot_dataset_1[['List_Price_UK']].values
print(l_p_uk)

# In[248]:

d_uk = scatterplot_dataset_1[['Discount_%_UK']].values
print(d_uk)

# In[249]:

#Creating two variables x and y
l_p_eur = scatterplot_dataset_2[['List_Price_EUR']].values
print(l_p_eur)

# In[250]:

d_eur = scatterplot_dataset_2[['Discount_%_EUR']].values
print(d_eur)

# In[251]:

#https://www.youtube.com/watch?v=ugh-NvZB6wA 'Python Basics'
#Drawing multiple scatterplot

plt.scatter(l_p_eur,d_eur, label = 'Prices vs discount for EUR' , 
            color = '#FF9933',
            edgecolor = 'black',
            linewidth = 1,
            alpha = 0.75)
plt.scatter(l_p_uk,d_uk, label = 'Prices vs discount for GBP' ,
            color = '#14AAF5',
            edgecolor = 'black',
            linewidth = 1,
            alpha = 0.75)
plt.xlabel('List Prices ($)')
plt.ylabel('Price Discount (%)')
plt.title('Relation between List Prices ($) from GBP, EUR vs. Price Discounts (%)')
plt. legend()
plt.show()
# In[252]:

#Linear regression within use of SciPy library for first dataset
from scipy import stats
slope, intercept, r_value, p_value, std_error = stats.linregress(scatterplot_dataset_1['List_Price_UK'],
                                                                 scatterplot_dataset_1['Discount_%_UK'])
#Displaying slope, intercept, R value and P value
print("Slope: " + str(slope))
print("Intercept: " + str(intercept))
print("R value: " + str(r_value))
print("P value: " + str(p_value))
# In[253]:

print("The linear regression is equal to : " + "y = " + str(slope) + "x + " +  str(intercept))

# In[254]:

#Linear regression within use of SciPy library for second dataset
from scipy import stats
slope, intercept, r_value, p_value, std_error = stats.linregress(scatterplot_dataset_2['List_Price_EUR'],
                                                                 scatterplot_dataset_2['Discount_%_EUR'])
#Displaying slope, intercept, R value and P value
print("Slope: " + str(slope))
print("Intercept: " + str(intercept))
print("R value: " + str(r_value))
print("P value: " + str(p_value))

print("The linear regression is equal to : " + "y = " + str(slope) + "x + " +  str(intercept))
# In[255]:
#Showing lists of elements in appriopriate forms
l_p_uk = ([272.15189873, 310.12658228, 322.78481013,373.41772152,487.34177215,981.01265823,
                    1056.96202532,1132.91139241,1234.17721519,1310.12658228,1348.10126582,1411.39240506,
                    1651.89873418,1727.84810127,2069.62025316,2094.93670886])

# In[256]:

#https://www.kite.com/python/answers/how-to-round-all-elements-of-a-list-of-numbers-in-python
#Converting elements values into whole numbers
l_p_uk_whole = [round(num) for num in l_p_uk]
print(l_p_uk_whole)

# In[257]:

# Convert to numpy array
l_p_uk_whole = np.array(l_p_uk_whole)
print(type(l_p_uk_whole))


# In[258]:
d_uk = ([0.52923123,0.69498256,0.36984105, 0.59677662,0.56193821,0.46684511,
         0.54629452,0.62084195,0.8,0.72465483,0.66534836,0.66984094,0.60525563,0.73399981,0.57314008,0.8])
# In[259]:

l_p_eur = ([302.20093511, 336.41236173,359.21997947,416.23902383,530.27711256,
            1077.65993842,1157.48660052,1248.7170715,1362.75516022,1442.58182233,
            1476.79324895,1556.61991105,1818.90751511,
            1898.73417722,2275.05987,2297.86748774])
# In[260]:

#Converting elements into whole numbers
l_p_eur_whole = [round(num) for num in l_p_eur]
print(l_p_eur_whole)

# In[261]:

# Convert to numpy array
l_p_eur_whole = np.array(l_p_eur_whole)
print(type(l_p_eur_whole))
# In[262]:

d_eur = ([0.35159817,0.35146444,0.36679537,0.18,0.30870712,0.18,0.18,0.2,0.58533194,0.2,0.35418672,0.37397634,0.2,
                  0.34503361,0.47467167,0.43051266])

# In[263]:

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()

# In[264]:

l_p_uk_whole = l_p_uk_whole.reshape(-1,1)
linreg.fit(l_p_uk_whole,d_uk)

# In[265]:

d_uk_pred = linreg.predict(l_p_uk_whole)

# In[266]:

l_p_eur_whole = l_p_eur_whole.reshape(-1,1)
linreg.fit(l_p_eur_whole,d_eur)

# In[267]:

d_eur_pred = linreg.predict(l_p_eur_whole)

# In[268]:

plt.plot(l_p_eur_whole, d_eur_pred, color = '#FF9933')
plt.plot(l_p_uk_whole, d_uk_pred, color = '#14AAF5')

#https://www.youtube.com/watch?v=ugh-NvZB6wA 'Python Basics'
#Drawing multiple scatterplot

plt.scatter(l_p_eur,d_eur, label = 'Prices vs discount for EUR' , 
            color = '#FF9933',
            edgecolor = 'black',
            linewidth = 1,
            alpha = 0.75)
plt.scatter(l_p_uk,d_uk, label = 'Prices vs discount for GBP' ,
            color = '#14AAF5',
            edgecolor = 'black',
            linewidth = 1,
            alpha = 0.75)
plt.xlabel('List Prices ($)')
plt.ylabel('Price Discount (%)')
plt.title('Relation between List Prices ($) from GBP, EUR vs. Price Discounts (%)')
plt. legend()

plt.show()
# In[269]:

#Pearson correlation test
#https://www.youtube.com/watch?v=TRNaMGkdn-A  author: stikpet


#Correlation for list prices vs discounts for the UK
scatterplot_dataset_1.corr()

# In[270]:

#Correlation for list prices vs discounts for the France and Germany
scatterplot_dataset_2.corr()

# In[271]:

#Calculating the p-value (significance) for the UK
#Importing pearsonr function from SciPy
from scipy.stats import pearsonr

pearson = pearsonr(scatterplot_dataset_1['List_Price_UK'], scatterplot_dataset_1['Discount_%_UK'])
# In[272]:

print('The p-value is equal to ' + str(pearson[1]) + '.')

# In[273]:

#Calculating the p-value (significance) for the France and Germany
pearson2 = pearsonr(scatterplot_dataset_2['List_Price_EUR'], scatterplot_dataset_2['Discount_%_EUR'])
print('The p-value is equal to ' + str(pearson2[1]) + '.')

# In[274]:

# Insight 2: 
# Checking three variables for the analysis

#First variable - List Price (EUR) as integers
print(l_p_eur_whole)
# In[275]:

#Second variable - List Price (GBP) as integers
print(l_p_uk_whole)

# In[276]:

#Third variable - Products
products = scatterplot_dataset_1[['Product']].values
print(products)
#Checkign type of the variable
print(type(products))

# In[277]:

#Drawing the line plot

plt.xlabel('List Prices ($)', fontsize = 14)
plt.ylabel('Price Discount (%)', fontsize = 14)
plt.title('List Prices in $ converted from EUR & GBP for the 16 products in 2021.')
#Rotation to the bars names

#Drawing a line chart
plt.plot(scatterplot_dataset_1.Product, scatterplot_dataset_1.List_Price_UK, label = "LP USD from GBP", color = '#FF9933')
plt.plot(scatterplot_dataset_2.Product, scatterplot_dataset_2.List_Price_EUR, label = "LP USD from EUR", color = '#14AAF5')
plt.xticks(rotation = 90)
plt.legend()
plt.show()
# In[279]:

#Insight 4:
#ANOVA test
from scipy.stats import f_oneway

# In[280]:

#Calculating average price discount rate for the UK
print(discount_rate.mean())
#Calculating average price discount rate for Germany
new_prices_germany_disc = prices_germany['Discount %']
print(new_prices_germany_disc.mean())
#Calculating average price discount rate for France
new_prices_france_disc = prices_france['Discount %']
print(new_prices_france_disc.mean())

# In[283]:

#ANOVA analysis
stats.f_oneway(new_prices_france_disc, new_prices_germany_disc, discount_rate)

