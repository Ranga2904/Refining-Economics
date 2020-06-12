#!/usr/bin/env python
# coding: utf-8

# In[288]:


# This code performs analysis and visualization seen in 'Refining Margins Over the Years'


# In[289]:


# (1) Import necessary packages, modules

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import matplotlib.ticker as ticker
import chart_studio.plotly as py
import plotly.graph_objects as pg
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from functools import reduce

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from datetime import datetime


# In[290]:


#  (2) Import necessary files. 
# Crude price is a trend of crude prices, crude rate of total refinery processing, the various prices files represent'
# each product, rates utilization represents refinery utilization (actual / capacity), and the last total crude sulfur and'
# API.

crudeprice = pd.read_csv('Crude Price.csv')
cruderate  = pd.read_csv('Crude Rate for Import.csv')
heatingoilprice = pd.read_csv('Heating Oil Price.csv')
keroprice = pd.read_csv('Kero Price.csv')
RBOBgasprice = pd.read_csv('RBOB Gas Price.csv')
Convgasprice = pd.read_csv('Conventional Gas Price.csv')
ULSDprice = pd.read_csv('ULSD Price.csv')
rates_utilization = pd.read_csv('Refinery rates, utilization.csv')
crudesulfurandAPI = pd.read_csv('Crude API and sulfur.csv')


# In[291]:


# (3) Data cleaning and preparation.


# In[292]:


# (3.a) First, convering columns to numeric to enable analysis.

crudecols = list(crudeprice[['Cushing, OK WTI Spot Price FOB (Dollars per Barrel)', 
                             'Europe Brent Spot Price FOB (Dollars per Barrel)']])

Convgascols = list(Convgasprice[['New York Harbor Conventional Gasoline Regular Spot Price FOB (Dollars per Gallon)', 
                               'U.S. Gulf Coast Conventional Gasoline Regular Spot Price FOB (Dollars per Gallon)']])

ULSDcols = list(ULSDprice[['New York Harbor Ultra-Low Sulfur No 2 Diesel Spot Price (Dollars per Gallon)', 
                            'U.S. Gulf Coast Ultra-Low Sulfur No 2 Diesel Spot Price (Dollars per Gallon)',
                           'Los Angeles, CA Ultra-Low Sulfur CARB Diesel Spot Price (Dollars per Gallon)']])

crudesulfurandAPIcols = list(crudesulfurandAPI[['U.S. Sulfur Content (Weighted Average) of Crude Oil Input to Refineries (Percent)', 
                       'U.S. API Gravity (Weighted Average) of Crude Oil Input to Refineries (Degrees)']])


heatingoilprice['New York Harbor No. 2 Heating Oil Spot Price FOB (Dollars per Gallon)'] = pd.to_numeric(
                            heatingoilprice['New York Harbor No. 2 Heating Oil Spot Price FOB (Dollars per Gallon)'])

keroprice['U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB (Dollars per Gallon)'] = pd.to_numeric(
                            keroprice['U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB (Dollars per Gallon)'])

RBOBgasprice['Los Angeles Reformulated RBOB Regular Gasoline Spot Price (Dollars per Gallon)'] = pd.to_numeric(RBOBgasprice['Los Angeles Reformulated RBOB Regular Gasoline Spot Price (Dollars per Gallon)'])


crudeprice[crudecols] = crudeprice[crudecols].apply(pd.to_numeric)
Convgasprice[Convgascols] = Convgasprice[Convgascols].apply(pd.to_numeric)
ULSDprice[ULSDcols] = ULSDprice[ULSDcols].apply(pd.to_numeric)
crudesulfurandAPI[crudesulfurandAPIcols] = crudesulfurandAPI[crudesulfurandAPIcols].apply(pd.to_numeric)


# In[293]:


# (3.b) Dropping columns with all null values, not needed for analysis.

crudeprice.drop(columns='Unnamed: 3',inplace=True)
crudeprice.drop(crudeprice.tail(1).index,inplace=True) 
heatingoilprice.drop(columns='Unnamed: 2',inplace=True)
heatingoilprice.drop(heatingoilprice.tail(1).index,inplace=True) 
keroprice.drop(columns='Unnamed: 2',inplace=True)
keroprice.drop(keroprice.tail(1).index,inplace=True)
RBOBgasprice.drop(columns='Unnamed: 2',inplace=True)
RBOBgasprice.drop(RBOBgasprice.tail(1).index,inplace=True)
Convgasprice.drop(columns='Unnamed: 3',inplace=True)
Convgasprice.drop(Convgasprice.tail(1).index,inplace=True)
ULSDprice.drop(columns='Unnamed: 4',inplace=True)
ULSDprice.drop(ULSDprice.tail(1).index,inplace=True)
crudesulfurandAPI.drop(columns='Unnamed: 3',inplace=True)
crudesulfurandAPI.drop(crudesulfurandAPI.tail(1).index,inplace=True)


# In[294]:


# (3.c) Replace existing null values (absent price or rate information) with 0

crudeprice.replace(np.NaN,0,inplace=True)
heatingoilprice.replace(np.NaN,0,inplace=True)
keroprice.replace(np.NaN,0,inplace=True)
RBOBgasprice.replace(np.NaN,0,inplace=True)
Convgasprice.replace(np.NaN,0,inplace=True)
ULSDprice.replace(np.NaN,0,inplace=True)


# In[295]:


# (3.d) Observation on data frequency mismatch.

# Data is largely available since 1996, but not at the same frequency. All prices are daily, but price data will be
# adjusted to show monthly rolling averages. Monthly data for prices then matches other datasets.


# In[296]:


# In addition to creating new columns with rolling monthly averages of daily price data, the code below removes daily data 
# and renames columns for brevity.

crudeprice['WTI mo. avg, $/bbl'] = crudeprice['Cushing, OK WTI Spot Price FOB (Dollars per Barrel)'].rolling(
                                    window=30).mean()
crudeprice['Brent mo. avg, $/bbl'] = crudeprice['Europe Brent Spot Price FOB (Dollars per Barrel)'].rolling(
                                    window=30).mean()

crudeprice.drop(columns=['Cushing, OK WTI Spot Price FOB (Dollars per Barrel)',
                         'Europe Brent Spot Price FOB (Dollars per Barrel)'],inplace=True)

heatingoilprice['NY #2 HO mo. avg, $/gal'] = heatingoilprice['New York Harbor No. 2 Heating Oil Spot Price FOB '''
                                                             '(Dollars per Gallon)'].rolling(window=30).mean()

heatingoilprice.drop(columns='New York Harbor No. 2 Heating Oil Spot Price FOB (Dollars per Gallon)', inplace=True)

keroprice['Gulf Coast Jet mo. avg, $/gal'] = keroprice['U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB '''
                                                       '(Dollars per Gallon)'].rolling(window=30).mean()

keroprice.drop(columns='U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB (Dollars per Gallon)', inplace=True)

RBOBgasprice['LA RBOB gas mo. avg, $/gal'] = RBOBgasprice['Los Angeles Reformulated RBOB Regular Gasoline Spot Price ''' 
                                                          '(Dollars per Gallon)'].rolling(window=30).mean()


RBOBgasprice.drop(columns='Los Angeles Reformulated RBOB Regular Gasoline Spot Price (Dollars per Gallon)', inplace=True)

Convgasprice['NY Conv Gas mo. avg, $/gal'] = Convgasprice['New York Harbor Conventional Gasoline Regular '''
                                                'Spot Price FOB (Dollars per Gallon)'].rolling(window=30).mean()
Convgasprice['Gulf Coast Gas mo. avg, $/gal'] = Convgasprice['U.S. Gulf Coast Conventional Gasoline Regular '''
                                                'Spot Price FOB (Dollars per Gallon)'].rolling(window=30).mean()

Convgasprice.drop(columns=['New York Harbor Conventional Gasoline Regular Spot Price FOB (Dollars per Gallon)',
                          'U.S. Gulf Coast Conventional Gasoline Regular Spot Price FOB (Dollars per Gallon)'],
                          inplace=True)

ULSDprice['NY ULSD #2, $/gal'] = ULSDprice['New York Harbor Ultra-Low Sulfur No 2 Diesel Spot Price '''
                                           '(Dollars per Gallon)'].rolling(window=30).mean()
ULSDprice['Gulf Coast ULSD #2, $/gal'] = ULSDprice['U.S. Gulf Coast Ultra-Low Sulfur No 2 Diesel Spot Price '''
                                                   '(Dollars per Gallon)'].rolling(window=30).mean()
ULSDprice['LA ULSD #2, $/gal'] = ULSDprice['Los Angeles, CA Ultra-Low Sulfur CARB Diesel Spot Price '''
                                           '(Dollars per Gallon)'].rolling(window=30).mean()

ULSDprice.drop(columns=['New York Harbor Ultra-Low Sulfur No 2 Diesel Spot Price (Dollars per Gallon)',
                       'U.S. Gulf Coast Ultra-Low Sulfur No 2 Diesel Spot Price (Dollars per Gallon)',
                       'Los Angeles, CA Ultra-Low Sulfur CARB Diesel Spot Price (Dollars per Gallon)'], inplace=True)


# In[297]:


# Having created columns with rolling averages, we only want data at the end of each month i.e. an average over the entire
# month. First block of code below isolates rows that aren't at the end of the month, second block deletes those rows.

# Get names of indexes for which column Date doesn't have 30 or 31.

crudeindexNames = crudeprice[crudeprice['Date'].str.contains('30','31')==False].index
hoindexNames = heatingoilprice[heatingoilprice['Date'].str.contains('30','31')==False].index
keroindexNames = keroprice[keroprice['Date'].str.contains('30','31')==False].index
RBOBgasindexNames = RBOBgasprice[RBOBgasprice['Date'].str.contains('30','31')==False].index
ConvgasindexNames = Convgasprice[Convgasprice['Date'].str.contains('30','31')==False].index
ULSDindexNames = ULSDprice[ULSDprice['Date'].str.contains('30','31')==False].index
 
    
# Delete these row indexes from the relevant columns, since they're redundant from above feature engineering.
crudeprice.drop(crudeindexNames , inplace=True)
heatingoilprice.drop(hoindexNames, inplace=True)
keroprice.drop(keroindexNames, inplace=True)
RBOBgasprice.drop(RBOBgasindexNames, inplace=True)
Convgasprice.drop(ConvgasindexNames, inplace=True)
ULSDprice.drop(ULSDindexNames, inplace=True)


# In[298]:


# 4) With all dataframes reading data on a constant basis - month - we now merge all columsn on Date.

combined = [crudeprice, heatingoilprice, keroprice, RBOBgasprice, Convgasprice, ULSDprice]

AllData = reduce(lambda  left,right: pd.merge(left,right,on=['Date'],
                                            how='outer'), combined)

AllData.replace(np.NaN,0,inplace=True)


# In[299]:


# For loops below separate 'Date' column by comma and create 'New2' column that has month and day. This is mapped to 
# remove any mention of the day.

AllData['New'] = AllData['Date']
AllData['New2'] = AllData['Date']

for i in range(0,len(AllData['New'])):
    AllData.loc[i,'New'] = AllData.loc[i,'Date'].split(',')[1]
for i in range(0,len(AllData['Date'])):    
    AllData.loc[i,'New2'] = AllData.loc[i,'Date'].split(',')[0]

replace_dates = {'Jan 30' : 'Jan', 'Feb 30' : 'Feb', 'Mar 30' : 'Mar','Apr 30' : 'Apr', 'May 30' : 'May', 'Jun 30' : 'Jun',
                 'Jul 30' : 'Jul', 'Aug 30' : 'Aug', 'Sep 30' : 'Sept', 'Oct 30' : 'Oct', 'Nov 30' : 'Nov', 'Dec 30' : 'Dec'}   

AllData['New2'] = AllData['New2'].map(replace_dates)

# Date columns is now equal to Month (from above mapping) + Year

# AllData.drop(columns='Date',inplace=True)

AllData['Date'] = AllData['New'] + ' ' + AllData['New2']
AllData.drop(columns=['New','New2'])


# In[300]:


crudesulfurandAPI['1st'] = crudesulfurandAPI['Date']   
crudesulfurandAPI['2nd'] = crudesulfurandAPI['Date']

for i in range(0,len(crudesulfurandAPI['Date'])):
    crudesulfurandAPI.loc[i,'1st'] = crudesulfurandAPI.loc[i,'Date'].split("-")[0]
    crudesulfurandAPI.loc[i,'2nd'] = crudesulfurandAPI.loc[i,'Date'].split("-")[1]

crudesulfurandAPI["Date"] = crudesulfurandAPI['2nd'] + " " + crudesulfurandAPI['1st']


# In[301]:


# Dropping columns now redundant due to feature engineering.

crudesulfurandAPI.drop(columns=['1st','2nd'], inplace=True)
AllData.drop(columns=['New','New2'], inplace=True)


# In[302]:


# 4.b) Visualiation -- how have crude prices trended over the years? Do we see differences between WTI and Brent?

WTI_prices = pg.Scatter(x=AllData['Date'], y=AllData['WTI mo. avg, $/bbl'], name = 'WTI,$/bbl')


Brent_prices = pg.Scatter(x=AllData['Date'], y=AllData['Brent mo. avg, $/bbl'],
                        # Specify axis
                        yaxis='y2', name = 'Brent,$/bbl')

layout = pg.Layout(height=600, width=1000,
                   title='Crude prices since 1986',
                   # Same x and first y
                   xaxis=dict(title='Date'),
                   yaxis=dict(title='WTI price', color='blue'),
                   # Add a second yaxis to the right of the plot
                   yaxis2=dict(title='Brent price', color='red',
                               overlaying='y', side='right'),
                   
                   )
fig = pg.Figure(data=[WTI_prices, Brent_prices], layout=layout)

fig.add_annotation(
            x=200,
            y=125,
    font=dict(
            family="Courier New, monospace",
            size=18,
            color="black",
            ),
            text="Divergence in prices")


fig.update_layout(legend=dict(x=-0, y=0.5))
fig


# In[303]:


#Brent and WTI largely track each other; we observe divergence starting in 2011 Nov through 2014 May specifically, with 
#Brent. 
# What happened here? Likely the rise of crude from shale that drove WTI prices to drop below Brent.


# In[304]:


# Since ULSD and gas prices are available only after 2006, let's restrict the focus of our analysis to that time frame. 
# Given that crude prices in the last 13 years haven't changed much, that provides a good window.

Post2006 = AllData[AllData['Gulf Coast ULSD #2, $/gal']>0]


# In[305]:


Post2006.reset_index(inplace=True, drop=True)


# In[306]:


# Before calculating crack spreads, do I really need all this data? Can I average some of it? Let's view the statistical
# properties of available data to decide what to include.

Post2006.describe()


# In[307]:


#WTI and Brent are sufficiently different - keep these two separate. Heating oil is its own product, so is Jet.
#Average NY and Gulf Coast Conv Gas, keep RBOB separate as it's a separate grade of gasoline
#Average NY, LA, Gulf Coast ULSD; they are the same grade of product


# In[308]:


Post2006['Conv Gas, $/gal'] = Post2006['NY Conv Gas mo. avg, $/gal']
Post2006['ULSD, $/gal'] = Post2006['NY ULSD #2, $/gal']

Post2006['Conv Gas, $'] = Post2006[['NY Conv Gas mo. avg, $/gal', 'Gulf Coast Gas mo. avg, $/gal']].mean(axis=1)
Post2006['ULSD, $/gal'] = Post2006[['NY ULSD #2, $/gal', 'Gulf Coast ULSD #2, $/gal', 'LA ULSD #2, $/gal']].mean(axis=1)


# After averaging columns above, dropping those unnecessary for analysis.
Post2006.drop(columns=['NY Conv Gas mo. avg, $/gal','Gulf Coast Gas mo. avg, $/gal','LA ULSD #2, $/gal',
                      'Gulf Coast ULSD #2, $/gal', 'NY ULSD #2, $/gal'], inplace=True)


# In[309]:


# Some details of refinery cracks.

# A 3:2:1 crack spread (the most commonly used crack spread for U.S. refining operations) denotes the spread between 
# the cost of buying 3 barrels of crude oil and the revenues from selling 2 barrels of gasoline and 1 barrel of 
# diesel fuel.  Remember to divide answer by 3 barrels.
# Similarly, a 6:3:2:1 crack spread denotes the spread between the cost of buying 6 barrels of crude oil and the revenues 
# from selling 3 barrels of gasoline, 2 barrels of diesel fuel, and 1 barrel of fuel oil or kerosene.


# In[310]:


Post2006['3-2-1, WTI-base'] = Post2006['Date']

Post2006['3-2-1, WTI-base'] = (2*Post2006['Conv Gas, $/gal']*42 + Post2006['ULSD, $/gal']*42- (3*Post2006['WTI mo. avg, $/bbl'])) / 3 


# In[311]:


# fig = plt.subplots(figsize=(8,8))
# fig = Final.plot(kind='scatter', x='Date', y='3-2-1, WTI-base')

plt.figure(figsize=(10,10))

plt.scatter(Post2006['Date'],Post2006['3-2-1, WTI-base'], c='b', marker='x', label='3-2-1 spread, $/bbl')
# plt.scatter(Post2006['Conv Gas, $/gal'],Post2006['Gulf Coast Jet mo. avg, $/gal'], c='r', marker='s', label='Gulf Coast Jet')
# plt.scatter(Post2006['LA RBOB gas mo. avg, $/gal'],Post2006['ULSD, $/gal'], c='g', marker='x', label='ULSD')

y_mean = [np.mean(Post2006['3-2-1, WTI-base'])]*len(Post2006['Date'])
# mean_line = ax.plot(Post2006['Date'],y_mean, label='Mean', linestyle='--')

plt.plot(Post2006['Date'],y_mean, c='blacK', marker='x', label='Avg')

plt.legend(loc='upper left', fontsize=12)
plt.xticks(np.arange(0, 100, step=12))
plt.xticks(rotation=80, fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# In[312]:


# (5) Trying a seaborn heatmap of correlations

corrmatrix = Post2006.corr()
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(corrmatrix, annot=True)


# In[313]:


# # We's expect fuel prices to be dependent on crude prices, since crude feeds refineries that make fuel products. 
# # Pearson correlations above are all > 90%, with WTI and Brent. Some interesting observations:
#    - Prices show on average 0.02 greater correlation with Brent, than WTI, suggesting that most product price markers are 
# correlated to Brent
#    - Within that, LA RBOB shows perhaps the weakest correlation. This product is specific to the CA market and likely 
# supplied mostly by western refineries, whose crude slate is less dependent on WTI and Brent - more on ANS.
#    - How do gas, diesel, jet prices vary? 


# In[314]:


# (5) We'd expect fuel prices to trend with each other. Analysis below confirms that all motor fuel prices trend each other
# linearly. The first plot below focuses on ULSD and Gulf Jet, while the second seaborn plot looks at all.


# In[315]:


plt.figure(figsize=(10,10))

plt.scatter(Post2006['Conv Gas, $/gal'],Post2006['ULSD, $/gal'], c='b', marker='x', label='ULSD')
plt.scatter(Post2006['Conv Gas, $/gal'],Post2006['Gulf Coast Jet mo. avg, $/gal'], c='r', marker='s', label='Gulf Coast Jet')
# plt.scatter(Post2006['LA RBOB gas mo. avg, $/gal'],Post2006['ULSD, $/gal'], c='g', marker='x', label='ULSD')
plt.legend(loc='upper left')
plt.show()


# In[316]:


g = sns.pairplot(Post2006[['Conv Gas, $/gal','Gulf Coast Jet mo. avg, $/gal','ULSD, $/gal', 'WTI mo. avg, $/bbl']], height=2)


# In[317]:


# Transportation fuel prices and heating oil prices all trend together linearly, the least healthy correlation between
# gasoline and heating oil unsurprisingly.


# In[318]:


# 6) Now shifting focus to the crude sulfur and API dataframe. First, rename columns for simplicity and then plot
# data.

crudesulfurandAPI['Crude sulfur, %'] = crudesulfurandAPI['U.S. Sulfur Content (Weighted Average) of Crude Oil Input to Refineries (Percent)']
crudesulfurandAPI['API'] = crudesulfurandAPI['U.S. API Gravity (Weighted Average) of Crude Oil Input to Refineries (Degrees)']
crudesulfurandAPI.drop(columns=['U.S. Sulfur Content (Weighted Average) of Crude Oil Input to Refineries (Percent)','U.S. API Gravity (Weighted Average) of Crude Oil Input to Refineries (Degrees)'], inplace=True)


# In[320]:


API = pg.Scatter(x=crudesulfurandAPI['Date'], y=crudesulfurandAPI['Crude sulfur, %'], name = 'Sulfur, %')


Sulfur = pg.Scatter(x=crudesulfurandAPI['Date'], y=crudesulfurandAPI['API'],
                        # Specify axis
                        yaxis='y2', name = 'API, deg F')

layout = pg.Layout(height=500, width=900,
                   title='Crude sulfur and API since 1985',
                   # Same x and first y
                   xaxis=dict(title='Date'),
                   yaxis=dict(title='Sulfur, %', color='blue'),
                   # Add a second yaxis to the right of the plot
                   yaxis2=dict(title='API, deg F', color='red',
                               overlaying='y', side='right'),
                   )


fig = pg.Figure(data=[API, Sulfur], layout=layout)

fig.add_annotation(
            x=200,
            y=1.45,
    font=dict(
            family="Courier New, monospace",
            size=18,
            color="black",
            ),
            text="29.78 API, 1.48% Sulfur")

fig.update_layout(legend=dict(x=0.15, y=0.9), title=dict(x=0.15, y=0.9))


# In[321]:


# Between 1985 and 2007, the U.S. steadily processsed increasingly heavy and sour crude. Zenith of 29 API and 1.46% Sulfur.
# Starting in 2007, crude API shot upwards as light shale crude began to feed U.S. refineries. Surprising that overall 
# sulfur hasn't reduced, as shale crude is also sweet.


# In[322]:


# 7) All code in this next section combines crudesulfurandAPI dataframe with prices data to try and uncover any correlation
# between crack spread and API


# In[323]:


crudesulfurandAPI_post2006 = crudesulfurandAPI[(crudesulfurandAPI.Date >= "2006 June")]
# df_filtered = df[(df.salary >= 30000) & (df.year == 2017)]


# In[324]:


for i in range(0,len(Post2006['Date'])):
    Post2006.loc[i,'Date'] = str(Post2006.loc[i,'Date']).strip()


# In[325]:


Final = crudesulfurandAPI_post2006.append(Post2006)


# In[326]:


Final = Final.groupby(Final['Date']).aggregate({'WTI mo. avg, $/bbl': 'sum','Brent mo. avg, $/bbl': 'sum',
                                               'NY #2 HO mo. avg, $/gal': 'sum','Gulf Coast Jet mo. avg, $/gal': 'sum',
                                                'Conv Gas, $/gal': 'sum', 'ULSD, $/gal': 'sum', '3-2-1, WTI-base': 'sum',
                                                'Crude sulfur, %': 'sum', 'API': 'sum'}).reset_index()


# In[327]:


for i in range(0,len(Final['Date'])):
    Final.loc[i,'Date'] = pd.to_datetime(Final.loc[i,'Date'])


# In[328]:


Final = Final.sort_values(by='Date')


# In[329]:


FinalCorrMatrix = Final.corr()
sns.heatmap(FinalCorrMatrix, annot=True)


# In[330]:


# plab.scatter(Final['Conv Gas, $/gal'], Final['API'])

APIvsGas = pg.Scatter(x=Final['Conv Gas, $/gal'], y=Final['API'])


layout = pg.Layout(height=700, width=1000,
                   title='Gas prices, and crude feed API',
                   # Same x and first y
                   xaxis=dict(title='Conv Gas Price, $/gal'),
                   yaxis=dict(title='Crude feed API', color='red'),
                   )
fig = pg.Figure(data=[APIvsGas], layout=layout)
fig


# In[ ]:


# No real correlation observed above.

