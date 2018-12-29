
<a href="https://cognitiveclass.ai"><img src = "https://ibm.box.com/shared/static/9gegpsmnsoo25ikkbl4qzlvlyjbgxs5x.png" width = 400> </a>

<h1 align=center><font size = 5>Pie Charts, Box Plots, Scatter Plots, and Bubble Plots</font></h1>

## Introduction

In this lab session, we continue exploring the Matplotlib library. More specificatlly, we will learn how to create pie charts, box plots, scatter plots, and bubble charts.

## Table of Contents

<div class="alert alert-block alert-info" style="margin-top: 20px">

1. [Exploring Datasets with *p*andas](#0)<br>
2. [Downloading and Prepping Data](#2)<br>
3. [Visualizing Data using Matplotlib](#4) <br>
4. [Pie Charts](#6) <br>
5. [Box Plots](#8) <br>
6. [Scatter Plots](#10) <br>
7. [Bubble Plots](#12) <br> 
</div>
<hr>

# Exploring Datasets with *pandas* and Matplotlib<a id="0"></a>

Toolkits: The course heavily relies on [*pandas*](http://pandas.pydata.org/) and [**Numpy**](http://www.numpy.org/) for data wrangling, analysis, and visualization. The primary plotting library we will explore in the course is [Matplotlib](http://matplotlib.org/).

Dataset: Immigration to Canada from 1980 to 2013 - [International migration flows to and from selected countries - The 2015 revision](http://www.un.org/en/development/desa/population/migration/data/empirical2/migrationflows.shtml) from United Nation's website.

The dataset contains annual data on the flows of international migrants as recorded by the countries of destination. The data presents both inflows and outflows according to the place of birth, citizenship or place of previous / next residence both for foreigners and nationals. In this lab, we will focus on the Canadian Immigration data.

# Downloading and Prepping Data <a id="2"></a>

Import primary modules.


```python
import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
```

Let's download and import our primary Canadian Immigration dataset using *pandas* `read_excel()` method. Normally, before we can do that, we would need to download a module which *pandas* requires to read in excel files. This module is **xlrd**. For your convenience, we have pre-installed this module, so you would not have to worry about that. Otherwise, you would need to run the following line of code to install the **xlrd** module:
```
!conda install -c anaconda xlrd --yes
```

Download the dataset and read it into a *pandas* dataframe.


```python
df_can = pd.read_excel('https://ibm.box.com/shared/static/lw190pt9zpy5bd1ptyg2aw15awomz9pu.xlsx',
                       sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                       skipfooter=2
                      )

print('Data downloaded and read into a dataframe!')
```

    Data downloaded and read into a dataframe!


Let's take a look at the first five items in our dataset.


```python
df_can.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Coverage</th>
      <th>OdName</th>
      <th>AREA</th>
      <th>AreaName</th>
      <th>REG</th>
      <th>RegName</th>
      <th>DEV</th>
      <th>DevName</th>
      <th>1980</th>
      <th>...</th>
      <th>2004</th>
      <th>2005</th>
      <th>2006</th>
      <th>2007</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Immigrants</td>
      <td>Foreigners</td>
      <td>Afghanistan</td>
      <td>935</td>
      <td>Asia</td>
      <td>5501</td>
      <td>Southern Asia</td>
      <td>902</td>
      <td>Developing regions</td>
      <td>16</td>
      <td>...</td>
      <td>2978</td>
      <td>3436</td>
      <td>3009</td>
      <td>2652</td>
      <td>2111</td>
      <td>1746</td>
      <td>1758</td>
      <td>2203</td>
      <td>2635</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Immigrants</td>
      <td>Foreigners</td>
      <td>Albania</td>
      <td>908</td>
      <td>Europe</td>
      <td>925</td>
      <td>Southern Europe</td>
      <td>901</td>
      <td>Developed regions</td>
      <td>1</td>
      <td>...</td>
      <td>1450</td>
      <td>1223</td>
      <td>856</td>
      <td>702</td>
      <td>560</td>
      <td>716</td>
      <td>561</td>
      <td>539</td>
      <td>620</td>
      <td>603</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Immigrants</td>
      <td>Foreigners</td>
      <td>Algeria</td>
      <td>903</td>
      <td>Africa</td>
      <td>912</td>
      <td>Northern Africa</td>
      <td>902</td>
      <td>Developing regions</td>
      <td>80</td>
      <td>...</td>
      <td>3616</td>
      <td>3626</td>
      <td>4807</td>
      <td>3623</td>
      <td>4005</td>
      <td>5393</td>
      <td>4752</td>
      <td>4325</td>
      <td>3774</td>
      <td>4331</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Immigrants</td>
      <td>Foreigners</td>
      <td>American Samoa</td>
      <td>909</td>
      <td>Oceania</td>
      <td>957</td>
      <td>Polynesia</td>
      <td>902</td>
      <td>Developing regions</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Immigrants</td>
      <td>Foreigners</td>
      <td>Andorra</td>
      <td>908</td>
      <td>Europe</td>
      <td>925</td>
      <td>Southern Europe</td>
      <td>901</td>
      <td>Developed regions</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 43 columns</p>
</div>



Let's find out how many entries there are in our dataset.


```python
# print the dimensions of the dataframe
print(df_can.shape)
```

    (195, 43)


Clean up data. We will make some modifications to the original dataset to make it easier to create our visualizations. Refer to *Introduction to Matplotlib and Line Plots* and *Area Plots, Histograms, and Bar Plots* for a detailed description of this preprocessing.


```python
# clean up the dataset to remove unnecessary columns (eg. REG) 
df_can.drop(['AREA', 'REG', 'DEV', 'Type', 'Coverage'], axis=1, inplace=True)

# let's rename the columns so that they make sense
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace=True)

# for sake of consistency, let's also make all column labels of type string
df_can.columns = list(map(str, df_can.columns))

# set the country name as index - useful for quickly looking up countries using .loc method
df_can.set_index('Country', inplace=True)

# add total column
df_can['Total'] = df_can.sum(axis=1)

# years that we will be using in this lesson - useful for plotting later on
years = list(map(str, range(1980, 2014)))
print('data dimensions:', df_can.shape)
```

    data dimensions: (195, 38)


# Visualizing Data using Matplotlib<a id="4"></a>

Import `Matplotlib`.


```python
%matplotlib inline

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ggplot') # optional: for ggplot-like style

# check for latest version of Matplotlib
print('Matplotlib version: ', mpl.__version__) # >= 2.0.0
```

    Matplotlib version:  2.2.2


# Pie Charts <a id="6"></a>

A `pie chart` is a circualr graphic that displays numeric proportions by dividing a circle (or pie) into proportional slices. You are most likely already familiar with pie charts as it is widely used in business and media. We can create pie charts in Matplotlib by passing in the `kind=pie` keyword.

Let's use a pie chart to explore the proportion (percentage) of new immigrants grouped by continents for the entire time period from 1980 to 2013. 

Step 1: Gather data. 

We will use *pandas* `groupby` method to summarize the immigration data by `Continent`. The general process of `groupby` involves the following steps:

1. **Split:** Splitting the data into groups based on some criteria.
2. **Apply:** Applying a function to each group independently:
       .sum()
       .count()
       .mean() 
       .std() 
       .aggregate()
       .apply()
       .etc..
3. **Combine:** Combining the results into a data structure.

<img src="https://ibm.box.com/shared/static/tkfhxqkehfzpclco8f0eazhie33uxj9j.png" height=400 align="center">


```python
# group countries by continents and apply sum() function 
df_continents = df_can.groupby('Continent', axis=0).sum()

# note: the output of the groupby method is a `groupby' object. 
# we can not use it further until we apply a function (eg .sum())
print(type(df_can.groupby('Continent', axis=0)))

df_continents.head()
```

    <class 'pandas.core.groupby.groupby.DataFrameGroupBy'>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1980</th>
      <th>1981</th>
      <th>1982</th>
      <th>1983</th>
      <th>1984</th>
      <th>1985</th>
      <th>1986</th>
      <th>1987</th>
      <th>1988</th>
      <th>1989</th>
      <th>...</th>
      <th>2005</th>
      <th>2006</th>
      <th>2007</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>Total</th>
    </tr>
    <tr>
      <th>Continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Africa</th>
      <td>3951</td>
      <td>4363</td>
      <td>3819</td>
      <td>2671</td>
      <td>2639</td>
      <td>2650</td>
      <td>3782</td>
      <td>7494</td>
      <td>7552</td>
      <td>9894</td>
      <td>...</td>
      <td>27523</td>
      <td>29188</td>
      <td>28284</td>
      <td>29890</td>
      <td>34534</td>
      <td>40892</td>
      <td>35441</td>
      <td>38083</td>
      <td>38543</td>
      <td>618948</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>31025</td>
      <td>34314</td>
      <td>30214</td>
      <td>24696</td>
      <td>27274</td>
      <td>23850</td>
      <td>28739</td>
      <td>43203</td>
      <td>47454</td>
      <td>60256</td>
      <td>...</td>
      <td>159253</td>
      <td>149054</td>
      <td>133459</td>
      <td>139894</td>
      <td>141434</td>
      <td>163845</td>
      <td>146894</td>
      <td>152218</td>
      <td>155075</td>
      <td>3317794</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>39760</td>
      <td>44802</td>
      <td>42720</td>
      <td>24638</td>
      <td>22287</td>
      <td>20844</td>
      <td>24370</td>
      <td>46698</td>
      <td>54726</td>
      <td>60893</td>
      <td>...</td>
      <td>35955</td>
      <td>33053</td>
      <td>33495</td>
      <td>34692</td>
      <td>35078</td>
      <td>33425</td>
      <td>26778</td>
      <td>29177</td>
      <td>28691</td>
      <td>1410947</td>
    </tr>
    <tr>
      <th>Latin America and the Caribbean</th>
      <td>13081</td>
      <td>15215</td>
      <td>16769</td>
      <td>15427</td>
      <td>13678</td>
      <td>15171</td>
      <td>21179</td>
      <td>28471</td>
      <td>21924</td>
      <td>25060</td>
      <td>...</td>
      <td>24747</td>
      <td>24676</td>
      <td>26011</td>
      <td>26547</td>
      <td>26867</td>
      <td>28818</td>
      <td>27856</td>
      <td>27173</td>
      <td>24950</td>
      <td>765148</td>
    </tr>
    <tr>
      <th>Northern America</th>
      <td>9378</td>
      <td>10030</td>
      <td>9074</td>
      <td>7100</td>
      <td>6661</td>
      <td>6543</td>
      <td>7074</td>
      <td>7705</td>
      <td>6469</td>
      <td>6790</td>
      <td>...</td>
      <td>8394</td>
      <td>9613</td>
      <td>9463</td>
      <td>10190</td>
      <td>8995</td>
      <td>8142</td>
      <td>7677</td>
      <td>7892</td>
      <td>8503</td>
      <td>241142</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>



Step 2: Plot the data. We will pass in `kind = 'pie'` keyword, along with the following additional parameters:
- `autopct` -  is a string or function used to label the wedges with their numeric value. The label will be placed inside the wedge. If it is a format string, the label will be `fmt%pct`.
- `startangle` - rotates the start of the pie chart by angle degrees counterclockwise from the x-axis.
- `shadow` - Draws a shadow beneath the pie (to give a 3D feel).


```python
# autopct create %, start angle represent starting point
df_continents['Total'].plot(kind='pie',
                            figsize=(5, 6),
                            autopct='%1.1f%%', # add in percentages
                            startangle=90,     # start angle 90° (Africa)
                            shadow=True,       # add shadow      
                            )

plt.title('Immigration to Canada by Continent [1980 - 2013]')
plt.axis('equal') # Sets the pie chart to look like a circle.

plt.show()
```


![png](output_24_0.png)


The above visual is not very clear, the numbers and text overlap in some instances. Let's make a few modifications to improve the visuals:

* Remove the text labels on the pie chart by passing in `legend` and add it as a seperate legend using `plt.legend()`.
* Push out the percentages to sit just outside the pie chart by passing in `pctdistance` parameter.
* Pass in a custom set of colors for continents by passing in `colors` parameter.
* **Explode** the pie chart to emphasize the lowest three continents (Africa, North America, and Latin America and Carribbean) by pasing in `explode` parameter.



```python
colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
explode_list = [0.1, 0, 0, 0, 0.1, 0.1] # ratio for each continent with which to offset each wedge.

df_continents['Total'].plot(kind='pie',
                            figsize=(15, 6),
                            autopct='%1.1f%%', 
                            startangle=90,    
                            shadow=True,       
                            labels=None,         # turn off labels on pie chart
                            pctdistance=1.12,    # the ratio between the center of each pie slice and the start of the text generated by autopct 
                            colors=colors_list,  # add custom colors
                            explode=explode_list # 'explode' lowest 3 continents
                            )

# scale the title up by 12% to match pctdistance
plt.title('Immigration to Canada by Continent [1980 - 2013]', y=1.12) 

plt.axis('equal') 

# add legend
plt.legend(labels=df_continents.index, loc='upper left') 

plt.show()
```


![png](output_26_0.png)


**Question:** Using a pie chart, explore the proportion (percentage) of new immigrants grouped by continents in the year 2013.

**Note**: You might need to play with the explore values in order to fix any overlapping slice values.


```python
### type your answer here




```

Double-click __here__ for the solution.
<!-- The correct answer is:
explode_list = [0.1, 0, 0, 0, 0.1, 0.2] # ratio for each continent with which to offset each wedge.
-->

<!--
df_continents['2013'].plot(kind='pie',
                            figsize=(15, 6),
                            autopct='%1.1f%%', 
                            startangle=90,    
                            shadow=True,       
                            labels=None,                 # turn off labels on pie chart
                            pctdistance=1.12,            # the ratio between the pie center and start of text label
                            explode=explode_list         # 'explode' lowest 3 continents
                            )
-->

<!--
\\ # scale the title up by 12% to match pctdistance
plt.title('Immigration to Canada by Continent in 2013', y=1.12) 
plt.axis('equal') 
-->

<!--
\\ # add legend
plt.legend(labels=df_continents.index, loc='upper left') 
-->

<!--
\\ # show plot
plt.show()
-->

# Box Plots <a id="8"></a>

A `box plot` is a way of statistically representing the *distribution* of the data through five main dimensions: 

- **Minimun:** Smallest number in the dataset.
- **First quartile:** Middle number between the `minimum` and the `median`.
- **Second quartile (Median):** Middle number of the (sorted) dataset.
- **Third quartile:** Middle number between `median` and `maximum`.
- **Maximum:** Highest number in the dataset.

<img src="https://ibm.box.com/shared/static/9nkxsfihu8mgt1go2kfasf61sywlu123.png" width=440, align="center">

To make a `box plot`, we can use `kind=box` in `plot` method invoked on a *pandas* series or dataframe.

Let's plot the box plot for the Japanese immigrants between 1980 - 2013.

Step 1: Get the dataset. Even though we are extracting the data for just one country, we will obtain it as a dataframe. This will help us with calling the `dataframe.describe()` method to view the percentiles.


```python
# to get a dataframe, place extra square brackets around 'Japan'.
df_japan = df_can.loc[['Japan'], years].transpose()
df_japan.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Country</th>
      <th>Japan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1980</th>
      <td>701</td>
    </tr>
    <tr>
      <th>1981</th>
      <td>756</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>598</td>
    </tr>
    <tr>
      <th>1983</th>
      <td>309</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>246</td>
    </tr>
  </tbody>
</table>
</div>



Step 2: Plot by passing in `kind='box'`.


```python
df_japan.plot(kind='box', figsize=(8, 6))

plt.title('Box plot of Japanese Immigrants from 1980 - 2013')
plt.ylabel('Number of Immigrants')

plt.show()
```


![png](output_36_0.png)


We can immediately make a few key observations from the plot above:
1. The minimum number of immigrants is around 200 (min), maximum number is around 1300 (max), and  median number of immigrants is around 900 (median).
2. 25% of the years for period 1980 - 2013 had an annual immigrant count of ~500 or fewer (First quartile).
2. 75% of the years for period 1980 - 2013 had an annual immigrant count of ~1100 or fewer (Third quartile).

We can view the actual numbers by calling the `describe()` method on the dataframe.


```python
df_japan.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Country</th>
      <th>Japan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>34.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>814.911765</td>
    </tr>
    <tr>
      <th>std</th>
      <td>337.219771</td>
    </tr>
    <tr>
      <th>min</th>
      <td>198.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>529.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>902.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1079.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1284.000000</td>
    </tr>
  </tbody>
</table>
</div>



One of the key benefits of box plots is comparing the distribution of multiple datasets. In one of the previous labs, we observed that China and India had very similar immigration trends. Let's analyize these two countries further using box plots.

**Question:** Compare the distribution of the number of new immigrants from India and China for the period 1980 - 2013.

Step 1: Get the dataset for China and India and call the dataframe **df_CI**.


```python
### type your answer here
df_CI= df_can.loc[['China', 'India'], years].transpose()
df_CI.head()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Country</th>
      <th>China</th>
      <th>India</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1980</th>
      <td>5123</td>
      <td>8880</td>
    </tr>
    <tr>
      <th>1981</th>
      <td>6682</td>
      <td>8670</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>3308</td>
      <td>8147</td>
    </tr>
    <tr>
      <th>1983</th>
      <td>1863</td>
      <td>7338</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>1527</td>
      <td>5704</td>
    </tr>
  </tbody>
</table>
</div>



Double-click __here__ for the solution.
<!-- The correct answer is:
df_CI= df_can.loc[['China', 'India'], years].transpose()
df_CI.head()
-->

Let's view the percentages associated with both countries using the `describe()` method.


```python
### type your answer here
df_CI.describe()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Country</th>
      <th>China</th>
      <th>India</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>34.000000</td>
      <td>34.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>19410.647059</td>
      <td>20350.117647</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13568.230790</td>
      <td>10007.342579</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1527.000000</td>
      <td>4211.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5512.750000</td>
      <td>10637.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>19945.000000</td>
      <td>20235.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>31568.500000</td>
      <td>28699.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>42584.000000</td>
      <td>36210.000000</td>
    </tr>
  </tbody>
</table>
</div>



Double-click __here__ for the solution.
<!-- The correct answer is:
df_CI.describe()
-->

Step 2: Plot data.


```python
### type your answer here
df_CI.plot(kind='box', figsize=(10, 7))
plt.title('Box plots of Immigrants from China and India (1980 - 2013)')
plt.xlabel('Number of Immigrants')
plt.show()


```


![png](output_47_0.png)


Double-click __here__ for the solution.
<!-- The correct answer is:
df_CI.plot(kind='box', figsize=(10, 7))
-->

<!--
plt.title('Box plots of Immigrants from China and India (1980 - 2013)')
plt.xlabel('Number of Immigrants')
-->

<!--
plt.show()
-->

We can observe that, while both countries have around the same median immigrant population (~20,000),  China's immigrant population range is more spread out than India's. The maximum population from India for any year (36,210) is around 15% lower than the maximum population from China (42,584).


If you prefer to create horizontal box plots, you can pass the `vert` parameter in the **plot** function and assign it to *False*. You can also specify a different color in case you are not a big fan of the default red color.


```python
# horizontal box plots
df_CI.plot(kind='box', figsize=(10, 7), color='blue', vert=False)

plt.title('Box plots of Immigrants from China and India (1980 - 2013)')
plt.xlabel('Number of Immigrants')

plt.show()
```


![png](output_51_0.png)


**Subplots**

Often times we might want to plot multiple plots within the same figure. For example, we might want to perform a side by side comparison of the box plot with the line plot of China and India's immigration.

To visualize multiple plots together, we can create a **`figure`** (overall canvas) and divide it into **`subplots`**, each containing a plot. With **subplots**, we usually work with the **artist layer** instead of the **scripting layer**. 

Typical syntax is : <br>
```python
    fig = plt.figure() # create figure
    ax = fig.add_subplot(nrows, ncols, plot_number) # create subplots
```
Where
- `nrows` and `ncols` are used to notionally split the figure into (`nrows` \* `ncols`) sub-axes,  
- `plot_number` is used to identify the particular subplot that this function is to create within the notional grid. `plot_number` starts at 1, increments across rows first and has a maximum of `nrows` * `ncols` as shown below.

<img src="https://ibm.box.com/shared/static/03rhrfcealyoi83tigscovgglfchfyor.png" width=500 align="center">

We can then specify which subplot to place each plot by passing in the `ax` paramemter in `plot()` method as follows:


```python
fig = plt.figure() # create figure

ax0 = fig.add_subplot(1, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)
ax1 = fig.add_subplot(1, 2, 2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**

# Subplot 1: Box plot
df_CI.plot(kind='box', color='blue', vert=False, figsize=(20, 6), ax=ax0) # add to subplot 1
ax0.set_title('Box Plots of Immigrants from China and India (1980 - 2013)')
ax0.set_xlabel('Number of Immigrants')
ax0.set_ylabel('Countries')

# Subplot 2: Line plot
df_CI.plot(kind='line', figsize=(20, 6), ax=ax1) # add to subplot 2
ax1.set_title ('Line Plots of Immigrants from China and India (1980 - 2013)')
ax1.set_ylabel('Number of Immigrants')
ax1.set_xlabel('Years')

plt.show()
```


![png](output_54_0.png)


** * Tip regarding subplot convention **

In the case when `nrows`, `ncols`, and `plot_number` are all less than 10, a convenience exists such that the a 3 digit number can be given instead, where the hundreds represent `nrows`, the tens represent `ncols` and the units represent `plot_number`. For instance,
```python
   subplot(211) == subplot(2, 1, 1) 
```
produces a subaxes in a figure which represents the top plot (i.e. the first) in a 2 rows by 1 column notional grid (no grid actually exists, but conceptually this is how the returned subplot has been positioned).

Let's try something a little more advanced. 

Previously we identified the top 15 countries based on total immigration from 1980 - 2013.

**Question:** Create a box plot to visualize the distribution of the top 15 countries (based on total immigration) grouped by the *decades* `1980s`, `1990s`, and `2000s`.

Step 1: Get the dataset. Get the top 15 countries based on Total immigrant population. Name the dataframe **df_top15**.


```python
### type your answer here
df_top15 = df_can.sort_values(['Total'], ascending=False, axis=0).head(15)
df_top15


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Continent</th>
      <th>Region</th>
      <th>DevName</th>
      <th>1980</th>
      <th>1981</th>
      <th>1982</th>
      <th>1983</th>
      <th>1984</th>
      <th>1985</th>
      <th>1986</th>
      <th>...</th>
      <th>2005</th>
      <th>2006</th>
      <th>2007</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>Total</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>India</th>
      <td>Asia</td>
      <td>Southern Asia</td>
      <td>Developing regions</td>
      <td>8880</td>
      <td>8670</td>
      <td>8147</td>
      <td>7338</td>
      <td>5704</td>
      <td>4211</td>
      <td>7150</td>
      <td>...</td>
      <td>36210</td>
      <td>33848</td>
      <td>28742</td>
      <td>28261</td>
      <td>29456</td>
      <td>34235</td>
      <td>27509</td>
      <td>30933</td>
      <td>33087</td>
      <td>691904</td>
    </tr>
    <tr>
      <th>China</th>
      <td>Asia</td>
      <td>Eastern Asia</td>
      <td>Developing regions</td>
      <td>5123</td>
      <td>6682</td>
      <td>3308</td>
      <td>1863</td>
      <td>1527</td>
      <td>1816</td>
      <td>1960</td>
      <td>...</td>
      <td>42584</td>
      <td>33518</td>
      <td>27642</td>
      <td>30037</td>
      <td>29622</td>
      <td>30391</td>
      <td>28502</td>
      <td>33024</td>
      <td>34129</td>
      <td>659962</td>
    </tr>
    <tr>
      <th>United Kingdom of Great Britain and Northern Ireland</th>
      <td>Europe</td>
      <td>Northern Europe</td>
      <td>Developed regions</td>
      <td>22045</td>
      <td>24796</td>
      <td>20620</td>
      <td>10015</td>
      <td>10170</td>
      <td>9564</td>
      <td>9470</td>
      <td>...</td>
      <td>7258</td>
      <td>7140</td>
      <td>8216</td>
      <td>8979</td>
      <td>8876</td>
      <td>8724</td>
      <td>6204</td>
      <td>6195</td>
      <td>5827</td>
      <td>551500</td>
    </tr>
    <tr>
      <th>Philippines</th>
      <td>Asia</td>
      <td>South-Eastern Asia</td>
      <td>Developing regions</td>
      <td>6051</td>
      <td>5921</td>
      <td>5249</td>
      <td>4562</td>
      <td>3801</td>
      <td>3150</td>
      <td>4166</td>
      <td>...</td>
      <td>18139</td>
      <td>18400</td>
      <td>19837</td>
      <td>24887</td>
      <td>28573</td>
      <td>38617</td>
      <td>36765</td>
      <td>34315</td>
      <td>29544</td>
      <td>511391</td>
    </tr>
    <tr>
      <th>Pakistan</th>
      <td>Asia</td>
      <td>Southern Asia</td>
      <td>Developing regions</td>
      <td>978</td>
      <td>972</td>
      <td>1201</td>
      <td>900</td>
      <td>668</td>
      <td>514</td>
      <td>691</td>
      <td>...</td>
      <td>14314</td>
      <td>13127</td>
      <td>10124</td>
      <td>8994</td>
      <td>7217</td>
      <td>6811</td>
      <td>7468</td>
      <td>11227</td>
      <td>12603</td>
      <td>241600</td>
    </tr>
    <tr>
      <th>United States of America</th>
      <td>Northern America</td>
      <td>Northern America</td>
      <td>Developed regions</td>
      <td>9378</td>
      <td>10030</td>
      <td>9074</td>
      <td>7100</td>
      <td>6661</td>
      <td>6543</td>
      <td>7074</td>
      <td>...</td>
      <td>8394</td>
      <td>9613</td>
      <td>9463</td>
      <td>10190</td>
      <td>8995</td>
      <td>8142</td>
      <td>7676</td>
      <td>7891</td>
      <td>8501</td>
      <td>241122</td>
    </tr>
    <tr>
      <th>Iran (Islamic Republic of)</th>
      <td>Asia</td>
      <td>Southern Asia</td>
      <td>Developing regions</td>
      <td>1172</td>
      <td>1429</td>
      <td>1822</td>
      <td>1592</td>
      <td>1977</td>
      <td>1648</td>
      <td>1794</td>
      <td>...</td>
      <td>5837</td>
      <td>7480</td>
      <td>6974</td>
      <td>6475</td>
      <td>6580</td>
      <td>7477</td>
      <td>7479</td>
      <td>7534</td>
      <td>11291</td>
      <td>175923</td>
    </tr>
    <tr>
      <th>Sri Lanka</th>
      <td>Asia</td>
      <td>Southern Asia</td>
      <td>Developing regions</td>
      <td>185</td>
      <td>371</td>
      <td>290</td>
      <td>197</td>
      <td>1086</td>
      <td>845</td>
      <td>1838</td>
      <td>...</td>
      <td>4930</td>
      <td>4714</td>
      <td>4123</td>
      <td>4756</td>
      <td>4547</td>
      <td>4422</td>
      <td>3309</td>
      <td>3338</td>
      <td>2394</td>
      <td>148358</td>
    </tr>
    <tr>
      <th>Republic of Korea</th>
      <td>Asia</td>
      <td>Eastern Asia</td>
      <td>Developing regions</td>
      <td>1011</td>
      <td>1456</td>
      <td>1572</td>
      <td>1081</td>
      <td>847</td>
      <td>962</td>
      <td>1208</td>
      <td>...</td>
      <td>5832</td>
      <td>6215</td>
      <td>5920</td>
      <td>7294</td>
      <td>5874</td>
      <td>5537</td>
      <td>4588</td>
      <td>5316</td>
      <td>4509</td>
      <td>142581</td>
    </tr>
    <tr>
      <th>Poland</th>
      <td>Europe</td>
      <td>Eastern Europe</td>
      <td>Developed regions</td>
      <td>863</td>
      <td>2930</td>
      <td>5881</td>
      <td>4546</td>
      <td>3588</td>
      <td>2819</td>
      <td>4808</td>
      <td>...</td>
      <td>1405</td>
      <td>1263</td>
      <td>1235</td>
      <td>1267</td>
      <td>1013</td>
      <td>795</td>
      <td>720</td>
      <td>779</td>
      <td>852</td>
      <td>139241</td>
    </tr>
    <tr>
      <th>Lebanon</th>
      <td>Asia</td>
      <td>Western Asia</td>
      <td>Developing regions</td>
      <td>1409</td>
      <td>1119</td>
      <td>1159</td>
      <td>789</td>
      <td>1253</td>
      <td>1683</td>
      <td>2576</td>
      <td>...</td>
      <td>3709</td>
      <td>3802</td>
      <td>3467</td>
      <td>3566</td>
      <td>3077</td>
      <td>3432</td>
      <td>3072</td>
      <td>1614</td>
      <td>2172</td>
      <td>115359</td>
    </tr>
    <tr>
      <th>France</th>
      <td>Europe</td>
      <td>Western Europe</td>
      <td>Developed regions</td>
      <td>1729</td>
      <td>2027</td>
      <td>2219</td>
      <td>1490</td>
      <td>1169</td>
      <td>1177</td>
      <td>1298</td>
      <td>...</td>
      <td>4429</td>
      <td>4002</td>
      <td>4290</td>
      <td>4532</td>
      <td>5051</td>
      <td>4646</td>
      <td>4080</td>
      <td>6280</td>
      <td>5623</td>
      <td>109091</td>
    </tr>
    <tr>
      <th>Jamaica</th>
      <td>Latin America and the Caribbean</td>
      <td>Caribbean</td>
      <td>Developing regions</td>
      <td>3198</td>
      <td>2634</td>
      <td>2661</td>
      <td>2455</td>
      <td>2508</td>
      <td>2938</td>
      <td>4649</td>
      <td>...</td>
      <td>1945</td>
      <td>1722</td>
      <td>2141</td>
      <td>2334</td>
      <td>2456</td>
      <td>2321</td>
      <td>2059</td>
      <td>2182</td>
      <td>2479</td>
      <td>106431</td>
    </tr>
    <tr>
      <th>Viet Nam</th>
      <td>Asia</td>
      <td>South-Eastern Asia</td>
      <td>Developing regions</td>
      <td>1191</td>
      <td>1829</td>
      <td>2162</td>
      <td>3404</td>
      <td>7583</td>
      <td>5907</td>
      <td>2741</td>
      <td>...</td>
      <td>1852</td>
      <td>3153</td>
      <td>2574</td>
      <td>1784</td>
      <td>2171</td>
      <td>1942</td>
      <td>1723</td>
      <td>1731</td>
      <td>2112</td>
      <td>97146</td>
    </tr>
    <tr>
      <th>Romania</th>
      <td>Europe</td>
      <td>Eastern Europe</td>
      <td>Developed regions</td>
      <td>375</td>
      <td>438</td>
      <td>583</td>
      <td>543</td>
      <td>524</td>
      <td>604</td>
      <td>656</td>
      <td>...</td>
      <td>5048</td>
      <td>4468</td>
      <td>3834</td>
      <td>2837</td>
      <td>2076</td>
      <td>1922</td>
      <td>1776</td>
      <td>1588</td>
      <td>1512</td>
      <td>93585</td>
    </tr>
  </tbody>
</table>
<p>15 rows × 38 columns</p>
</div>



Double-click __here__ for the solution.
<!-- The correct answer is:
df_top15 = df_can.sort_values(['Total'], ascending=False, axis=0).head(15)
df_top15
-->

Step 2: Create a new dataframe which contains the aggregate for each decade. One way to do that:
   1. Create a list of all years in decades 80's, 90's, and 00's.
   2. Slice the original dataframe df_can to create a series for each decade and sum across all years for each country.
   3. Merge the three series into a new data frame. Call your dataframe **new_df**.


```python
### type your answer here
years_80s = list(map(str, range(1980, 1990))) 
years_90s = list(map(str, range(1990, 2000))) 
years_00s = list(map(str, range(2000, 2010))) 
df_80s = df_top15.loc[:, years_80s].sum(axis=1) 
df_90s = df_top15.loc[:, years_90s].sum(axis=1) 
df_00s = df_top15.loc[:, years_00s].sum(axis=1)
new_df = pd.DataFrame({'1980s': df_80s, '1990s': df_90s, '2000s':df_00s}) 
new_df.head()



```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1980s</th>
      <th>1990s</th>
      <th>2000s</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>India</th>
      <td>82154</td>
      <td>180395</td>
      <td>303591</td>
    </tr>
    <tr>
      <th>China</th>
      <td>32003</td>
      <td>161528</td>
      <td>340385</td>
    </tr>
    <tr>
      <th>United Kingdom of Great Britain and Northern Ireland</th>
      <td>179171</td>
      <td>261966</td>
      <td>83413</td>
    </tr>
    <tr>
      <th>Philippines</th>
      <td>60764</td>
      <td>138482</td>
      <td>172904</td>
    </tr>
    <tr>
      <th>Pakistan</th>
      <td>10591</td>
      <td>65302</td>
      <td>127598</td>
    </tr>
  </tbody>
</table>
</div>



Double-click __here__ for the solution.
<!-- The correct answer is:
\\ # create a list of all years in decades 80's, 90's, and 00's
years_80s = list(map(str, range(1980, 1990))) 
years_90s = list(map(str, range(1990, 2000))) 
years_00s = list(map(str, range(2000, 2010))) 
-->

<!--
\\ # slice the original dataframe df_can to create a series for each decade
df_80s = df_top15.loc[:, years_80s].sum(axis=1) 
df_90s = df_top15.loc[:, years_90s].sum(axis=1) 
df_00s = df_top15.loc[:, years_00s].sum(axis=1)
-->

<!--
\\ # merge the three series into a new data frame
new_df = pd.DataFrame({'1980s': df_80s, '1990s': df_90s, '2000s':df_00s}) 
-->

<!--
\\ # display dataframe
new_df.head()
-->

Let's learn more about the statistics associated with the dataframe using the `describe()` method.


```python
### type your answer here
new_df.describe()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1980s</th>
      <th>1990s</th>
      <th>2000s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15.000000</td>
      <td>15.000000</td>
      <td>15.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>44418.333333</td>
      <td>85594.666667</td>
      <td>97471.533333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>44190.676455</td>
      <td>68237.560246</td>
      <td>100583.204205</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7613.000000</td>
      <td>30028.000000</td>
      <td>13629.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>16698.000000</td>
      <td>39259.000000</td>
      <td>36101.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>30638.000000</td>
      <td>56915.000000</td>
      <td>65794.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>59183.000000</td>
      <td>104451.500000</td>
      <td>105505.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>179171.000000</td>
      <td>261966.000000</td>
      <td>340385.000000</td>
    </tr>
  </tbody>
</table>
</div>



Double-click __here__ for the solution.
<!-- The correct answer is:
new_df.describe()
-->

Step 3: Plot the box plots.


```python
### type your answer here
new_df.plot(kind='box', figsize=(10, 6))
plt.title('Immigration from top 15 countries for decades 80s, 90s and 2000s')
plt.show()


```


![png](output_67_0.png)


Double-click __here__ for the solution.
<!-- The correct answer is:
new_df.plot(kind='box', figsize=(10, 6))
-->

<!--
plt.title('Immigration from top 15 countries for decades 80s, 90s and 2000s')
-->

<!--
plt.show()
-->

Note how the box plot differs from the summary table created. The box plot scans the data and identifies the outliers. In order to be an outlier, the data value must be:<br>
* larger than Q3 by at least 1.5 times the interquartile range (IQR), or,
* smaller than Q1 by at least 1.5 times the IQR.

Let's look at decade 2000s as an example: <br>
* Q1 (25%) = 36,101.5 <br>
* Q3 (75%) = 105,505.5 <br>
* IQR = Q3 - Q1 = 69,404 <br>

Using the definition of outlier, any value that is greater than Q3 by 1.5 times IQR will be flagged as outlier.

Outlier > 105,505.5 + (1.5 * 69,404) <br>
Outlier > 209,611.5


```python
# let's check how many entries fall above the outlier threshold 
new_df[new_df['2000s']> 209611.5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1980s</th>
      <th>1990s</th>
      <th>2000s</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>India</th>
      <td>82154</td>
      <td>180395</td>
      <td>303591</td>
    </tr>
    <tr>
      <th>China</th>
      <td>32003</td>
      <td>161528</td>
      <td>340385</td>
    </tr>
  </tbody>
</table>
</div>



China and India are both considered as outliers since their population for the decade exceeds 209,611.5. 

The box plot is an advanced visualizaiton tool, and there are many options and customizations that exceed the scope of this lab. Please refer to [Matplotlib documentation](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.boxplot) on box plots for more information.

# Scatter Plots <a id="10"></a>

A `scatter plot` (2D) is a useful method of comparing variables against each other. `Scatter` plots look similar to `line plots` in that they both map independent and dependent variables on a 2D graph. While the datapoints are connected together by a line in a line plot, they are not connected in a scatter plot. The data in a scatter plot is considered to express a trend. With further analysis using tools like regression, we can mathematically calculate this relationship and use it to predict trends outside the dataset.

Let's start by exploring the following:

Using a `scatter plot`, let's visualize the trend of total immigrantion to Canada (all countries combined) for the years 1980 - 2013.

Step 1: Get the dataset. Since we are expecting to use the relationship betewen `years` and `total population`, we will convert `years` to `int` type.


```python
# we can use the sum() method to get the total population per year
df_tot = pd.DataFrame(df_can[years].sum(axis=0))

# change the years to type int (useful for regression later on)
df_tot.index = map(int, df_tot.index)

# reset the index to put in back in as a column in the df_tot dataframe
df_tot.reset_index(inplace = True)

# rename columns
df_tot.columns = ['year', 'total']

# view the final dataframe
df_tot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>99137</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1981</td>
      <td>110563</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1982</td>
      <td>104271</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1983</td>
      <td>75550</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1984</td>
      <td>73417</td>
    </tr>
  </tbody>
</table>
</div>



Step 2: Plot the data. In `Matplotlib`, we can create a `scatter` plot set by passing in `kind='scatter'` as plot argument. We will also need to pass in `x` and `y` keywords to specify the columns that go on the x- and the y-axis.


```python
df_tot.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='darkblue')

plt.title('Total Immigration to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')

plt.show()
```


![png](output_76_0.png)


Notice how the scatter plot does not connect the datapoints together. We can clearly observe an upward trend in the data: as the years go by, the total number of immigrants increases. We can mathematically analyze this upward trend using a regression line (line of best fit). 

So let's try to plot a linear line of best fit, and use it to  predict the number of immigrants in 2015.

Step 1: Get the equation of line of best fit. We will use **Numpy**'s `polyfit()` method by passing in the following:
- `x`: x-coordinates of the data. 
- `y`: y-coordinates of the data. 
- `deg`: Degree of fitting polynomial. 1 = linear, 2 = quadratic, and so on.


```python
x = df_tot['year']      # year on x-axis
y = df_tot['total']     # total on y-axis
fit = np.polyfit(x, y, deg=1)

fit
```




    array([ 5.56709228e+03, -1.09261952e+07])



The output is an array with the polynomial coefficients, highest powers first. Since we are plotting a linear regression `y= a*x + b`, our output has 2 elements `[5.56709228e+03, -1.09261952e+07]` with the the slope in position 0 and intercept in position 1. 

Step 2: Plot the regression line on the `scatter plot`.


```python
df_tot.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='darkblue')

plt.title('Total Immigration to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')

# plot line of best fit
plt.plot(x, fit[0] * x + fit[1], color='red') # recall that x is the Years
plt.annotate('y={0:.0f} x + {1:.0f}'.format(fit[0], fit[1]), xy=(2000, 150000))

plt.show()

# print out the line of best fit
'No. Immigrants = {0:.0f} * Year + {1:.0f}'.format(fit[0], fit[1]) 
```


![png](output_81_0.png)





    'No. Immigrants = 5567 * Year + -10926195'



Using the equation of line of best fit, we can estimate the number of immigrants in 2015:
```python
No. Immigrants = 5567 * Year - 10926195
No. Immigrants = 5567 * 2015 - 10926195
No. Immigrants = 291,310
```
When compared to the actuals from Citizenship and Immigration Canada's (CIC) [2016 Annual Report](http://www.cic.gc.ca/english/resources/publications/annual-report-2016/index.asp), we see that Canada accepted 271,845 immigrants in 2015. Our estimated value of 291,310 is within 7% of the actual number, which is pretty good considering our original data came from United Nations (and might differ slightly from CIC data).

As a side note, we can observe that immigration took a dip around 1993 - 1997. Further analysis into the topic revealed that in 1993 Canada introcuded Bill C-86 which introduced revisions to the refugee determination system, mostly restrictive. Further amendments to the Immigration Regulations cancelled the sponsorship required for "assisted relatives" and reduced the points awarded to them, making it more difficult for family members (other than nuclear family) to immigrate to Canada. These restrictive measures had a direct impact on the immigration numbers for the next several years.

**Question**: Create a scatter plot of the total immigration from Denmark, Norway, and Sweden to Canada from 1980 to 2013?

Step 1: Get the data:
   1. Create a dataframe the consists of the numbers associated with Denmark, Norway, and Sweden only. Name it **df_countries**.
   2. Sum the immigration numbers across all three countries for each year and turn the result into a dataframe. Name this new dataframe **df_total**.
   3. Reset the index in place.
   4. Rename the columns to **year** and **total**.
   5. Display the resulting dataframe.


```python
### type your answer here
df_countries = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()
df_total = pd.DataFrame(df_countries.sum(axis=1))
df_total.reset_index(inplace=True)
df_total.columns = ['year', 'total']
df_total['year'] = df_total['year'].astype(int)
df_total.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>669</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1981</td>
      <td>678</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1982</td>
      <td>627</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1983</td>
      <td>333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1984</td>
      <td>252</td>
    </tr>
  </tbody>
</table>
</div>



Double-click __here__ for the solution.
<!-- The correct answer is:
\\ # create df_countries dataframe
df_countries = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()
-->

<!--
\\ # create df_total by summing across three countries for each year
df_total = pd.DataFrame(df_countries.sum(axis=1))
-->

<!--
\\ # reset index in place
df_total.reset_index(inplace=True)
-->

<!--
\\ # rename columns
df_total.columns = ['year', 'total']
-->

<!--
\\ # change column year from string to int to create scatter plot
df_total['year'] = df_total['year'].astype(int)
-->

<!--
\\ # show resulting dataframe
df_total.head()
-->

Step 2: Generate the scatter plot by plotting the total versus year in **df_total**.


```python
### type your answer here
df_total.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='darkblue')
plt.title('Immigration from Denmark, Norway, and Sweden to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.show()



```


![png](output_88_0.png)


Double-click __here__ for the solution.
<!-- The correct answer is:
\\ # generate scatter plot
df_total.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='darkblue')
-->

<!--
\\ # add title and label to axes
plt.title('Immigration from Denmark, Norway, and Sweden to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
-->

<!--
\\ # show plot
plt.show()
-->

# Bubble Plots <a id="12"></a>

A `bubble plot` is a variation of the `scatter plot` that displays three dimensions of data (x, y, z). The datapoints are replaced with bubbles, and the size of the bubble is determined by the third variable 'z', also known as the weight. In `maplotlib`, we can pass in an array or scalar to the keyword `s` to `plot()`, that contains the weight of each point.

**Let's start by analyzing the effect of Argentina's great depression**.

Argentina suffered a great depression from 1998 - 2002, which caused widespread unemployment, riots, the fall of the government, and a default on the country's foreign debt. In terms of income, over 50% of Argentines were poor, and seven out of ten Argentine children were poor at the depth of the crisis in 2002. 

Let's analyze the effect of this crisis, and compare Argentina's immigration to that of it's neighbour Brazil. Let's do that using a `bubble plot` of immigration from Brazil and Argentina for the years 1980 - 2013. We will set the weights for the bubble as the *normalized* value of the population for each year.

Step 1: Get the data for Brazil and Argentina. Like in the previous example, we will convert the `Years` to type int and bring it in the dataframe.


```python
df_can_t = df_can[years].transpose() # transposed dataframe

# cast the Years (the index) to type int
df_can_t.index = map(int, df_can_t.index)

# let's label the index. This will automatically be the column name when we reset the index
df_can_t.index.name = 'Year'

# reset index to bring the Year in as a column
df_can_t.reset_index(inplace=True)

# view the changes
df_can_t.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Country</th>
      <th>Year</th>
      <th>Afghanistan</th>
      <th>Albania</th>
      <th>Algeria</th>
      <th>American Samoa</th>
      <th>Andorra</th>
      <th>Angola</th>
      <th>Antigua and Barbuda</th>
      <th>Argentina</th>
      <th>Armenia</th>
      <th>...</th>
      <th>United States of America</th>
      <th>Uruguay</th>
      <th>Uzbekistan</th>
      <th>Vanuatu</th>
      <th>Venezuela (Bolivarian Republic of)</th>
      <th>Viet Nam</th>
      <th>Western Sahara</th>
      <th>Yemen</th>
      <th>Zambia</th>
      <th>Zimbabwe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>16</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>368</td>
      <td>0</td>
      <td>...</td>
      <td>9378</td>
      <td>128</td>
      <td>0</td>
      <td>0</td>
      <td>103</td>
      <td>1191</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
      <td>72</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1981</td>
      <td>39</td>
      <td>0</td>
      <td>67</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>426</td>
      <td>0</td>
      <td>...</td>
      <td>10030</td>
      <td>132</td>
      <td>0</td>
      <td>0</td>
      <td>117</td>
      <td>1829</td>
      <td>0</td>
      <td>2</td>
      <td>17</td>
      <td>114</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1982</td>
      <td>39</td>
      <td>0</td>
      <td>71</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>626</td>
      <td>0</td>
      <td>...</td>
      <td>9074</td>
      <td>146</td>
      <td>0</td>
      <td>0</td>
      <td>174</td>
      <td>2162</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
      <td>102</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1983</td>
      <td>47</td>
      <td>0</td>
      <td>69</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>241</td>
      <td>0</td>
      <td>...</td>
      <td>7100</td>
      <td>105</td>
      <td>0</td>
      <td>0</td>
      <td>124</td>
      <td>3404</td>
      <td>0</td>
      <td>6</td>
      <td>7</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1984</td>
      <td>71</td>
      <td>0</td>
      <td>63</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>42</td>
      <td>237</td>
      <td>0</td>
      <td>...</td>
      <td>6661</td>
      <td>90</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>7583</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 196 columns</p>
</div>



Step 2: Create the normalized weights. 

There are several methods of normalizations in statistics, each with its own use. In this case, we will use [feature scaling](https://en.wikipedia.org/wiki/Feature_scaling) to bring all values into the range [0,1]. The general formula is:

<img src="https://ibm.box.com/shared/static/3e43kt5j9wj4326x1lh8z2jeqzgpk3jv.png" align="center">

where *`X`* is an original value, *`X'`* is the normalized value. The formula sets the max value in the dataset to 1, and sets the min value to 0. The rest of the datapoints are scaled to a value between 0-1 accordingly.



```python
# normalize Brazil data
norm_brazil = (df_can_t['Brazil'] - df_can_t['Brazil'].min()) / (df_can_t['Brazil'].max() - df_can_t['Brazil'].min())

# normalize Argentina data
norm_argentina = (df_can_t['Argentina'] - df_can_t['Argentina'].min()) / (df_can_t['Argentina'].max() - df_can_t['Argentina'].min())
```

Step 3: Plot the data. 
- To plot two different scatter plots in one plot, we can include the axes one plot into the other by passing it via the `ax` parameter. 
- We will also pass in the weights using the `s` parameter. Given that the normalized weights are between 0-1, they won't be visible on the plot. Therefore we will:
    - multiply weights by 2000 to scale it up on the graph, and,
    - add 10 to compensate for the min value (which has a 0 weight and therefore scale with x2000).


```python
# Brazil
ax0 = df_can_t.plot(kind='scatter',
                    x='Year',
                    y='Brazil',
                    figsize=(14, 8),
                    alpha=0.5,                  # transparency
                    color='green',
                    s=norm_brazil * 2000 + 10,  # pass in weights 
                    xlim=(1975, 2015)
                   )

# Argentina
ax1 = df_can_t.plot(kind='scatter',
                    x='Year',
                    y='Argentina',
                    alpha=0.5,
                    color="blue",
                    s=norm_argentina * 2000 + 10,
                    ax = ax0
                   )

ax0.set_ylabel('Number of Immigrants')
ax0.set_title('Immigration from Brazil and Argentina from 1980 - 2013')
ax0.legend(['Brazil', 'Argentina'], loc='upper left', fontsize='x-large')
```




    <matplotlib.legend.Legend at 0x7fe46d4b0748>




![png](output_96_1.png)


The size of the bubble corresponds to the magnitude of immigrating population for that year, compared to the 1980 - 2013 data. The larger the bubble, the more immigrants in that year.

From the plot above, we can see a corresponding increase in immigration from Argentina during the 1998 - 2002 great depression. We can also observe a similar spike around 1985 to 1993. In fact, Argentina had suffered a great depression from 1974 - 1990, just before the onset of 1998 - 2002 great depression. 

On a similar note, Brazil suffered the *Samba Effect* where the Brazilian real (currency) dropped nearly 35% in 1999. There was a fear of a South American financial crisis as many South American countries were heavily dependent on industrial exports from Brazil. The Brazilian government subsequently adopted an austerity program, and the economy slowly recovered over the years, culminating in a surge in 2010. The immigration data reflect these events.

**Question**: Previously in this lab, we created box plots to compare immigration from China and India to Canada. Create bubble plots of immigration from China and India to visualize any differences with time from 1980 to 2013. You can use **df_can_t** that we defined and used in the previous example.

Step 1: Normalize the data pertaining to China and India.


```python
### type your answer here
norm_china = (df_can_t['China'] - df_can_t['China'].min()) / (df_can_t['China'].max() - df_can_t['China'].min())
norm_india = (df_can_t['India'] - df_can_t['India'].min()) / (df_can_t['India'].max() - df_can_t['India'].min())



```

Double-click __here__ for the solution.
<!-- The correct answer is:
\\ # normalize China data
norm_china = (df_can_t['China'] - df_can_t['China'].min()) / (df_can_t['China'].max() - df_can_t['China'].min())
-->

<!--
# normalize India data
norm_india = (df_can_t['India'] - df_can_t['India'].min()) / (df_can_t['India'].max() - df_can_t['India'].min())
-->

Step 2: Generate the bubble plots.


```python
### type your answer here
ax0 = df_can_t.plot(kind='scatter',
                    x='Year',
                    y='China',
                    figsize=(14, 8),
                    alpha=0.5,                  # transparency
                    color='green',
                    s=norm_china * 2000 + 10,  # pass in weights 
                    xlim=(1975, 2015)
                   )

ax1 = df_can_t.plot(kind='scatter',
                    x='Year',
                    y='India',
                    alpha=0.5,
                    color="blue",
                    s=norm_india * 2000 + 10,
                    ax = ax0
                   )

ax0.set_ylabel('Number of Immigrants')
ax0.set_title('Immigration from China and India from 1980 - 2013')
ax0.legend(['China', 'India'], loc='upper left', fontsize='x-large')
```




    <matplotlib.legend.Legend at 0x7fe46d443080>




![png](output_103_1.png)


Double-click __here__ for the solution.
<!-- The correct answer is:
\\ # China
ax0 = df_can_t.plot(kind='scatter',
                    x='Year',
                    y='China',
                    figsize=(14, 8),
                    alpha=0.5,                  # transparency
                    color='green',
                    s=norm_china * 2000 + 10,  # pass in weights 
                    xlim=(1975, 2015)
                   )
-->

<!--
\\ # India
ax1 = df_can_t.plot(kind='scatter',
                    x='Year',
                    y='India',
                    alpha=0.5,
                    color="blue",
                    s=norm_india * 2000 + 10,
                    ax = ax0
                   )
-->

<!--
ax0.set_ylabel('Number of Immigrants')
ax0.set_title('Immigration from China and India from 1980 - 2013')
ax0.legend(['China', 'India'], loc='upper left', fontsize='x-large')
-->

### Thank you for completing this lab!

This notebook was created by [Jay Rajasekharan](https://www.linkedin.com/in/jayrajasekharan) with contributions from [Ehsan M. Kermani](https://www.linkedin.com/in/ehsanmkermani), and [Slobodan Markovic](https://www.linkedin.com/in/slobodan-markovic).

This notebook was recently revamped by [Alex Aklson](https://www.linkedin.com/in/aklson/). I hope you found this lab session interesting. Feel free to contact me if you have any questions!

This notebook is part of a course on **Coursera** called *Data Visualization with Python*. If you accessed this notebook outside the course, you can take this course online by clicking [here](http://cocl.us/DV0101EN_Coursera_Week2_LAB2).

<hr>

Copyright &copy; 2018 [Cognitive Class](https://cognitiveclass.ai/?utm_source=bducopyrightlink&utm_medium=dswb&utm_campaign=bdu). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/).
