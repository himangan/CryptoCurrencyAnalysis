# CryptoCurrencyAnalysis
The cryptocurrency market has been developing accordingly to the development of the technology, making it the market which grabs the most attention from the investors and the media. People talk and write about how they have managed to earn a decent sum of money by trading cryptocurrencies, which inspires others to do the same. Even those who have never thought about investing, have done so because the predictions say this is the market of the future. Reading internet articles from unknown sources may raise interest and motivate people to invest, but a certain level of knowledge about the market and the way it functions is important if one wishes to avoid the risk of losing the investment. So, analysis of cryptocurrencies trends and having a decent idea about its future is important for anyone willing to dive into the world of crypto. 
This is an end-to-end data analytics project where I have collected cryptocurrency data via the means of web scrapping and generated insights and predictions from the same. Finally, I have presented my work in the form of an interactive PowerBI dashboard to make it more presentable to the stakeholders
I have discussed the steps involved one by one:

STEP 1: Web-scrapping

For this first step, we have used the CoinDesk api that has been made public. (URL : https://production.api.coindesk.com/v2/price/values/). I have selected the top 20 cryptocurrencies for this procedure. They are BTC', 'ETH', 'XRP', 'ADA', 'USDT', 'DOGE', 'XLM', 'DOT', 'UNI', 'LINK', 'USDC', 'BCH', 'LTC', 'GRT', 'ETC', 'FIL', 'AAVE', 'ALGO', 'EOS' which I have store as a  dataframe named coin_list.
The values are available for every minute so for the sake of demonstration, I have shown the web scrapping of 12 hours of data (around 1400 for each cryptocurrency). We are using the request to fetch the data from the api in the JSON format and then storing it into the dataframe.

Note: JSON is  an open standard file format and data interchange format that uses human-readable text to store and transmit data objects consisting of attribute–value pairs and arrays

In numerous occasions, I observed that CoinDesk did not collect every minute of data. According to observation, in a normal circumstance, the timestamp would have a discrepancy value of 60000 for 1 minute. As a result, I can quickly tell that the missing period is the time gap if the row difference for the timestamp is larger than 60000. To deal with it, a hot-deck imputation procedure is used. To put it another way, the nearest minute data will  be substituted for the missing one.
For storing the data I have used df.to_csv to convert the fetched data into a csv file. Since the data has the price list of all the cryptocurrencies together in a stack manner (i.e. one after the another) , If one wants to analyse only a specific currency, then one might use the following piece of code for extraction of a seperate dataframe having the price list of that currency.
bitcoin_df = main_df.loc[main_df['Symbol'] == 'BTC'] 
bitcoin_df
So using this we can extract all the data for bitcoin

Note: We can also use the Yahoofinance api (yfinance) for the scrapping of cryptocurrency data. I have shown that in my time-series forecasting part of the project for collecting the bitcoin prices from over 7 years
STEP 2: EXPLORATORY DATA ANALYSIS

The first step of any EDA process is data cleaning.  Here the cleaning of data involved checking for any null values or values with wrong formatting. We found that the data type for the dates were string. So, we used to_datetime to convert the strings to a date-time format. This will be required in all of our further steps. 
We go over the market caps of the top 5 cryptocurrencies and observe that bitcoin and Ethereum are way bigger than all of the other coins.
Then we plot the closing prices of all the coins for the whole of the dataset to analyse the market trends over the years. Due to large value of the bitcoin and Ethereum closing prices, the scaling of our graphs is affected. Hence, to go over the trends of the other currencies we remove bitcoin and Ethereum and plot the graph.
In statistics, a moving average (rolling average or running average) is a calculation to analyze data points by creating a series of averages of different subsets of the full data set. We find out the moving average for the top cryptocurrency by using to different rolling windows in order to determine the trend direction of securities
For calculating multiple coins daily return over the whole time period, we have to add another column. The values are obtained by iterating through the closing prices and subtracting the current days price from the previous day to find the daily return value. We plot the daily returns of all the currencies to find insights regarding the volatility of the currencies.
Through these plots we can observe that the volatility of BTC and ETH is very low as their daily returns plots have very less spikes compared to other lower coins like USDT and ADA meaning they are high risk high reward investments.
Finally, we want to look at the dependencies of the variables on each other. We do that by obtaining a correlation matrix using df.corr(). Then we use a seaborn heatmap to visualize the correlation of the variables with each other. From the correlation matrix we inferred that the daily returns/losses are inversely related to the size of market cap and hence coins with low market caps are subject to more changes hence more returns.
Using proper financial analysis techniques, we can use the obtained charts and graphs to come with highly impactful insights that will help us in investing in cryptocurrencies.


STEP 3: Time-Series Forecasting of Bitcoin prices 
Time series forecasting occurs when you make scientific predictions based on historical time stamped data. It’s not always an exact prediction, and likelihood of forecasts can vary wildly—especially when dealing with the commonly fluctuating variables in time series data as well as factors outside our control. The same is the case with cryptocurrencies which are highly volatile in nature.
For achieving our forecast, we will be using an ARIMA model which stands for Auto Regressive Integrated Moving Average. is actually a class of models that ‘explains’ a given time series based on its own past values, that is, its own lags and the lagged forecast errors, so that equation can be used to forecast future values.   

A Stationary series is one whose statistical properties like mean, variance, covariance do not vary with time or these stats properties are not the function of time. Stationarity is important because many useful analytical tools and statistical tests and models rely on it and the same goes for time-series forecasting which needs to be performed on a stationary series for best results.
Differencing of a time series in discrete time is the transformation of the series to a new time series where the values are the differences between consecutive values of the original values .This procedure may be applied consecutively more than once, giving rise to the "first differences", "second differences", etc. 
A time-series like bitcoin prices is non stationary in nature owing to the volatility of cryptocurrencies. Hence differencing becomes a must. ARIMA model does this by itself and that is the main reason we use ARIMA for our forecasting purpose. 

Auto regressive (AR) model is a time series forecasting model where current values are dependent on past values. The higher the value of p will be the stronger will be the dependency of our forecast on our past values

In moving average (MA) model, the series is dependent on past error terms. It attempts to reduce the noise in our time series data by performing some sort of aggregation operation to your past observations in terms of residual error. A residual error is calculated as the expected outcome minus the forecast

Auto Regressive Integrated Moving Average Model is a combination of both AR and MA models. It makes the time series stationary by itself through the process of differencing.

An ARIMA model is characterized by 3 terms: p, d, q 
Where,

•	p is the order of the AR term

•	q is the order of the MA term

•	d is the number of differencing required to make the time series stationary

For this exercise we scrapped the data using the yahoofinance api which I mentioned previously. The main reason is that data from over 7 years can be very easily and quickly obtained and is necessary for running a time-series forecast. A total of 2826 records was obtained.
There was not any data cleaning to be done because there were no data irregularities (NA/null)
And the formatting was also as per our requirements (another advantage of yfinance).
Next, we split the dataset into a training set for training our model and test set for evaluating the performance of the same. We used simple pandas technique to make the training set the first 90% of the data set and remaining was the test set. We then create our ARIMA model by setting the hyperparameters as (4,1,0). The ideal values can simply be found by using the auto.arima function. Next step is to plot our original test set data and our predictions and compare the graphs. We then calculated the mean squared error for accuracy and found out that our forecast was more than 95% accurate.

STEP 4: Creating a dashboard for presentation

The final step involved exporting our scrapped data onto PowerBI as a CSV file and creating an interactive dashboard for presentation. It is a simple dashboard with a home screen and 3 other pages that can accessed with navigation buttons provided. 
1.	The first page was a basic information page giving us the price trends curve, average volume curve along with the current values for those columns. I added slicers to switch between the different currencies by selecting buttons and also added a slider that would be used to select a date range for the visualizations.
2.	The second page has a price comparison table that is also dynamic and can be changed using the last date input (which is also interactive and can be changed according to the viewers requirement). Also, according to the last date, I built a few key measures like best and worst performer over the last year and month. I made a bunch of other key measures by writing the respective scripts for practice purpose but did not add them in the final dashboard. These included %change , average volume, average price etc.
