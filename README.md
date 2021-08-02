# Cryptocurrency Value and Risk Analysis Financial Model using Keras and Plotly

My first solo internship project involving statistical analysis and testing of a self made prediction model based on the data of a bunch of major cryptocurrency coins

# Project Brief

**Language used:** Python 3

**Modules used:** Pandas, Numpy, Matplotlib, Plotly, Seaborn, Keras

**IDE:** Google Colaboratory

**Data:** Daily Closing Prices of 22 different crypto coins compiled in a range from 05/10/2020 to 27/02/2021

**Prediction Model:** Keras Deep learning RNN

**Notebook:** [Link](https://colab.research.google.com/drive/1_7JOCQM4dHSGqj0YbmtLZ1_ni0nqWJIZ?usp=sharing)

# Introduction

The popularity of cryptocurrencies skyrocketed in 2017 due to several consecutive months of exponential growth of their market capitalization. The prices peaked at more than $800 billion in January 2018.

Although machine learning has been successful in predicting stock market prices through a host of different time series models, its application in predicting cryptocurrency prices has been quite restrictive. The reason behind this is obvious as prices of cryptocurrencies depend on a lot of factors like technological progress, internal competition, pressure on the markets to deliver, economic problems, security issues, political factors etc. Their high volatility leads to the great potential of high profit if intelligent inventing strategies are taken. Unfortunately, due to their lack of indexes, cryptocurrencies are relatively unpredictable compared to traditional financial predictions like stock market prediction.

In just 10 years, the cryptocurrency market shot up, from zero to approximately US$400B. The market that started with Bitcoin has now more than 3000 cryptocurrencies. As of June 2021, the total market cap of cryptocurrencies is US$1,746,285,217,570, but this success came as a result of a lot of volatility. Bitcoin, alone, fluctuated from its peak of US$60,000 to around US$30,000 recently. Over time, investors observed this volatility and realized that a lot of money can be made through crypto investments. But to make smart choices, anticipating the prices is crucial.

The stock market is also volatile. Yet, it is a tad bit easier to predict stock prices as compared to crypto prices. One of the reasons for that is the fact that cryptocurrency is still a new phenomenon. Unlike the stock market, crypto values do not exactly correspond with factors like cash flow and asset availability. Instead, a factor that does affect is sentiment.

Almost 90% of all crypto movement is linked to the herd instinct, which is when a majority of people think and act in the same way. News headlines, Reddit posts, and tweets navigate the direction of crypto prices. Using RNTN or recursive neural tensor networks, the sentiments of these texts can be analyzed to create an AI bot for cryptocurrency trading.

Cryptocurrency markets trade 24/7, which means, at any given minute, there are active traders monitoring crypto prices. This generates tons of data for AI to analyze for systematic trading that uses back-data findings (collecting and analyzing historical market pricing) to predict future prices. Employing AI for crypto price predictions makes it reliable by eliminating the risk of human error while calculating, and makes the entire process faster.

Crypto trading firms like Endor, Signal, and platforms like CryptoHawk.ai are leveraging this capability of AI to provide their users with crypto insights. Endor portrays itself as the ‘Google for predictive analytics.’ Endor’s protocol ensures small traders also receive crucial insights about the market without them having to conduct an extensive analysis of their own. To sharpen the model for accurate prediction, the firm takes data related to the user’s activity and recycles it back into their model.

So in this project, we will try to replicate a crypto price predictor based on a *deep learning model* and split the data into train-test subparts to counter check the predictions and evaluate the predictions of each coin individually on the basis of their mean-squared errors after some statistical analysis.

# Insights


After the data cleaning process, the following data visualization has been procured.


**Historical Price Data Comparison (05/10/2020 to 27/02/2021)**

<img src="https://github.com/babasxn/cryptoAnalysis/blob/main/screenshots/Picture1.png">

As we can see, ***Bitcoin*** and its decentralised counterpart ***Wrapped-Bitcoin*** stayed on the top of charts during this period.

<img src="https://github.com/babasxn/cryptoAnalysis/blob/main/screenshots/Picture2.png">

Meanwhile, coins like ***Aave, Binance, Lite and Monero*** dominated the mid range market and were the favourable choices for full time investors.

<img src="https://github.com/babasxn/cryptoAnalysis/blob/main/screenshots/Picture3.png">

On the bottom of the list, we had the incredible rise of the ***Doge Coin*** as a market leading coin (which even reached *$0.73* in May) and got worldwide fame which even attracted casual investors and newcomers.



**Simple Returns Scatter Plot Comparison**

A Simple returns graph shows us the investment advantage of a particular asset and can be used to pinpoint the timeline where that asset made its buyers some heavy profit by giving them enormous ‘returns’.

<img src="https://github.com/babasxn/cryptoAnalysis/blob/main/screenshots/Picture4.png">

Easily, ***Doge Coin*** dominated this comparison spectrum giving its investors some huge gains around the last week of Jan 2021.




**Simple Returns Frequency Distribution Comparison**

Frequency distribution shows the stability of a particular asset and can determine whether the returns of that asset is profitable or not.

<img src="https://github.com/babasxn/cryptoAnalysis/blob/main/screenshots/Picture5.png">

Again, ***Doge*** was the choice of investors giving its buyers the most net profit.

<img src="https://github.com/babasxn/cryptoAnalysis/blob/main/screenshots/Picture6.png">

While other coins were pretty much stable, ***Stellar*** and ***Uniswap*** showed significant dips around this period.

# Modelling and Predictions

To do ML based predictions, we have to create a deep learning model from scratch and implement it with the data of each coin so it evolves seamlessly.

For this we have used ***Keras*** based deep learning ***Simple RNN model*** which is a fully-connected RNN where the output from the previous timestep is to be fed to the next timestep.

RNNs support a number of useful features:

    1. Recurrent dropout, via the dropout and recurrent_dropout arguments
    
    2. Ability to process an input sequence in reverse, via the go_backwards argument
    
    3. Loop unrolling (which can lead to a large speedup when processing short sequences on CPU), via the unroll argument

After splitting the data into train and test data (70-30%), we implemented the RNN model to each coin individually and used an ***Epoch*** based fitting to train it to predict the prices for the test set. Epoch is an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.

So, in other words, a number of epochs means how many times you go through your training set. The model is updated each time a batch is processed, which means that it can be updated multiple times during one epoch. If batch_size is set equal to the length of x, then the model will be updated once per epoch.

In this project we have set the epochs to a maximum limit of ***500***. The fitting process of the first few coins took a bit of time but after going through 6-7 train data-sets, the epochs are restricted to around 20 as the model has learnt enough to do instant predictions for the rest.

# Results

After cross checking the predictions we evaluated a ***mean squared error*** for each coin by cross checking the predictions with the premade test data split set. 

The mean squared error (MSE) tells you how close a regression line is to a set of points. It does this by taking the distances from the points to the regression line (these distances are the “errors”) and squaring them. The squaring is necessary to remove any negative signs. It also gives more weight to larger differences. It’s called the mean squared error as you’re finding the average of a set of errors. 
        
                                 
                                            (1/n) * Σ(actual – forecast)^2
       
                     Where: 
                         n = number of items, 
                         Actual = original or observed y-value
                         Forecast = y-value from regression.


So here is the final list which shows the rounded-off mean squared errors of all the coins in a sorted order. 
        
                                          The lower the MSE, the better the predictions.

<img src="https://github.com/babasxn/cryptoAnalysis/blob/main/screenshots/Picture7.png">

Summarising, ***Tron*** and ***Doge Coin*** had the least MSE and therefore their prediction models were near perfect while ***Bitcoin*** (and WrappedBitcoin) had the most MSE and therefore the accuracy of its RNN model was the least.
