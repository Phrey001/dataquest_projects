from datetime import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import style
"""
Expanding column width
"""
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


"""
Reading in the Data:
"""
df = pd.read_csv('sphist.csv')

df['Date'] = pd.to_datetime(df['Date'])

df.sort_values(by='Date',
                ascending=True,
                inplace=True)
df.reset_index(inplace=True, drop=True)

# print(df.head())
# df.info()
"""
Generating Indicators:
    Last 5 days computation
    Last 30 days computation
    Last 365 days computation
    last 5 days avg / last 365 days avg computation
"""
df["avg_5_days"] = 0
df["std_5_days"] = 0
df["avg_30_days"] = 0
df["std_30_days"] = 0
df["avg_365_days"] = 0
df["std_365_days"] = 0
df["ratio_avg_5_over_365"] = 0
for index, row in df.iterrows():

    last_5 = df.iloc[index-5:index, :]
    last_5_avg = last_5["Close"].mean()
    last_5_std = last_5["Close"].std()
    df.loc[index, "avg_5_days"] = last_5_avg
    df.loc[index, "std_5_days"] = last_5_std
    
    last_30 = df.iloc[index-30:index, :]
    last_30_avg = last_30["Close"].mean()
    last_30_std = last_30["Close"].std()
    df.loc[index, "avg_30_days"] = last_30_avg
    df.loc[index, "std_30_days"] = last_30_std
    
    last_365 = df.iloc[index-365:index, :]
    last_365_avg = last_365["Close"].mean()
    last_365_std = last_365["Close"].std()
    df.loc[index, "avg_365_days"] = last_365_avg
    df.loc[index, "std_365_days"] = last_365_std
    
    ratio_avg_5_over_365 = last_5_avg / last_365_avg
    df.loc[index, "ratio_avg_5_over_365"] = ratio_avg_5_over_365  

print(df.iloc[365:400, :])

"""
Splitting Up the Data:
    # Removing data points where there may not be enough historical data
    to generate derived values, eg. avg of last 365 days..etc
    # Next, generate train-test-split datasets, using a certain date as a cut-off
"""
df.dropna(axis=0,
          inplace=True)
df_train = df[df["Date"] < datetime(year=2013, month=1, day=1)]
df_test = df[df["Date"] >= datetime(year=2013, month=1, day=1)]

"""
Making Predictions:
    Let's try using the below 3 types of input data.
    # Linear Regression model with one possible predictor at a time on 3
    separate scenario at a time.
    past 5 days avg
    past 30 days avg
    past 365 days avg
"""
lr = LinearRegression()
lr.fit(df_train[["avg_5_days"]], df_train["Close"])
predictions_5days = lr.predict(df_test[["avg_5_days"]])

lr = LinearRegression()
lr.fit(df_train[["avg_30_days"]], df_train["Close"])
predictions_30days = lr.predict(df_test[["avg_30_days"]])

lr = LinearRegression()
lr.fit(df_train[["avg_365_days"]], df_train["Close"])
predictions_365days = lr.predict(df_test[["avg_365_days"]])


rmse_5days = mean_squared_error(predictions_5days, df_test["Close"]) ** 1/2
rmse_30days = mean_squared_error(predictions_30days, df_test["Close"]) ** 1/2
rmse_365days = mean_squared_error(predictions_365days, df_test["Close"]) ** 1/2


# Plot the model vs actual values
style.use("fivethirtyeight")
plt.figure(figsize=(15,15))
plt.plot(df_test["Date"], df_test["Close"], color='black', lw=1.0)
plt.plot(df_test["Date"], predictions_5days, lw=1.0)
plt.plot(df_test["Date"], predictions_30days, lw=1.0)
plt.plot(df_test["Date"], predictions_365days, lw=1.0)
plt.legend(["Actual",
            "Prediction (5 days)",
            "Prediction (30 days)",
            "Prediction (365 days)"])
plt.show()

# Display RMSE value
print("RMSE value (5 days):", round(rmse_5days,1))
print("RMSE value (30 days):", round(rmse_30days,1))
print("RMSE value (365 days):", round(rmse_365days,1))


"""
Improving Error:
    We can also try introducing more input parameters to try reduce the error
    term.
"""
lr = LinearRegression()
lr.fit(df_train[["avg_5_days", "avg_30_days", "avg_365_days",
                 "std_5_days", "std_30_days", "std_365_days"]],
       df_train["Close"])
predictions_mix = lr.predict(
    df_test[["avg_5_days", "avg_30_days", "avg_365_days",
             "std_5_days", "std_30_days", "std_365_days"]])

# Plot the model vs actual values
style.use("fivethirtyeight")
plt.figure(figsize=(15,15))
plt.plot(df_test["Date"], df_test["Close"], color='black', lw=1.0)
plt.plot(df_test["Date"], predictions_mix, lw=1.0)
plt.legend(["Actual",
            "Prediction (mix)"])
plt.show()

rmse_mix = mean_squared_error(predictions_mix, df_test["Close"]) ** 1/2
print("RMSE value (mix):", round(rmse_mix,1))

"""
Conclusion:
    Based on the line plotted, it would look like the error term RMSE value is lowest
    when the model is trained using 5 days, while the prediction line trained using the 
    365 days training data would appear to be the most smooth line, although it may have
    the highest RMSE from the actual test data.
    
    Also, the prediction model trained using multiple parameters does not appear to
    see significant improvements when compared to the model trained using avg past 5 days.
    Hence, we may suggest to ignore the model trained with mixed parameters.
"""
