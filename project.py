

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import os



path = r"C:\Users\babo2\Downloads\project_folder\project_data.xlsx"
df = pd.read_excel(path)


date_col = None
target_col = None
for c in df.columns:
    if c.lower() in ['transdate','settlement date','date']:
        date_col = c
        break
for c in df.columns:
    if c.lower() in ['amount','total','revenue']:
        target_col = c
        break



if date_col is None:
    for c in df.columns:
        try:
            pd.to_datetime(df[c].iloc[:10])
            date_col = c
            break
        except:
            pass
if target_col is None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = num_cols[0] if num_cols else None



df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col])
df = df.sort_values(date_col)





df2 = df.groupby(pd.Grouper(key=date_col, freq='D'))[target_col].sum().reset_index().rename(columns={date_col:'ds', target_col:'y'})


full_range = pd.date_range(df2['ds'].min(), df2['ds'].max(), freq='D')
df2 = df2.set_index('ds').reindex(full_range).rename_axis('ds').reset_index()
df2['y'] = df2['y'].fillna(0)




plt.figure(figsize=(10,4))
plt.plot(df2['ds'], df2['y'])
plt.title('Sales over Time')
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/eda_plot.png")
plt.close()





df_feat = df2.copy()
df_feat['month'] = df_feat['ds'].dt.month
df_feat['day'] = df_feat['ds'].dt.day
df_feat['dow'] = df_feat['ds'].dt.dayofweek
df_feat['lag1'] = df_feat['y'].shift(1)
df_feat['lag7'] = df_feat['y'].shift(7)
df_feat['roll7'] = df_feat['y'].rolling(7).mean()
df_feat = df_feat.fillna(0)



test_days = 90
train = df_feat[:-test_days]
test = df_feat[-test_days:]
X_cols = ['month','day','dow','lag1','lag7','roll7']
X_train, y_train = train[X_cols], train['y']
X_test, y_test = test[X_cols], test['y']




model_xgb = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
model_xgb.fit(X_train, y_train)
pred_xgb = model_xgb.predict(X_test)
mae = mean_absolute_error(y_test, pred_xgb)
rmse = np.sqrt(mean_squared_error(y_test, pred_xgb))
joblib.dump(model_xgb, "outputs/model_xgb.pkl")



m = Prophet()
m.fit(df2.rename(columns={'ds':'ds','y':'y'}))
future = m.make_future_dataframe(periods=90, freq='D')
forecast = m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].to_csv("outputs/forecast_prophet.csv", index=False)
joblib.dump(m, "outputs/model_prophet.pkl")




st.set_page_config(page_title="Sales Forecast Analysis", layout="wide")
st.title("Sales Forecast Analysis")
tab1, tab2, tab3 = st.tabs(["Raw Data","Prophet Forecast","XGBoost Prediction"])




with tab1:
    st.subheader("Raw Time Series Data")
    st.line_chart(df2.set_index('ds')['y'])
    st.dataframe(df2.tail(20))





with tab2:
    st.subheader("Prophet Forecast")
    st.line_chart(forecast.set_index('ds')[['yhat','yhat_lower','yhat_upper']])
    st.write(forecast.tail(10))





with tab3:
    st.subheader("XGBoost Model Evaluation")
    st.metric("MAE", round(mae,2))
    st.metric("RMSE", round(rmse,2))
    pred_df = pd.DataFrame({'ds': test['ds'], 'Actual': y_test, 'Predicted': pred_xgb})
    st.line_chart(pred_df.set_index('ds'))





st.success("Models trained and forecasts generated successfully.")





