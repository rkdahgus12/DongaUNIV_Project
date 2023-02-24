########################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
file_path = 'banana.csv'
apple_predict_df = pd.read_csv(file_path, names=['day', 'price'])
a = apple_predict_df['price']
apple_predict_df['price'] = a[:].astype(np.float)
print(apple_predict_df.shape)
print(apple_predict_df.info())

apple_predict_df['day'] = pd.to_datetime(apple_predict_df['day'])
apple_predict_df.index = apple_predict_df['day']
apple_predict_df.set_index('day', inplace=True)
print(apple_predict_df.head())
apple_predict_df.describe()
print(apple_predict_df.describe())
apple_predict_df.plot()
plt.show()

from statsmodels.tsa import arima_model
import statsmodels.api as sm

# (AR=2, 차분=1, MA=2) 파라미터로 ARIMA 모델을 학습합니다.
model = arima_model.ARIMA(apple_predict_df.price.values, order=(2, 1, 2))

# trend : constant를 가지고 있는지, c - constant / nc - no constant
# disp : 수렴 정보를 나타냄
model_fit = model.fit(trend='nc', full_output=True, disp=1)
print(model_fit.summary())
fig = model_fit.plot_predict()
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
forecast_data = model_fit.forecast(steps=5)  # 학습 데이터셋으로부터 5일 뒤를 예측합니다.

# 테스트 데이터셋을 불러옵니다.
test_file_path = 'banana.csv'
apple_predict_test_df = pd.read_csv(test_file_path, names=['ds', 'y'])

pred_y = forecast_data[0].tolist()  # 마지막 5일의 예측 데이터입니다.
test_y = apple_predict_test_df.y.values  # 실제 5일 가격 데이터입니다
print(test_y)
pred_y_lower = []  # 마지막 5일의 예측 데이터의 최소값입니다.
pred_y_upper = []  # 마지막 5일의 예측 데이터의 최대값입니다.
for lower_upper in forecast_data[2]:
    lower = lower_upper[0]
    upper = lower_upper[1]
    pred_y_lower.append(lower)
    pred_y_upper.append(upper)
plt.plot(pred_y, color="gold")  # 모델이 예상한 가격 그래프입니다.
plt.plot(pred_y_lower, color="red")  # 모델이 예상한 최소가격 그래프입니다.
plt.plot(pred_y_upper, color="blue")  # 모델이 예상한 최대가격 그래프입니다.
plt.plot(test_y[len(test_y) - 5:len(test_y)], color="green")
plt.show()
plt.plot(pred_y, color="gold")
plt.plot(test_y[len(test_y) - 5:len(test_y)], color="green")
plt.show()
from sklearn.metrics import mean_absolute_error, r2_score
from math import sqrt

# rmse는 Root mean supports error(회귀 모형 평가 낮을수록 정확도가 상승)
rmse = sqrt(mean_absolute_error(pred_y, test_y[len(test_y) - 5:len(test_y)]))
print(rmse)
from fbprophet import Prophet

apple_predict_df = pd.read_csv('banana.csv', names=['ds', 'y'])
prophet = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True, weekly_seasonality=True,
                  daily_seasonality=True, changepoint_prior_scale=0.5)
prophet.fit(apple_predict_df)
future_data = prophet.make_future_dataframe(periods=5, freq='d')
forecast_data = prophet.predict(future_data)
print(forecast_data.tail(5))
print(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5))
# 전체데이터를 기반으로 학습한 5일단위의 예측결과 시각화
fig1 = prophet.plot(forecast_data)
plt.show()
fig2 = prophet.plot_components(forecast_data)
plt.show()
y = apple_predict_df.y.values[5:]
y_pred = forecast_data.yhat.values[5:-5]
rmse = sqrt(mean_absolute_error(y, y_pred))
r2 = r2_score(y, y_pred)
print(rmse)
print(r2)
print(y_pred)
apple_predict_test_df = pd.read_csv('predict_test.csv', names=['ds', 'y'])
pred_y = forecast_data.yhat.values[-5:]
test_y = apple_predict_test_df.y.values
pred_y_lower = forecast_data.yhat_lower.values[-5:]
pred_y_upper = forecast_data.yhat_upper.values[-5:]
plt.plot(pred_y, color="gold")  # 모델이 예상한 가격 그래프입니다.
plt.plot(pred_y_lower, color="red")  # 모델이 예상한 최소가격 그래프입니다.
plt.plot(pred_y_upper, color="blue")  # 모델이 예상한 최대가격 그래프입니다.
plt.plot(test_y[len(test_y) - 5:len(test_y)], color="green")
plt.show()
plt.plot(pred_y, color="gold")
plt.plot(test_y[len(test_y) - 5:len(test_y)], color="green")
plt.show()
rmse = sqrt(mean_absolute_error(pred_y, test_y))
print(rmse)
'''
forecast_data = model_fit.forecast(steps=5)
forecast_data=list(forecast_data)
test_file_path = 'predict_test.csv'

b = apple_predict_test_df['y']
apple_predict_test_df['y'] = b[:].astype(np.float)
pred_y = forecast_data.yhat.values[-5:]
test_y = apple_predict_test_df.y.values
pred_y_lower = forecast_data.yhat_lower.values[-5:]
pred_y_upper = forecast_data.yhat_upper.values[-5:]
plt.plot(pred_y, color="gold")  # 모델이 예상한 가격 그래프입니다.
plt.plot(pred_y_lower, color="red")  # 모델이 예상한 최소가격 그래프입니다.
plt.plot(pred_y_upper, color="blue")  # 모델이 예상한 최대가격 그래프입니다.
plt.plot(test_y[len(test_y) - 5:len(test_y)], color="green")
plt.show()
plt.plot(pred_y, color="gold")
plt.plot(test_y[len(test_y) - 5:len(test_y)], color="green")
plt.show()
'''
# print(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5))
apple_ds = forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
# 데이터프레임 행렬로 출력해서 json 변경

a = forecast_data[['yhat'][0]].tail(5)
print(a)
for i in range(0,5):
    print(a.iloc[i])
a = a.iloc[0]
a1 = a.iloc[1]
a2 = a.iloc[2]
a3 = a.iloc[3]
a4 = a.iloc[4]
a = str(a)
a1 = str(a1)
a2 = str(a2)
a3 = str(a3)
a4 = str(a4)

b = forecast_data[['ds'][0]].tail(5)
b = b.iloc[0]
b1 = b.iloc[1]
b2 = b.iloc[2]
b3 = b.iloc[3]
b4 = b.iloc[4]
b = str(b)
b1 = str(b1)
b2 = str(b2)
b3 = str(b3)
b4 = str(b4)

# 데이터 프레임이라서 안들어감 하나씩
# 결과값 DB에 저장
import pymysql

juso_db = pymysql.connect(
    user='root',
    passwd='qmffpdlem12',
    host='127.0.0.1',
    db='test',
    charset='utf8'
)
cursor = juso_db.cursor(pymysql.cursors.DictCursor)
var1 = a
var1_1 = a1
var1_2 = a2
var1_3 = a3
var1_4 = a4
var2 = b
var2_1 = b1
var2_2 = b2
var2_3 = b3
var2_4 = b4
sql = "INSERT INTO db_test VALUES (%s, %s)"
val = (var1, var2)
cursor.execute(sql, val)
sql = "INSERT INTO db_test VALUES (%s, %s)"
val = (var1_1, var2_1)
cursor.execute(sql, val)
sql = "INSERT INTO db_test VALUES (%s, %s)"
val = (var1_2, var2_2)
cursor.execute(sql, val)
sql = "INSERT INTO db_test VALUES (%s, %s)"
val = (var1_3, var2_3)
cursor.execute(sql, val)
sql = "INSERT INTO db_test VALUES (%s, %s)"
val = (var1_4, var2_4)
cursor.execute(sql, val)
juso_db.commit()

# result = cursor.fetchall()
# print(result)
# sql = "select *from db_test;"
# cursor.execute(sql)
