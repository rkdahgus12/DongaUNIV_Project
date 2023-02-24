
<div align="center" style="color:blue;">
<h1>Visualization of Future Price Forecasts Using the Web</h2>
  <h2>ViewMent</h2>
</div>

![캡처](https://user-images.githubusercontent.com/71003685/221066756-7df3420b-e20e-49c7-8359-3330ae09bd31.PNG)

### Required Library(No version)
- pymsql
- numpy
- pandas (csv)
- matplotlib
- arima_model
- sklearn (mean_absolutre_error, r2_score)
- Prophet (facebookprophet)


### Dataset Utilization
- https://www.data.go.kr/
- banana.csv, apple.csv


### ARIMA Model Description

- ARMA is a combination of 1)'s AR and 2)'s MA model. It means that it will reflect all of the past time self and trends that grasp the state of the current time point, and if you look at the formula for the ARMA(1,1) model, it can be simply combined as follows. (Concept of increasing terms of independent variables in regression analysis)


![캡처](https://user-images.githubusercontent.com/71003685/221067785-06974e1b-45ff-4a4e-8cba-550d8d32a56f.PNG)

### Facebook Prophet Model Description

- Simply describing the content that can be modeled means that you can construct a black box that can mimic sequential data in time series data. Black boxes sometimes contain hundreds or tens of millions of parameters.


![캡처](https://user-images.githubusercontent.com/71003685/221068225-2e61abcc-aadc-4cde-95a4-afa7bd3c125e.PNG)

- g(t) : piecewise linear or logistic growth curve for modelling non-periodic changes in time series
- s(t): periodic changes (e.g. weekly/yearly seasonality)
- h(t): effects of holidays (user provided) with irregular schedules
- ϵi: error term accounts for any unusual changes not accommodated by the model


### Data Visualization

#### ARIMA  <> Facebook Prophet

![arimaface](https://user-images.githubusercontent.com/71003685/221068699-a97a3b91-a42d-4924-9e8a-5b6f1758723a.PNG)

#### Facebook Prophet

![face](https://user-images.githubusercontent.com/71003685/221068772-9740a11e-2d23-49c6-9b4c-1b553c654ff3.PNG)


