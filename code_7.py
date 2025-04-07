import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm_api
filename = 'Flight_Delays_2018.csv'
df = pd.read_csv('./Flight_Delays_2018.csv')

# Filter dataset
query = "OP_CARRIER_NAME == 'American Airlines Inc.' or OP_CARRIER_NAME == 'Delta Air Lines Inc.' or OP_CARRIER_NAME == 'United Air Lines Inc.'"
df = df.query(query)

# Define predictors and target
x = df[[
    'DEP_DELAY', 'CARRIER_DELAY', 'LATE_AIRCRAFT_DELAY',
    'NAS_DELAY', 'WEATHER_DELAY', 'SECURITY_DELAY'
]]
y = df['ARR_DELAY']
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())

# Fit ols model using formula style
model = sm_api.ols('ARR_DELAY ~ DEP_DELAY + CARRIER_DELAY + LATE_AIRCRAFT_DELAY + NAS_DELAY + WEATHER_DELAY + SECURITY_DELAY', data=df).fit()
print(model.summary())

# Boxplot of ARR_DELAY by airline
df.boxplot(column="ARR_DELAY", by="OP_CARRIER_NAME")
plt.xticks(rotation=90)
plt.show()