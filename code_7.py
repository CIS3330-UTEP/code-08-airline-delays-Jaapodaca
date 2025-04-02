import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
filename = 'Flight_Delays_2018.csv'
df = pd.read_csv('./Flight_Delays_2018.csv')

# Show basic info
print("\nSummary Statistics:")
print(df.describe())

# Plot distribution of ARR_DELAY
plt.figure(figsize=(10, 6))
plt.hist(df['ARR_DELAY'].dropna(), bins=50, edgecolor='black', color='skyblue')
plt.title('Distribution of Arrival Delays')
plt.xlabel('Arrival Delay (minutes)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Correlation matrix to help choose predictors
print("\nCorrelation with ARR_DELAY:")
print(df.corr(numeric_only=True)['ARR_DELAY'].sort_values(ascending=False))

# Analyze delay by airline
avg_delay_airline = df.groupby('OP_UNIQUE_CARRIER')['ARR_DELAY'].mean().sort_values(ascending=False)
print("\nAverage Arrival Delay by Airline:")
print(avg_delay_airline)

# Analyze delay by airport
avg_delay_airport = df.groupby('ORIGIN')['ARR_DELAY'].mean().sort_values(ascending=False)
print("\nAverage Arrival Delay by Origin Airport:")
print(avg_delay_airport)

# Boxplot: ARR_DELAY by day of week
plt.figure(figsize=(10, 6))
df.boxplot(column='ARR_DELAY', by='DAY_OF_WEEK')
plt.title('Arrival Delays by Day of Week')
plt.suptitle('')
plt.xlabel('Day of Week (1=Monday, 7=Sunday)')
plt.ylabel('Arrival Delay (minutes)')
plt.grid(True)
plt.show()

# -----------------------------
# 2. PREDICTIVE ANALYTICS
# -----------------------------

# Choose predictors based on descriptive analysis
predictors = ['DEP_DELAY', 'DISTANCE', 'DAY_OF_WEEK']
df_model = df[['ARR_DELAY'] + predictors].dropna()

# Set up X (independent variables) and Y (dependent variable)
X = df_model[predictors]
X = sm.add_constant(X)  # Add intercept
Y = df_model['ARR_DELAY']

# Fit the OLS regression model
model = sm.OLS(Y, X).fit()

# Output the model summary
print("\nOLS Regression Model Summary:")
print(model.summary())

# Visualize fit of one predictor (DEP_DELAY)
fig, ax = plt.subplots(figsize=(8, 6))
sm.graphics.plot_fit(model, exog_idx=1, ax=ax)  # Index 1 = DEP_DELAY (after constant)
plt.title('Fit Plot: DEP_DELAY vs. ARR_DELAY')
plt.grid(True)
plt.show()

#ARR_DELAY is the column name that should be used as dependent variable (Y).