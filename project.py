import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

#cleaning just for testing
data = pd.read_csv('C:\\Users\\ahmed\\Downloads\\masked_kiva_loans.csv')
#print(data.head())
#Counts non-null values for each column.
print(data.count())
# data=data.dropna()
#bi3d 3dadd alnulls
print('Null data in columns: ')
print(data.isnull().sum())?
print('==========================================')
#bi3d alcountries
print('Countries')
print(data['country'].value_counts())
print('==========================================')
#bi4il kol alulls mn al tlata dol
data = data.dropna(subset=['funded_amount', 'sector', 'lender_count'])
#b3wad mkan alnull di b almean
partner_mean = data['partner_id'].mean()
data.loc[data['partner_id'].isnull(), 'partner_id'] = partner_mean
#b3wad mkan alnull di b almode
# may cause problems because borrower genders are multivalued
borrower_mode = data['borrower_genders'].mode()[0]
data.loc[data['borrower_genders'].isnull(), 'borrower_genders'] = borrower_mode
#btb3 alnulls tani b4of lsa fy nulls wla la
print(data.isnull().sum())
#bdrob kol alduplicates
data = data.drop_duplicates()
#b7awl colam aldate mn text l date
data['date'] = pd.to_datetime(data['date'])
# print(f"dulicates : {data['partner_id'].isnull().sum()}")
#bsave aldata alndifa fy file gdid
data.to_csv('C:\\Users\\ahmed\\Downloads\\cleand_masked_kiva_loans.csv', index=False)

# get sectors that received the highest funding
#bngma3 data bta3t colm sector b3dha bngm3 alfunded_amount lkol sector b3dha nrtbhom ascending
sector_highest_funding = data.groupby('sector')['funded_amount'].sum().sort_values(ascending=False)
#brsm b dol barchart lkol sector w alfunded amount bt3to
sector_highest_funding.plot(kind='bar', figsize=(12, 6), color='skyblue')
plt.title('Total Funded Amount with Sector')
plt.xlabel('Sector')
plt.ylabel('Funded Amount')
plt.xticks(rotation=45)
plt.show()

# is the funded amount increasing or decreasing over time?
#bn7wl aldate l 4hor w bngm3 alfunded amount 7sab kol 4ahr
funding_over_months = data.groupby(data['date'].dt.to_period('M'))['funded_amount'].sum()
#bnrsm barchart bs lkol 4ahr
funding_over_months.plot(kind='line', figsize=(12, 6), color='red')
plt.title('funding over time')
plt.xlabel('month')
plt.ylabel('funded amount')
plt.show()
#bnrsm  Scatter Plot mbin lender_count (3dd alnas ali 5dt korod) w al fundeing amount
# is there a correlation between funded amount and loan amount?
plt.figure(figsize=(12, 6))
plt.scatter(data['lender_count'], data['funded_amount'], alpha=0.5)
plt.title('Lender Count vs Funded Amount')
plt.xlabel('Lender Count')
plt.ylabel('Funded Amount')
plt.show()


#  (EDA)
print("\nData Info:")
#bi3rd 3dd alcolms w 3dd alnulls ali fihom
print(data.info())
print("\nSummary Statistics:")
#bi3rd almaxmum w alminimum w alstandard deviation
print(data.describe())
#bnrsm histogram lkol mn funded_amount loan_amount lender_count 34an a4of fy outlier wla la
numeric_cols = ['funded_amount', 'loan_amount', 'lender_count']
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    data[col].hist(bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
#bt7sb correlation mbin llcolmas alrkmia bs
plt.figure(figsize=(10, 8))
numeric_df = data.select_dtypes(include='number')

sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
#bnrsm heat map
plt.title('Correlation Matrix')
plt.show()
#bn3ml sum llfundeed amount w bnrtbhom asnding
sector_funding = data.groupby('sector')['funded_amount'].sum().sort_values(ascending=False)
print("\nTop Funded Sectors:")
print(sector_funding)
#bnrsm graph biwtd7 fundined amount lkol sector
sector_funding.plot(kind='bar', figsize=(12, 6))
plt.title('Total Funded Amount by Sector')
plt.ylabel('Funded Amount')
plt.xticks(rotation=45)
plt.show()
#bn7sb altotal funding amount lkol country w bntb3 awl 10
country_funding = data.groupby('country')['funded_amount'].sum().sort_values(ascending=False)
print("\nTop Funded Countries:")
print(country_funding.head(10))
#bnrsm barchart l top 10 countrys dol
country_funding.head(10).plot(kind='bar', figsize=(12, 6), color='green')
plt.title('Top 10 Countries by Funded Amount')
plt.ylabel('Funded Amount')
plt.xticks(rotation=45)
plt.show()
#bnrsm scatter plot l allender_conut (3dadd almoktrdin) w alfunded_amount
plt.figure(figsize=(8, 6))

sns.scatterplot(x='lender_count', y='funded_amount', data=data)
plt.title('Lender Count vs Funded Amount')
plt.xlabel('Lender Count')
plt.ylabel('Funded Amount')
plt.show()

## Data cleaning
#bn3rd alnulls tani
print("Missing values before cleaning:")
print(data.isnull().sum())

# data.drop('borrower_genders', axis=1, inplace=True)
# data.drop('partner_id', axis=1, inplace=True)
#bn7sb 3dd aldublicats w n4lhom
print(f"Number of duplicate rows:")
data.duplicated().sum()
data.drop_duplicates(inplace=True)
#bn7sb al IQR 34an n4il aloutliers
Q1 = data['funded_amount'].quantile(0.25)
Q3 = data['funded_amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_clean = data[(data['funded_amount'] >= lower_bound) & (data['funded_amount'] <= upper_bound)]
#bnkarn mbin 3dd alcolams abl m4il aloutliers w b3d
print("Number of row before cleaning:", len(data))
print("Number of row after cleaning:", len(df_clean))
#bnrsm blotbox abl w b3d aloutliers
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
axes[0].boxplot(data['funded_amount'])
axes[0].set_title('Before Removing Outliers')
axes[0].set_ylabel('Funded Amount')
axes[1].boxplot(df_clean['funded_amount'])
axes[1].set_title('After Removing Outliers')
axes[1].set_ylabel('Funded Amount')
plt.tight_layout()
plt.show()
print("\nData cleaning completed successfully")

# Models
#bnt2kd an alrows ali hnst5dmha mfha4 nulls
df = data.dropna(subset=['funded_amount', 'loan_amount', 'term_in_months', 'lender_count'])
df = data.dropna(subset=['sector', 'country', 'repayment_interval'])
#festures hya alinput ali h3tmd 3lih
#target alkima ali htnb2 biha
features = ['loan_amount', 'term_in_months', 'lender_count']
target = 'funded_amount'
# bn2sm aldata ali htest whtrain biha 80% train 20%test
#random state da rakm 34wa2i bs 34an asbt kimt altrain w altest
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Define the models
#bn3rf almodels ali hn4t8l biha
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42)

}
#bndrb kol model 3la al 80% bto3 altrain
# w bngrb 3ltest
#b7sb al R² Score w da kol ma bi3rb mn al 1 kol mkan almodel a7sn
# bn7sb al RMSE w da kol ma 2al kol ma alerror 2al
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name}:")
    print("R² Score:", round(r2_score(y_test, y_pred), 3))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))

# df['date'] = pd.to_datetime(df['date'])
#
# # is the funded amount increasing or decreasing over time?
# funding_over_months = df.groupby(df['date'].dt.to_period('M'))['funded_amount'].sum()
# funding_over_months.plot(kind='line', figsize=(12, 6), color='red')
# plt.title('funding over time')
# plt.xlabel('month')
# plt.ylabel('funded amount')
# plt.show()
#bn5ali aldate hwa alindex bta3 aldata 34an aldata tkon mtrtba zamanin s7
df.set_index('date', inplace=True)
df = df.sort_index()

###### Time Series
# Step 1: Visualize the time series
#bnrsm graph bnzba llw2t 34an n4of alfunded amount bnzba llw2t
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['funded_amount'], label='funded_amount', color='blue')
plt.title('funded_amount Time Series')
plt.xlabel('Date')
plt.ylabel('funded_amount')
plt.legend(loc='best')
plt.grid(True)
plt.show()

############################################################################
# Step 2: Stationarize the series
#m3naha an almean w al standard deviation m4 bit8iro m3 alzman
def stationarize_series(series):
    # Calculate rolling statistics
    #bn7sb al mean w al standard deviation 34an n4of fy t8iir kbir wla la
    rolling_mean = series.rolling(window=12).mean()
    rolling_std = series.rolling(window=12).std()
#b3rd al al mean w al standard deviation
    # Plot rolling statistics
    plt.figure(figsize=(10, 6))
    plt.plot(series, label='Original', color='blue')
    plt.plot(rolling_mean, label='Rolling Mean', color='red')
    plt.plot(rolling_std, label='Rolling Std', color='green')
    plt.title('Rolling Mean & Standard Deviation')
    plt.xlabel('Date')
    plt.ylabel('funded_amount')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
#m4 fahmo
    ///////////////////
    # Perform Dickey-Fuller test
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))


# Apply stationarize_series function
stationarize_series(df['funded_amount'])

# Finding the value of the d parameter
plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})
# Original Series
fig, (ax1) = plt.subplots(1)
ax1.plot(df.funded_amount)
ax1.set_title('Original Series')
ax1.axes.xaxis.set_visible(False)
plt.show()
/////////////////////////
# Step 3: plot ACF & PACF
# AFC btsa3d 3la t7did kimt al q fy al ARIMA
# PAFC btsa3d 3la t7did kimt al p fy al ARIMA

fig, (ax1) = plt.subplots(1)
plot_acf(df.funded_amount, ax=ax1)

plot_pacf(df.funded_amount.dropna())

plot_acf(df.funded_amount.dropna())
# b2sm aldata  test 20% train 80% tani
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]
# ARIMA bikon leha 3 7gat bt7tghom
# p 3dd alkim ali fatt total (lag)
# q 3dd alkim ali fatt ali 8lt fiha
#d 3dd mrat aldifferencing
model = ARIMA(train_data['funded_amount'], order=(1, 1, 1))
fitted_model = model.fit()
#alpredicted values bona2n 3n altraining
predicted = fitted_model.forecast(steps=len(test_data))
plt.figure(figsize=(14, 7))
#brsm alreal values w alpredicted values fnfs alw2t
plt.plot(train_data.index, train_data["funded_amount"], label='Train', color='#203147')
plt.plot(test_data.index, test_data["funded_amount"], label='Test', color='#01ef63')
plt.plot(test_data.index, predicted, label='Forecast', color='orange')
plt.ylabel('Close Price')
plt.xlabel('Date')
plt.title('Funded amount Forecast')
plt.legend()
plt.show()
#pprint al BIC w al AIC (m3aiir almokrna mbin alnmazg al772i2i w alprecectied )
#kol ma 2alo kol ma kan a7sn
print(f"AIC: {fitted_model.aic}")
print(f"BIC: {fitted_model.bic}")