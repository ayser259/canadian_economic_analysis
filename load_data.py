import numpy as np
import pandas as pd
from sklearn import preprocessing

def load_data():
    gdp = pd.read_csv('data/gdp.csv',dtype=str)
    gdp = gdp.T
    gdp.columns = gdp.iloc[0]
    gdp = gdp[1:]
    gdp = gdp[['GDP']]

    wages = pd.read_csv('data/wages.csv',dtype=str)
    wages = wages.T
    wages.columns = wages.iloc[0]
    wages = wages[1:]
    wages.set_index(wages.columns[0])
    wages.columns = ['wages -'+ str(col) for col in wages.columns]

    cap_markets = pd.read_csv('data/capital_markets.csv',dtype=str)
    cap_markets = cap_markets.T
    cap_markets.columns = cap_markets.iloc[0]
    cap_markets = cap_markets[1:]
    cap_markets = cap_markets[['Gross new issues','Retirements','Net new issues']]
    cap_markets.set_index(cap_markets.columns[0])
    cap_markets.columns = ['cap_markets -'+ str(col) for col in cap_markets.columns]

    security_flows = pd.read_csv('data/security_net_flows.csv',dtype=str)
    security_flows = security_flows.T
    security_flows.columns = security_flows.iloc[0]
    security_flows = security_flows[1:]
    security_flows.set_index(security_flows.columns[0])
    security_flows.columns = ['securities -'+ str(col) for col in security_flows.columns]

    electricity = pd.read_csv('data/electricity_generation.csv',dtype=str)
    electricity = electricity.T
    electricity.columns = electricity.iloc[0]
    electricity = electricity[1:]
    electricity.set_index(electricity.columns[0])
    electricity.columns = ['electricity -'+ str(col) for col in electricity.columns]

    data = (gdp.merge(electricity,left_index= True,right_index=True,how='left'))
    data = (data.merge(cap_markets,left_index= True,right_index=True,how='left'))
    data = (data.merge(security_flows,left_index= True,right_index=True,how='left'))
    data = (data.merge(wages,left_index= True,right_index=True,how='left'))

    for col in list(data.columns):
        data[col] = data[col].str.replace(',', '')
        data[col] = data[col].str.replace('r', '')
        data[col] = data[col].astype(int)

    return data

def load_normalized_data():
    data = load_data()
    x = data.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    df.columns = data.columns
    df.index = data.index
    return df
