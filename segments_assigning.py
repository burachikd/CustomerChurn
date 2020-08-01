import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import os
from datetime import datetime as dt
import sys

filename = 'rfm_data_full_july2020.csv'
segment_dir = 'Segmentation'
data_dir = 'data'
models_dir = 'models'
output_dir = 'output'
path = os.path.join(os.getcwd(), segment_dir, data_dir, filename)
n_clusters = 10

def predict_cluster(path,segment_dir, data_dir,models_dir, outputdir):
    df = pd.read_csv(path, sep=';')
    df['Monetary'].fillna(0.00000000001, inplace=True)
    df['Monetary'].replace(0.0, 0.00000000001, inplace=True)
    df['days_since_last_activity_Recency'].fillna(0.00000000001, inplace=True)
    df['days_since_last_activity_Recency'].replace(0.0, 0.00000000001, inplace=True)
    df['days_since_last_activity_Recency'].replace(0, 0.00000000001, inplace=True)
    df['PaidBack'].fillna(0.00000000001, inplace=True)
    df['PaidBack'].replace(0.0, 0.00000000001, inplace=True)
    df['Days_cycle'].fillna(0.00000000001, inplace=True)
    df['Days_cycle'].replace(0.0, 0.00000000001, inplace=True)
    df['T_days_since_first_loan'].fillna(0.00000000001, inplace=True)
    df['T_days_since_first_loan'].replace(0.0, 0.00000000001, inplace=True)
    df.rename(columns={"days_since_last_activity_Recency": "Recency"}, inplace=True)
    frequency_quartile = pd.qcut(df['Frequency'].rank(method='first'), q=4, labels=range(1, 5))
    recency_quartile = pd.qcut(df['Recency'], q=4, labels=range(1, 5))
    monetary_quartile = pd.qcut(df['Monetary'], q=4, labels=range(1, 5))

    df['Frequency_Quartile'] = frequency_quartile
    df['Recency_Quartile'] = recency_quartile
    df['Monetary_Quartile'] = monetary_quartile

    df['RFM_Segment'] = df[['Frequency_Quartile', 'Recency_Quartile', 'Monetary_Quartile']].astype(str).sum(axis=1)
    df['RFM_Score'] = df[['Frequency_Quartile', 'Recency_Quartile', 'Monetary_Quartile']].sum(axis=1)

    def rfm_level(dataframe):
        if dataframe['RFM_Score'] >= 10:
            return 'Top'
        elif ((dataframe['RFM_Score'] >= 6) and (dataframe['RFM_Score'] < 10)):
            return 'Middle'
        else:
            return 'Low'

    df['RFM_Level'] = df.apply(rfm_level, axis=1)

    rfm_level_agg = df.groupby('RFM_Level').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(1)

    df['Recency_log'] = np.log(df['Recency'])
    df['Frequency_log'] = np.log(df['Frequency'])
    df['Monetary_log'] = np.log(df['Monetary'])

    df['T_days_since_first_loan_log'] = np.log(df['T_days_since_first_loan'])
    df['PaidBack_log'] = np.log(df['PaidBack'])
    df['Days_cycle_log'] = np.log(df['Days_cycle'])

    data = df[['Recency_log', 'Frequency_log', 'Monetary_log', 'T_days_since_first_loan_log', 'PaidBack_log',
               'Days_cycle_log']]

    loaded_scaler = pickle.load(open(os.path.join(os.getcwd(), segment_dir, models_dir, 'segmentation_scaler_ver1'), 'rb'))
    data_normalized = loaded_scaler.transform(data)
    data_normalized = pd.DataFrame(data=data_normalized, index=data.index, columns=data.columns)
    data_normalized.columns = ['Recency', 'Frequency', 'Monetary', 'T_days_since_first_loan', 'PaidBack', 'Days_cycle']
    kmeans = pickle.load(open(os.path.join(os.getcwd(), segment_dir, models_dir, 'segmentation_model_ver1'), 'rb'))

    cluster_labels = kmeans.predict(data_normalized)

    data_rfm_k5 = df.assign(Cluster=cluster_labels)
    # print(data_normalized_k3.head())

    grouped = data_rfm_k5.groupby(['Cluster'])
    stats = grouped.agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'T_days_since_first_loan': 'mean',
        'PaidBack': 'mean',
        'Days_cycle': ['mean', 'count']
    }).round(2)

    data_rfm_k5_ = data_rfm_k5[['Recency', 'Frequency', 'Monetary', 'T_days_since_first_loan', 'PaidBack',
                                'Days_cycle', 'Cluster']]

    df_ = df[['ClientID', 'Recency', 'Frequency', 'Monetary', 'T_days_since_first_loan', 'PaidBack', 'Days_cycle']]

    # Calculate average RFM values for each cluster
    cluster_avg = data_rfm_k5_.groupby(['Cluster']).mean()

    # Calculate average RFM values for the total customer population
    population_avg = df_.mean()

    # Calculate relative importance of cluster's attribute value compared to population
    relative_imp = cluster_avg / population_avg - 1
    plt.figure(figsize=(10, 4))
    plt.title('Relative importance of attributes')
    sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdYlGn')
    plt.show()
    data_rfm_k5.to_excel(os.path.join(os.getcwd(), segment_dir, output_dir, 'CLusters-' + str(dt.today().year) + '-' +
                                      str(dt.today().month) + '.xlsx'))

predict_cluster(path,segment_dir,data_dir,models_dir, output_dir)

