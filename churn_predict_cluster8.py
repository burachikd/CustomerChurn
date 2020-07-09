## Packages and libraries

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Masking
from keras.layers import BatchNormalization
from keras.layers import Dropout

from keras import backend as k
from keras import callbacks

from sklearn.preprocessing import normalize
from sklearn import pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd
import os
import math

from six.moves import xrange
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.layers import LSTM,GRU
from keras.layers import Lambda
from keras.layers.wrappers import TimeDistributed

from keras.optimizers import RMSprop,adam
from keras.callbacks import History

import wtte.weibull as weibull
import wtte.wtte as wtte

from wtte.wtte import WeightWatcher
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix

import tqdm
from tqdm import tqdm

import keras.backend as K

from pickle import dump,load
from datetime import date

import warnings
warnings.filterwarnings('ignore')

np.random.seed(0)


K.set_epsilon(1e-10)
today = date.today()
models = os.path.join(os.getcwd(), 'Churn', 'models', 'Cluster8')
cluster = 8

print(models)

file = 'test_data_may2020.csv'
path_predict_test = os.path.join(os.getcwd(), 'Churn', 'data', 'Cluster8', file)

treshhold = 45
# # Configurable observation look-back period for each user/day
max_time = 100
mask_value = -99
# event probability
probability = .3
churn_month_start = '2020-06-01'
churn_month_end = '2020-06-30'

"""
    Discrete log-likelihood for Weibull hazard function on censored survival data
    y_true is a (samples, 2) tensor containing time-to-event (y), and an event indicator (u)
    ab_pred is a (samples, 2) tensor containing predicted Weibull alpha (a) and beta (b) parameters
    For math, see https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf (Page 35)
"""
def weibull_loglik_discrete(y_true, ab_pred, name=None):
    y_ = y_true[:, 0]
    u_ = y_true[:, 1]
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]

    hazard0 = k.pow((y_ + 1e-35) / a_, b_)
    hazard1 = k.pow((y_ + 1) / a_, b_)

    return -1 * k.mean(u_ * k.log(k.exp(hazard1 - hazard0) - 1.0) - hazard1)

"""
    Not used for this model, but included in case somebody needs it
    For math, see https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf (Page 35)
"""


def weibull_loglik_continuous(y_true, ab_pred, name=None):
    y_ = y_true[:, 0]
    u_ = y_true[:, 1]
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]

    ya = (y_ + 1e-35) / a_
    return -1 * k.mean(u_ * (k.log(b_) + b_ * k.log(ya)) - k.pow(ya, b_))


"""
    Custom Keras activation function, outputs alpha neuron using exponentiation and beta using softplus
"""


def activate(ab):
    a = k.exp(ab[:, 0])
    b = k.softplus(ab[:, 1])

    a = k.reshape(a, (k.shape(a)[0], 1))
    b = k.reshape(b, (k.shape(b)[0], 1))

    return k.concatenate((a, b), axis=1)

'''for targer calculaction from alpha and beta '''


def weibull_median(alpha, beta):
    return alpha*(-np.log(probability))**(1/beta)


def weibull_mean(alpha, beta):
    return alpha * math.gamma(1 + 1/beta)


## FUNCTIONS

def build_test_data(path, churn_treshhold):
    data = pd.read_csv(path)
    data = data.copy()
    churn_counts = data[data['running_activity'] == churn_treshhold]['ClientID'].value_counts()
    churn_counts = pd.DataFrame(churn_counts)
    churn_counts = churn_counts.rename(columns={'ClientID': 'Churn_counts'})
    churn_counts = churn_counts.reset_index()

    data = pd.merge(data, churn_counts, left_on='ClientID', right_on='index', how='left')
    data['Churn_counts'] = data['Churn_counts'].fillna(0)

    nov_churners = data[
        (data['running_activity'] == churn_treshhold) & (data['CreatedOn'].between(churn_month_start,
                                                                                   churn_month_end)) & (
                    data['Churn_counts'] < 2)]

    cl_grouped = nov_churners.groupby(['ClientID'])['ClientNumber'].count()
    cl_grouped = pd.DataFrame(cl_grouped)
    cl_grouped = cl_grouped.rename(columns={'ClientNumber': 'Real_Churn'})
    cl_grouped = cl_grouped.reset_index()
    data1 = pd.merge(data, cl_grouped, left_on='ClientID', right_on='ClientID', how='left')
    data1['Real_Churn'] = data1['Real_Churn'].fillna(0)

    df_todrop = data1[data1['running_activity'] == churn_treshhold][['ClientID', 'rnk']]
    df_drop = df_todrop.groupby('ClientID')['rnk'].min()
    df_min = pd.DataFrame(df_drop)
    df_min = df_min.rename(columns={'rnk': 'rnk_min'})
    df_min = df_min.reset_index()
    data1 = pd.merge(data1, df_min, left_on='ClientID', right_on='ClientID', how='left')
    idx_to_drop = data1[data1['rnk'] >= data1['rnk_min']].index
    train = data1.drop(idx_to_drop)

    churn_before = train[(train['Churn'] == 1) & (train['Real_Churn'] == 0)]['ClientID'].unique()
    churn_before = pd.DataFrame(churn_before, columns=['ClientID'])

    train1 = train[~((train['Churn'] == 1) & (train['Real_Churn'] == 0))]

    replace_list = {}
    for k, v in enumerate(train1['ClientNumber'].unique()):
        replace_list[v] = k + 1
    train1['ClientNumber'] = train1['ClientNumber'].map(replace_list)
    train1['rnk_min'] = train1['rnk_min'].fillna(0)

    df_drop2 = train1.groupby(['ClientID'])['rnk_min'].min()
    df_drop2 = df_drop2.apply(lambda x: 37 if x == 0 else x - 183)
    df_drop2 = pd.DataFrame(df_drop2)
    df_drop2 = df_drop2.reset_index()
    test_y = df_drop2[['ClientID', 'rnk_min']]

    train1 = train1.drop(train1[train1['CreatedOn'] >= churn_month_start].index)  # delete observed month
    train1 = train1.drop(
        columns=['CreatedOn', 'ClientID', 'rnk_min', 'Real_Churn', 'index', 'Churn_counts'])  # ,'rnk_max'])
    train1 = train1.rename(columns={'ClientNumber': 'user_number', 'rnk': 'time'})
    train1.set_index(['user_number', 'time'], verify_integrity=True)

    return churn_before, train1, test_y


def scaler_load(test):
    scaler = load(open(os.path.join(models, '2020-05-14_segment_8.pkl'), 'rb'))
    all_data = np.concatenate([test[['user_number', 'time']],
                               scaler.fit_transform(test[feature_cols]), test[churn].values.reshape(-1, 1)], axis=1)

    # then split them back out
    test = all_data.copy()
    # Make engine numbers and days zero-indexed, for everybody's sanity
    test[:, 0:2] -= 1
    return test


def build_data(engine, time, x, churn, max_time, is_test, mask_value):
    # y[0] will be days remaining, y[1] will be event indicator, always 1 for this data
    out_y = []

    # number of features
    d = x.shape[1]

    # A full history of sensor readings to date for each x
    out_x = []

    n_engines = len(np.unique(engine))

    for i in tqdm(range(n_engines)):
        # When did the engine fail? (Last day + 1 for train data, irrelevant for test.)
        max_engine_time = int(np.max(time[engine == i])) + 1
        churn_marker = int(np.mean(churn[engine == i]))
        if is_test:
            start = max_engine_time - 1
        else:
            start = 0

        this_x = []

        for j in range(start, max_engine_time):
            engine_x = x[engine == i]

            out_y.append(np.array((max_engine_time - j, churn_marker), ndmin=2))

            xtemp = np.zeros((1, max_time, d))
            xtemp += mask_value
            #             xtemp = np.full((1, max_time, d), mask_value)

            xtemp[:, max_time - min(j, max_time - 1) - 1:max_time, :] = engine_x[max(0, j - max_time + 1):j + 1, :]
            this_x.append(xtemp)

        this_x = np.concatenate(this_x)
        out_x.append(this_x)
    out_x = np.concatenate(out_x)
    out_y = np.concatenate(out_y)
    return out_x, out_y


def create_model_and_load_weights(train_x):
    history = History()
    weightwatcher = WeightWatcher()
    nanterminator = callbacks.TerminateOnNaN()
    n_features = train_x.shape[-1]
    # Start building our model
    model = Sequential()
    # Mask parts of the lookback period that are all zeros (i.e., unobserved) so they don't skew the model
    model.add(Masking(mask_value=mask_value, input_shape=(None, n_features)))
    model.add(LSTM(24, activation='tanh', recurrent_dropout=0.25))
    #     model.add(Dropout(0.5))
    #     model.add(BatchNormalization())
    #     model.add(LSTM(2, activation='relu', return_sequences=False))
    #     model.add(Dropout(0.5))
    #     model.add(BatchNormalization())
    model.add(Dense(2))
    model.add(Lambda(wtte.output_lambda,
                     arguments={"init_alpha": 23,
                                "max_beta_value": 1.9,
                                "alpha_kernel_scalefactor": 0.5
                                },
                     ))
    # Use the discrete log-likelihood for Weibull survival data as our loss function
    loss = wtte.loss(kind='discrete', reduce_loss=False).loss_function

    model.compile(loss=loss, optimizer=adam(lr=.001, clipvalue=0.5, beta_1=0.91, beta_2=0.999))
    model.load_weights(os.path.join(models, '2020-05-14_segment_8_LSTM24.h5'.format(today, cluster)))
    return model


churn_before, test_x, test_y = build_test_data(path_predict_test, treshhold)

feature_cols = test_x.columns[2:-1]
churn = test_x.columns[-1]
test2 = scaler_load(test_x)

test_x2, _ = build_data(engine=test2[:, 0], time=test2[:, 1], x=test2[:, 2:-1], churn=test2[:, -1],
                        max_time=max_time, is_test=True, mask_value=mask_value)

test_y2 = test_y.copy()
test_y2 = pd.DataFrame(test_y2)
test_y2 = test_y2.rename(columns={'rnk': 'T'})
test_y2['E'] = 1

model = create_model_and_load_weights(test_x2)

test_predict2 = model.predict(test_x2)
test_predict2 = np.resize(test_predict2, (test_predict2.shape[0], 2))
test_result2 = np.concatenate((test_y2, test_predict2), axis=1)

test_results_df2 = pd.DataFrame(test_result2, columns=['ClientID','T', 'E', 'alpha', 'beta'])
test_results_df2['predicted_median'] = test_results_df2[['alpha', 'beta']].apply(
    lambda row: weibull_median(row[0], row[1]), axis=1)
test_results_df2['error'] = test_results_df2['T']-test_results_df2['predicted_median']
test_results_df2['error_abs'] = np.abs(test_results_df2['T']-test_results_df2['predicted_median'])

test_results_df2.to_excel(os.path.join(os.getcwd(), 'Churn', 'output', 'Cluster8', 'potential_churn_cluster8_' +
                                       churn_month_start + '.xlsx'))
churn_before.to_excel(os.path.join(os.getcwd(), 'Churn', 'output', 'Cluster8', 'churn_before_cluster8_' +
                                   churn_month_start + '.xlsx'))







