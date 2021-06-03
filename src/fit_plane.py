"""
Fit plane to tremor depths
"""
import numpy as np
import pandas as pd
import pickle

from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error

arrays = ['BH', 'BS', 'DR', 'GC', 'PA', 'TB']
#arrays = ['BH', 'BS', 'DR', 'PA', 'TB']

type_stack = 'PWS'
cc_stack = 'PWS'

threshold = 0.005

# Linear regression
for num, array in enumerate(arrays):
    df_temp = pickle.load(open('cc/{}/{}_{}_{}_width_0.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))
    quality = pickle.load(open('cc/{}/quality_{}_{}.pkl'.format( \
        array, type_stack, cc_stack), 'rb'))
    df_temp = df_temp.merge(quality, on=['i', 'j'], how='left', indicator=True)
    if (num == 0):
        df = df_temp
    else:
        df = pd.concat([df, df_temp], ignore_index=True)

df.drop(df[(df.maxE < threshold) & (df.maxN < threshold)].index, inplace=True)
df.reset_index(drop=True, inplace=True)

df.drop(df[df.quality > 2].index, inplace=True)
df.reset_index(drop=True, inplace=True)

error = np.zeros((len(df), 3))

for i in range(0, len(df)):
    error[i, 0] = df['longitude'][i]
    error[i, 1] = df['latitude'][i]
    if df['maxE'][i] > df['maxN'][i]:
        error[i, 2] = df['dist_EW'][i]
    else:
        error[i, 2] = df['dist_NS'][i]

regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(error[:, 0:2], error[:, 2])
prediction = regr.predict(error[:, 0:2])
R2 = r2_score(error[:, 2], prediction)

print(R2)

# Write error and prediction to file
for num, array in enumerate(arrays):
    df = pickle.load(open('cc/{}/{}_{}_{}_width_0.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))
    quality = pickle.load(open('cc/{}/quality_{}_{}.pkl'.format( \
        array, type_stack, cc_stack), 'rb'))
    df = df.merge(quality, on=['i', 'j'], how='left', indicator=True)

    df.drop(df[(df.maxE < threshold) & (df.maxN < threshold)].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.drop(df[df.quality > 2].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    prediction = np.zeros((len(df), 3))
    error = np.zeros((len(df), 3))

    for i in range(0, len(df)):
        prediction[i, 0] = df['longitude'][i]
        prediction[i, 1] = df['latitude'][i]
        error[i, 0] = df['longitude'][i]
        error[i, 1] = df['latitude'][i]
        if df['maxE'][i] > df['maxN'][i]:
            error[i, 2] = df['dist_EW'][i]
        else:
            error[i, 2] = df['dist_NS'][i]

    prediction[:, 2] = regr.predict(prediction[:, 0:2])
    error[:, 2] = prediction[:, 2] - error[:, 2]

    np.savetxt('map_depth/error_{}_{}_{}.txt'.format(type_stack, cc_stack, array), error, fmt='%10.5f')
    np.savetxt('map_depth/prediction_{}_{}_{}.txt'.format(type_stack, cc_stack, array), prediction, fmt='%10.5f')
