"""
Fit plane to tremor depths
"""
import numpy as np
import pandas as pd
import pickle

from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error

arrays = ['BH', 'BS', 'CL', 'DR', 'GC', 'LC', 'PA', 'TB']

type_stack = 'PWS'
cc_stack = 'PWS'

threshold = 0.005

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

df.drop(df[df.quality != 1].index, inplace=True)
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

# Linear regression
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(error[:, 0:2], error[:, 2])
prediction[:, 2] = regr.predict(prediction[:, 0:2])
R2 = r2_score(error[:, 2], prediction[:, 2])
error[:, 2] = prediction[:, 2] - error[:, 2]

np.savetxt('map_depth/error_{}_{}.txt'.format(type_stack, cc_stack), error, fmt='%10.5f')
np.savetxt('map_depth/prediction_{}_{}.txt'.format(type_stack, cc_stack), prediction, fmt='%10.5f')

print(R2)
