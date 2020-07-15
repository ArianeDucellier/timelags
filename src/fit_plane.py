"""
Fit plane to tremor depths
"""
import numpy as np
import pandas as pd
import pickle

from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error

arrays = ['BH', 'BS', 'GC', 'PA', 'TB']

type_stack = 'PWS'
cc_stack = 'PWS'

threshold = 4.0

for num, array in enumerate(arrays):
    df_temp = pickle.load(open('cc/{}/{}_{}_{}_width_0.pkl'.format( \
        array, array, type_stack, cc_stack), 'rb'))
    if (num == 0):
        df = df_temp
    else:
        df = pd.concat([df, df_temp], ignore_index=True)

df.drop(df[(df.thick_EW < 0.01) | (df.thick_NS < 0.01)].index, inplace=True)
df.reset_index(drop=True, inplace=True)

uncertainty = np.zeros((len(df), 3))

df.drop(df[(df.thick_EW > threshold) & (df.thick_NS > threshold)].index, inplace=True)
df.reset_index(drop=True, inplace=True)

prediction = np.zeros((len(df), 3))
error = np.zeros((len(df), 3))

for i in range(0, len(df)):
    prediction[i, 0] = df['longitude'][i]
    prediction[i, 1] = df['latitude'][i]
    error[i, 0] = df['longitude'][i]
    error[i, 1] = df['latitude'][i]
    if df['thick_EW'][i] < df['thick_NS'][i]:
        error[i, 2] = df['dist_EW'][i]
    else:
        error[i, 2] = df['dist_NS'][i]

# Linear regression
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(error[:, 0:2], error[:, 2])
prediction[:, 2] = regr.predict(prediction[:, 0:2])
R2 = r2_score(error[:, 2], prediction[:, 2])
error[:, 2] = prediction[:, 2] - error[:, 2]

np.savetxt('map_depth/error_{}_{}_0.txt'.format(type_stack, cc_stack), error, fmt='%10.5f')
np.savetxt('map_depth/prediction_{}_{}_0.txt'.format(type_stack, cc_stack), prediction, fmt='%10.5f')

print(R2)
