
from datetime import datetime
import numpy as np
import pandas as pd

import statsmodels.api as stm

from multiprocessing import Pool

if __name__ == "__main__":
    t_start = datetime.now()

    P_start = '200001'; P_end = '202212' ## for example, chirps
    E_start = '200001'; E_end = '202212' ## for example, gleam
    R_start = '200001'; R_end = '201412' ## for example, grun
    S_start = '200001'; S_end = '201912' ## for example, somo

    P_values = np.load('chirps_mn.npy')
    E_values = np.load('gleam_mn.npy')
    R_values = np.load('grun_mn.npy')
    S_values = np.load('somo_mn.npy')

    ds1 = pd.DataFrame(pd.date_range(P_start+'01', P_end+'01', freq='MS'), index=np.arange(P_values.shape[0]), columns=['month'])
    ds2 = pd.DataFrame(pd.date_range(E_start+'01', E_end+'01', freq='MS'), index=np.arange(E_values.shape[0]), columns=['month'])
    ds3 = pd.DataFrame(pd.date_range(R_start+'01', R_end+'01', freq='MS'), index=np.arange(R_values.shape[0]), columns=['month'])
    ds4 = pd.DataFrame(pd.date_range(S_start+'01', S_end+'01', freq='MS'), index=np.arange(S_values.shape[0]), columns=['month'])
    
    time_serie1 = pd.date_range(P_start+'01', P_end+'01', freq='MS')
    time_serie2 = pd.date_range(E_start+'01', E_end+'01', freq='MS')
    time_serie3 = pd.date_range(R_start+'01', R_end+'01', freq='MS')
    time_serie4 = pd.date_range(S_start+'01', S_end+'01', freq='MS')
    time_serie_temp = time_serie1.intersection(time_serie2)
    time_serie_temp = time_serie_temp.intersection(time_serie3)
    time_serie_temp = time_serie_temp.intersection(time_serie4)
    assert time_serie_temp.shape[0]>0
    print(time_serie_temp)
    
    index1 = ds1.loc[ds1['month'].isin(time_serie_temp)].index.values
    index2 = ds2.loc[ds2['month'].isin(time_serie_temp)].index.values
    index3 = ds3.loc[ds3['month'].isin(time_serie_temp)].index.values
    index4 = ds4.loc[ds4['month'].isin(time_serie_temp)].index.values
    print(index1, index2, index3)
    equation_left = np.full((len(time_serie_temp),720,1440), np.nan)
    for mn, (i1, i2, i3) in enumerate(zip(index1, index2, index3)):
        equation_left[mn] = P_values[i1] - E_values[i2] - R_values[i3]
    equation_right = S_values[index4]

    R_squared = np.full((720,1440),np.nan)
    for la in range(equation_right.shape[1]):
        for lo in range(equation_right.shape[2]):

            equation_mask = (~np.isnan(equation_right[:,la,lo]))&(~np.isnan(equation_left[:,la,lo]))
            if (np.sum(equation_mask)==0):
                continue
            elif (np.sum(equation_mask)<36):
                # print(la, lo, 'has low available values')
                continue
            else:
                equation_right_array = equation_right[:,la,lo][equation_mask]
                equation_left_array = equation_left[:,la,lo][equation_mask]
                
                X = pd.DataFrame({'left': equation_left_array})
                y = pd.DataFrame({'right': equation_right_array})

                model = stm.OLS(y, X).fit()
                R_squared[la,lo] = model.rsquared_adj

    np.save('0_example_name_combination.npy', [('chirps_p','gleam_et','grun_r','somo_sm')])
    np.save('0_example_Rsquare.npy', R_squared)

    print('Finished in', datetime.now(), 'cost', datetime.now()-t_start)
#EOF