from datetime import datetime
import numpy as np
import pandas as pd

import statsmodels.api as stm

from scipy.stats import bootstrap as sbootstrap

def cal_R2_gird(index):
    equation_left_array_re = equation_left_array[index]
    equation_right_array_re = equation_right_array[index]

    X = pd.DataFrame({'left': equation_left_array_re})
    y = pd.DataFrame({'right': equation_right_array_re})

    model = stm.OLS(y, X).fit()
    R_squared_grid = model.rsquared_adj
    return R_squared_grid

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
    
    first_ten_overlap = pd.date_range('2000-01-01','2010-12-01', freq='MS').intersection(time_serie_temp)
    secondten_overlap = pd.date_range('2011-01-01','2022-12-01', freq='MS').intersection(time_serie_temp)
    two_periods_ci = []
    if (len(first_ten_overlap)>=11*12*2/3)and(len(secondten_overlap)>=12*12*2/3): ## 88 and 96 months
        for time_serie_overlap in [first_ten_overlap, secondten_overlap]:
            index1 = ds1.loc[ds1['month'].isin(time_serie_overlap)].index.values
            index2 = ds2.loc[ds2['month'].isin(time_serie_overlap)].index.values
            index3 = ds3.loc[ds3['month'].isin(time_serie_overlap)].index.values
            index4 = ds4.loc[ds4['month'].isin(time_serie_overlap)].index.values
            print(index1.shape, index2.shape, index3.shape)

            equation_left = np.full((len(time_serie_overlap),720,1440), np.nan)
            for mn, (i1, i2, i3) in enumerate(zip(index1, index2, index3)):
                equation_left[mn] = P_values[i1] - E_values[i2] - R_values[i3]
            equation_right = S_values_dict[which_S][index4].values
            
            R_ci_eachPeriod = np.full((2,equation_right.shape[1],equation_right.shape[2]),np.nan)
            for la in range(equation_right.shape[1]):
                for lo in range(equation_right.shape[2]):
                    equation_mask = (~np.isnan(equation_right[:,la,lo]))&(~np.isnan(equation_left[:,la,lo]))
                    if (np.sum(equation_mask)==0):
                        continue
                    elif (np.sum(equation_mask)<36):
                        # print(independent_combinations[i], la, lo, 'has low available values')
                        continue
                    else:
                        equation_right_array = equation_right[:,la,lo][equation_mask]
                        equation_left_array = equation_left[:,la,lo][equation_mask]
                        equation_index = np.arange(len(equation_right_array))
            
                        re_bst = sbootstrap(equation_index[np.newaxis,:], cal_R2_gird, n_resamples=100, #batch=1,
                                            confidence_level=0.95, alternative='two-sided', method='percentile', random_state=42)
                        
                        R_ci_eachPeriod[0,la,lo] = re_bst.confidence_interval[0]
                        R_ci_eachPeriod[1,la,lo] = re_bst.confidence_interval[1]
            
            assert R_ci_eachPeriod.shape == (2,720,1440)
            two_periods_ci.append(R_ci_eachPeriod)
        two_periods_ci = np.concatenate(two_periods_ci,axis=0)
    else:
        print(i, independent_combinations[i], 'has less than 2/3 months overlap for either first or second periods')
        two_periods_ci = np.array([-999])

    np.save('two_periods_ci.npy', two_periods_ci)

    print('Finished in', datetime.now(), 'cost', datetime.now()-t_start)
#EOF