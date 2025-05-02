
from datetime import datetime
import numpy as np

import glob

from multiprocessing import Pool

if __name__ == "__main__":
    t_start = datetime.now()

    all_name_combination_f = glob.glob('*_name_combination.npy')
    all_Rsquare_f = glob.glob('*_Rsquare.npy')
    all_name_combination_f.sort(); all_Rsquare_f.sort()
    print(all_name_combination_f)
    print(all_Rsquare_f)

    all_name_combination = []
    for f in all_name_combination_f:
        all_name_combination.append(np.load(f))
    all_Rsquare = []
    for f in all_Rsquare_f:
        all_Rsquare.append(np.load(f))
    all_name_combination = np.array(all_name_combination)
    all_Rsquare = np.array(all_Rsquare)

    DS_example = 'gleam_et'

    performance = []
    for i, c in enumerate(all_name_combination):
        if DS_example in c:
            performance.append(all_Rsquare[i])
    performance = np.array(performance)
    print(DS_example, performance.shape)
    
    overall_performance = np.full(performance.shape[1:],np.nan)
    for la in range(overall_performance.shape[0]):
        for lo in range(overall_performance.shape[1]):
            overall_performance[la,lo] = np.nanmean(performance[:,la,lo])

    np.save('overall_performance_gleam_et.npy', overall_performance)
    np.save('performance_num_gleam_et.npy', performance.shape[0])

    print('Finished in', datetime.now(), 'cost', datetime.now()-t_start)
#EOF
