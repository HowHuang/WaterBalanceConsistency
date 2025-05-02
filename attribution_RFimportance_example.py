from datetime import datetime
import numpy as np

from multiprocessing import Pool

from sklearn.ensemble import RandomForestRegressor
import shap

if __name__ == '__main__':
    t_start = datetime.now()

    sc = np.load('soilclay.npy')
    EP = np.load('aridity_index.npy')
    TC = np.load('tree_cover.npy')
    IA = np.load('irrigation_area.npy')
    aT = np.load('average_T.npy')
    sd = np.load('surround_density.npy')
    tp = np.load('topography.npy')
    r2 = np.load('overall_performance_gleam_et.npy')

    mask_dt = (~np.isnan(sc))&(~np.isnan(EP))&(~np.isnan(TC))&(~np.isnan(IA))&(~np.isnan(aT))&(~np.isnan(sd))&(~np.isnan(tp))&(~np.isnan(r2))

    print('mask_dt', mask_dt.shape, np.sum(mask_dt))
    data_in = np.hstack([sc[mask_dt].reshape(-1,1),EP[mask_dt].reshape(-1,1),
                         TC[mask_dt].reshape(-1,1),IA[mask_dt].reshape(-1,1),
                         aT[mask_dt].reshape(-1,1),sd[mask_dt].reshape(-1,1),
                         tp[mask_dt].reshape(-1,1)])
        
    data_in = np.hstack([data_in,r2[mask_dt].reshape(-1,1)])
    print('the shape of shape data_in', data_in.shape)

    '''refer to Li et al. Nature Communications 2022'''
    rf = RandomForestRegressor(n_estimators=100,
                               max_features=0.3,
                               n_jobs=1,
                               bootstrap=True,
                               oob_score=True,
                               random_state=42)
    rf.fit(data_in[:, :-1], data_in[:, -1])
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(data_in[:, :-1])
    
    np.save('oob_scores_gleam_et.npy', [rf.oob_score_])
    np.save('shap_values_gleam_et.npy', shap_values)
    
    t_end = datetime.now()
    print('finished at', t_end, 'cost', t_end-t_start)
##EOF