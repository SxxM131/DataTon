"""표 형태 모델 비교 (lightgbm n_jobs=4, 공통 fold): 센서 단독 / 이미지(손피처) 단독 / 결합."""
import sys
import numpy as np
import pandas as pd
import imglib as L
import handfeat
from imglib import cv

pool, evalp = L.load_all()
Xs_fit, Xs_eval = L.sensor_features(pool, evalp)
H_fit, H_eval = handfeat.get(pool, evalp)
y = pool.label.astype(int).to_numpy()
color_cols = [c for c in Xs_fit.columns if c.startswith('pt_') or c == 'white_frac']
print('sensor+color feats', Xs_fit.shape, 'hand', H_fit.shape, flush=True)


def runner(A_fit, A_eval):
    def fp(tr, va, f):
        m = L.lgbm().fit(A_fit[tr], y[tr])
        return m.predict(A_eval[va])
    return fp


cfgs = {
    'img_tab_sensor_only': (Xs_fit.drop(columns=color_cols), Xs_eval.drop(columns=color_cols)),
    'img_tab_sensor_color': (Xs_fit, Xs_eval),
    'img_tab_hand_only': (pd.concat([Xs_fit[color_cols], H_fit], axis=1), pd.concat([Xs_eval[color_cols], H_eval], axis=1)),
    'img_tab_sensor_color_hand': (pd.concat([Xs_fit, H_fit], axis=1), pd.concat([Xs_eval, H_eval], axis=1)),
}
oofs = {}
for name, (a, b) in cfgs.items():
    _, oofs[name] = L.run_cv(name, pool, evalp, runner(a.to_numpy(np.float32), b.to_numpy(np.float32)))
L.cmp(evalp, oofs['img_tab_sensor_color'], oofs['img_tab_sensor_only'], 'sensor+color', 'sensor_only')
L.cmp(evalp, oofs['img_tab_sensor_color_hand'], oofs['img_tab_sensor_only'], 'sensor+color+hand', 'sensor_only')
L.cmp(evalp, oofs['img_tab_sensor_color_hand'], oofs['img_tab_sensor_color'], 'sensor+color+hand', 'sensor+color')
