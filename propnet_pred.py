import os
import sys
import inspect
import joblib
from sklearn.preprocessing import MinMaxScaler
sys.path.append('D:/AnacondaProject/3d/pinn')
from propnet import NavierStokesPINN2, NavierStokesPINNLoss2
from propnet_load_cfddem import initial_value_load
import torch.nn as nn
import torch
import numpy as np
import scipy
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import matplotlib as mpl
from scipy.interpolate import griddata
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load flattened cynlinder wake data
# x_all, y_all, t_all, u_all, v_all, p_all, (N, T) = load_cylinder_wake() # (NT, 1)
fpath = 'E:/propnet-data/con_data_arrays.npz'
x_all, y_all, z_all, t_all, pu_all, pv_all, pw_all, fu_all, fv_all, fw_all, fvol_all, p_all, (N, T) = initial_value_load(fpath)# (NT, 1)
# print(x_all.min(),x_all.max()==1)
# print(y_all.min(),y_all.max())
# print(z_all.min(),z_all.max())
# Separate N and T dimensions
XX = x_all.reshape(N, T)
YY = y_all.reshape(N, T)
ZZ = z_all.reshape(N, T)
TT = t_all.reshape(N, T)

PU = pu_all.reshape(N, T)
PV = pv_all.reshape(N, T)
PW = pw_all.reshape(N, T)

FU = fu_all.reshape(N, T)
FV = fv_all.reshape(N, T)
FW = fw_all.reshape(N, T)

FL = fvol_all.reshape(N, T)
PP = p_all.reshape(N, T)
# print(N, T)
# UU = u_all.reshape(N, T)
# VV = v_all.reshape(N, T)
# tt =1
# print(XX[:, [tt]])
# For example, XX[:, [0]].reshape(50, 100) is the the grid of x-coordinates at the first timestep
# (50, 100) is the size of the original simulation grid

def load_saved_model(num_layers, hidden_size, epochs, model, train_selection):
    """ 
    Params:
    hidden_size - int, # of hidden units for each neural network layer
    num_layers - int, # of neural network layers
    epochs - int, # of training epochs
    model - int, whether to use model 1 (Raissi 2019) or model 2 (continuity PDE)
    train_selection - data selection scheme at training time:
                    float, frac of all data (N*T) to selected for training OR 
                    'BC', selected the boundary conditions for training (all timesteps)
    """
    nu = 0.001/1000 #1mPa·s / 1000kg/m3 0.001/1000
    rho = 1000 #1000kg/m3

    # Instantiate model, load saved state_dict
    torch.manual_seed(0)
    if model ==2:
        
        PINN_model = NavierStokesPINN2(hidden_size=hidden_size, num_layers=num_layers, nu=nu,rho=rho)
    file_path = f'E:/propnet-data/well-trained.pt'
    PINN_model.load_state_dict(torch.load(file_path, weights_only=True))
    PINN_model.to(device)
    PINN_model.eval() # Set model to evaluation mode
    return PINN_model

def predict_time_t(model, t):
    """ 
    Evaluate the model at time t, i.e., at TT[:, [t]]
    Params:
    model - PyTorch model to evaluate
    t - int between 0 and T-1, time index to evaluate
    Return:
    np.array() of shape (N, )
    """
    # Get predictions timestep t
    # model_output = model(XX[:, [t]], YY[:, [t]], ZZ[:, [t]], TT[:, [t]])
    model_output = model(XX[:, [t]], YY[:, [t]], ZZ[:, [t]], TT[:, [t]])
    
    pu_pred, pv_pred, pw_pred,fu_pred, fv_pred, fw_pred, p_pred, fvol_pred = model_output[0],\
        model_output[1], model_output[2], model_output[3], model_output[4], model_output[5], model_output[6], model_output[7]
    
    # u_pred, v_pred, p_pred = model_output[0], model_output[1], model_output[2]
    
    # Convert to numpy and flatten
    pu_pred, pv_pred, pw_pred,fu_pred, fv_pred, fw_pred, p_pred, fvol_pred = (
                            pu_pred.detach().cpu().numpy().flatten(),
                            pv_pred.detach().cpu().numpy().flatten(),
                            pw_pred.detach().cpu().numpy().flatten(),
                            fu_pred.detach().cpu().numpy().flatten(),
                            fv_pred.detach().cpu().numpy().flatten(),
                            fw_pred.detach().cpu().numpy().flatten(),
                            p_pred.detach().cpu().numpy().flatten(),
                            fvol_pred.detach().cpu().numpy().flatten())
    return pu_pred, pv_pred, pw_pred, fu_pred, fv_pred, fw_pred, p_pred, fvol_pred

def model_eva(pred_value, real_value, xloc, zloc, stepnum=2, var_name='fu'):
    # 读取scaler
    scaler = joblib.load('E:/propnet-data/scaler.pkl')
    prefix_to_index = {
        'x': 0,
        'y': 1,
        'z': 2,   
        #no t, t is input
        'pu': 4,  
        'pv': 5,  
        'pw': 6,   
        'fu': 7,
        'fv': 8,
        'fw': 9,
        'fvol':10,
        'p': 11,
    }
    if var_name not in prefix_to_index:
        raise ValueError(f"unknown name: {var_name}")
    
    # 
    min_x = scaler.data_min_[0]  # 
    max_x = scaler.data_max_[0]
    min_z = scaler.data_min_[2]  # 2
    max_z = scaler.data_max_[2]
    min_val = scaler.data_min_[prefix_to_index[var_name]]
    max_val = scaler.data_max_[prefix_to_index[var_name]]
    
    # 进行反归一化
    pred_value_actual = pred_value * (max_val - min_val) + min_val
    real_value_actual = real_value * (max_val - min_val) + min_val
    
    x_value_actual = xloc * (max_x - min_x) + min_x# 
    z_value_actual = zloc * (max_z - min_z) + min_z# 
    
    pred_value_actual_step = pred_value_actual
    real_value_actual_step = real_value_actual.reshape(N,T)[:, int(stepnum)]
    
    fig = plt.figure(figsize=(15,5), dpi=150)
    ax = fig.add_subplot(111)
    ax.set_xticks([0,0.1,0.2])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks([0,0.025,0.05])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlabel("Normalized X")
    ax.set_ylabel("Normalized Z")
    ax.set_title("True proppant volume fraction", fontsize=12)
    # ax.tick_params(labelbottom=False, labelleft=False)    
    xx = x_value_actual.reshape(N,T)
    zz = z_value_actual.reshape(N,T)
    x = xx[:,0]
    z = zz[:,0]
    # print(x.shape)
    xi = np.linspace(x.min(), x.max(), 101)
    zi = np.linspace(z.min(), z.max(), 26)
    # print(len(xi))
    xi, zi = np.meshgrid(xi, zi)

    # 
    vi = griddata((x, z), pred_value_actual_step, (xi, zi), method='nearest')
    contour = ax.contourf(xi, zi, 1-vi, 30, cmap='jet') #----
    # contour = ax.contourf(xi, zi, vi, 30, cmap='jet') 
    norm = mpl.colors.Normalize(vmin=real_value_actual_step.min(), vmax=real_value_actual_step.max())
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax,pad=0.01)  # 添加颜色条到图形上
    cbar.set_ticks(np.linspace(real_value_actual_step.min(), real_value_actual_step.max(), 4))  
    cbar.set_label("Proppant volume fraction")
    
    fig2 = plt.figure(figsize=(15,5), dpi=150)
    ax2 = fig2.add_subplot(111)
    ax2.set_xticks([0,0.1,0.2])
    ax2.set_xticklabels(["0", "0.5", "1"])
    ax2.set_yticks([0,0.025,0.05])
    ax2.set_yticklabels(["0", "0.5", "1"])
    # ax2.tick_params(labelbottom=False, labelleft=False)
    # 坐标轴标签
    ax2.set_xlabel("Normalized X")
    ax2.set_ylabel("Normalized Z")
    ax2.set_title("Predicted proppant volume fraction", fontsize=12)
    vi_real = griddata((x, z), real_value_actual_step, (xi, zi), method='nearest')
    contour = ax2.contourf(xi, zi, 1-vi_real, 30, cmap='jet') #----
    # contour = ax2.contourf(xi, zi, vi_real, 30, cmap='jet') 
    norm = mpl.colors.Normalize(vmin=real_value_actual_step.min(), vmax=real_value_actual_step.max())
    cbar = fig2.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax2,pad=0.01)  
    cbar.set_ticks(np.linspace(real_value_actual_step.min(), real_value_actual_step.max(), 4)) 
    cbar.set_label("Proppant volume fraction")
    

    fig3 = plt.figure(figsize=(15,5), dpi=150)
    ax3 = fig3.add_subplot(111)
    ax3.set_xticks([0,0.1,0.2])
    ax3.set_xticklabels(["0", "0.5", "1"])
    ax3.set_yticks([0,0.025,0.05])
    ax3.set_yticklabels(["0", "0.5", "1"])
    ax3.set_xlabel("Normalized X")
    ax3.set_ylabel("Normalized Z")
    ax3.set_title("Absolute error", fontsize=12)
    # ax3.tick_params(labelbottom=False, labelleft=False)
    # print(real_value_actual_step.shape, pred_value_actual_step.shape)
    er = abs(real_value_actual_step - pred_value_actual_step)
    
    print(str('current time step: '+ str(stepnum))+', MAE=',er.mean())
    
    vi_er = griddata((x, z), er, (xi, zi), method='nearest')
    contour = ax3.contourf(xi, zi, vi_er, 30, cmap='jet') 
    norm = mpl.colors.Normalize(vmin=er.min(), vmax=er.max())
    #, format = mpl.ticker.FuncFormatter(lambda x, pos: '')
    cbar = fig3.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax3,pad=0.01)
    cbar.set_ticks(np.linspace(er.min(), er.max(), 4)) 
    cbar.set_label("Absolute error")
    
    filename1 = ax.get_title() +' at time step-'+ str(stepnum)+ ".png"
    fig.savefig(os.path.join('E:/propnet-data/', filename1), bbox_inches='tight')
    print(f"Saved: {filename1}")
    
    filename2 = ax2.get_title() +' at time step-'+ str(stepnum)+ ".png"
    fig2.savefig(os.path.join('E:/propnet-data/', filename2), bbox_inches='tight')
    print(f"Saved: {filename2}")
    
    filename3 = ax3.get_title() +' at time step-'+ str(stepnum)+ ".png"
    fig3.savefig(os.path.join('E:/propnet-data/', filename3), bbox_inches='tight')
    print(f"Saved: {filename3}")

    

model_read = load_saved_model(num_layers=8, hidden_size=48, epochs=600, model=2, train_selection=0.6)
pu_pred, pv_pred, pw_pred, fu_pred, fv_pred, fw_pred, p_pred, fvol_pred = predict_time_t(model=model_read, t=70)
model_eva(fvol_pred, fvol_all, xloc=x_all, zloc=z_all, stepnum=70, var_name='fvol')#第1 2 和最后一个参数为你要验证的量
plt.tight_layout()
plt.show()