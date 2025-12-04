### TRAIN MODEL ON CLYNDER WAKE DATA ###
import os
import pickle
import sys

from matplotlib import pyplot as plt
sys.path.append('D:/AnacondaProject/3d/pinn')
from propnet import NavierStokesPINN2, NavierStokesPINNLoss2
from propnet_load_cfddem import initial_value_load
import torch.nn as nn
import torch
import numpy as np
import scipy
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_loss_list = []

def boundary_indices(N, T):
    """ 
    Returns a boolean mask marking the boundary condition for all timesteps
    Params:
    N - # of data samples (space locations) in problem
    T - # of timesteps
    Return:
    nd-array of shape (N*T, )
    """
    # Create grid for one timestep
    grid_t_0 = np.zeros((50, 100))

    # Set boundary to 1
    grid_t_0[0, :] = 1
    grid_t_0[:, 0] = 1
    grid_t_0[-1, :] = 1
    grid_t_0[:, -1] = 1

    # Flatten and propagate over T timesteps
    grid_t_all = np.tile(grid_t_0.reshape(-1, 1), (1, T)) # (N, T)
    boundary_positions = grid_t_all.astype(bool).flatten()

    # For example:
    # boundary_positions.reshape(N, T)[:, [t]].reshape(50, 100) is the same as grid_t_0
    return boundary_positions

def main(hidden_size, num_layers, epochs, model, train_selection):
    """ 
    Params:
    hidden_size - int, # of hidden units for each neural network layer
    num_layers - int, # of neural network layers
    epochs - int, # of training epochs
    model - int, whether to use model 1 (Raissi 2019) or model 2 (continuity PDE)
    train_selection - float, frac of all data (N*T) to select for training OR 
                      'BC', select the boundary conditions for training (all timesteps)
    """
    # Load flattened cynlinder wake data
    fpath = 'E:/propnet-data/data_arrays.npz'
    x_all, y_all, z_all, t_all, pu_all, pv_all, pw_all, fu_all, fv_all, fw_all, fvol_all, p_all, (N, T) = initial_value_load(fpath) # (NT, 1)

    # Select training data 
    if train_selection == 'BC':
        idx = boundary_indices(N, T)
    else:
        samples = int(round(T * train_selection))
        np.random.seed(0)
        idx = np.random.choice(T, samples, replace=False)
    #
    x_all = x_all.reshape(N,T)
    y_all = y_all.reshape(N,T)
    z_all = z_all.reshape(N,T)
    t_all = t_all.reshape(N,T)

    pu_all = pu_all.reshape(N,T)
    pv_all = pv_all.reshape(N,T)
    pw_all = pw_all.reshape(N,T)

    fu_all = fu_all.reshape(N,T)
    fv_all = fv_all.reshape(N,T)
    fw_all = fw_all.reshape(N,T)

    fvol_all = fvol_all.reshape(N,T)
    p_all = p_all.reshape(N,T)

    #
    x_train = x_all[:, idx]
    y_train = y_all[:, idx]
    z_train = z_all[:, idx]
    t_train = t_all[:, idx]
    
    pu_train = pu_all[:, idx]
    pv_train = pv_all[:, idx]
    pw_train = pw_all[:, idx]
    
    fu_train = fu_all[:, idx]
    fv_train = fv_all[:, idx]
    fw_train = fw_all[:, idx]
    
    fvol_train = fvol_all[:, idx]
    p_train = p_all[:, idx]

    #
    x_train = x_train.reshape(-1,1)
    y_train = y_train.reshape(-1,1)
    z_train = z_train.reshape(-1,1)
    t_train = t_train.reshape(-1,1)
    
    pu_train = pu_train.reshape(-1,1)
    pv_train = pv_train.reshape(-1,1)
    pw_train = pw_train.reshape(-1,1)

    fu_train = fu_train.reshape(-1,1)
    fv_train = fv_train.reshape(-1,1)
    fw_train = fw_train.reshape(-1,1)

    fvol_train = fvol_train.reshape(-1,1)
    p_train = p_train.reshape(-1,1)

    # 
    test_indices = np.setdiff1d(np.arange(T), idx)  # 
    # print(test_indices)
    np.save("E:/propnet-data/Htest_indices.npy", test_indices)
    # Instantiate model, criterion, and optimizer
    nu = 0.001/1000 #1mPa·s / 1000kg/m3 0.001/1000
    rho = 1000 #1000kg/m3

    torch.manual_seed(0)
    if model == 2:
        PINN_model = NavierStokesPINN2(hidden_size=hidden_size, num_layers=num_layers, nu=nu, rho=rho).to(device)
        criterion = NavierStokesPINNLoss2().to(device)

    optimizer = torch.optim.LBFGS(PINN_model.parameters(),lr=0.5, line_search_fn='strong_wolfe')

    # Training loop
    def closure():
        """Define closure function to use with LBFGS optimizer"""
        optimizer.zero_grad()   # Clear gradients from previous iteration

        if model == 2:
            pu_pred, pv_pred, pw_pred, fu_pred, fv_pred, fw_pred, p_pred, fvol_pred = PINN_model(x_train, y_train, z_train, t_train)
            # print('shape',fvol_pred.shape)
            loss = criterion(pu_train, pu_pred, pv_train, pv_pred, pw_train, pw_pred, \
                fu_train, fu_pred, fv_train, fv_pred, fw_train, fw_pred, p_train, p_pred, fvol_train, fvol_pred)
            
        loss.backward() # Backprogation
        return loss 

    def training_loop(epochs):
        """Run full training loop"""
        for i in tqdm(range(epochs), desc='Training epochs: '):
            global global_loss_list
            los = closure()
            global_loss_list.append(los.detach().cpu().numpy())
            optimizer.step(closure)

    training_loop(epochs=epochs)

    # Save trained model
    torch.save(PINN_model.state_dict(),
               f'E:/propnet-data/trained.pt')
            
                
    return

if __name__ == '__main__':
    
    num_layers = 8
    hidden_size=48
    epochs = 10
    train_selection = 0.6
    main(hidden_size=hidden_size, num_layers=num_layers, epochs=epochs, model=2, train_selection=train_selection)
    with open(f'E:/propnet-data/loss.pkl', "wb") as f:
        
        pickle.dump(global_loss_list, f)
     
    # with open(f"E:/propnet-data/loss.pkl", "rb") as f:
    #     global_loss_list = pickle.load(f)
    print(np.array(global_loss_list).min())
    
    print(len(global_loss_list))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_yscale('log')
    
    ax.plot(global_loss_list, label="Training Loss", color="blue")
    
    
    ax.set_title("Loss over Epochs", fontsize=16)
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    
    
    ax.legend()
    
    
    ax.grid(True)
    
    
    plt.show()