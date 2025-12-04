### DEFINES PINN MODEL (AND LOSS FUNCTION) FOR SOLVING NAVIER STOKES ###

import numpy as np
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SinusoidalActivation(nn.Module):
    """
    Sinusoidal activation on the first layer (learning in the sinusoidal space) improves PINN performance, helps
    escape local minima (Wong, 2022)
    """
    def forward(self, x):
        return torch.sin(2 * torch.pi * x)
    


class NavierStokesPINN2(nn.Module):
    def __init__(self, hidden_size, num_layers, nu, rho):
        """
        Model 2 conserves mass via continuity PDE. Both pressure and velocity used in training.
        Params:
        hidden_size - int, number of hidden units per layer
        num_layers - int, number of feedforward layers
        nu - float, kinematic viscosity
        rho - float, density
        """
        self.nu = nu
        self.rho = rho

        super(NavierStokesPINN2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Input layer: input size = 4 (x, y, z, t)
        layer_list = [nn.Linear(4, self.hidden_size)] 
        layer_list.append(SinusoidalActivation()) # Sinusoidal activation for first layer

        for _ in range(self.num_layers - 2):
            layer_list.append(nn.Linear(self.hidden_size, self.hidden_size)) # Hidden layers
            layer_list.append(nn.Tanh()) # Tanh activation for hidden layers
        
        # Output layer: output size = 5 (u, v, w, p, fvol)
        layer_list.append(nn.Linear(self.hidden_size, 8)) # No activation for last layer
        self.layers = nn.ModuleList(layer_list) # Save as Module List

    def forward(self, x, y, z, t):
        """ 
        Params:
        x - nd array of shape (N, 1), input x coordinates
        y - nd array of shape (N, 1), input y coordinates
        z - nd array of shape (N, 1), input z coordinates
        t - nd array of shape (N, 1), input time coordinate
        Returns:
        u - tensor of shape (N, 1), output x-velocity
        v - tensor of shape (N, 1), output y-velocity
        p - tensor of shape (N, 1), output pressure
        f - x-momentum PDE evaluation of shape (N, 1)
        g - y-momentum PDE evaluation of shape (N, 1)

        """
        # Convert input data to tensor
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
        y = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(device)
        z = torch.tensor(z, dtype=torch.float32, requires_grad=True).to(device)
        t = torch.tensor(t, dtype=torch.float32, requires_grad=True).to(device)

        # Feed-forward to calculate pressure and latent psi 
        input_data = torch.hstack([x, y, z, t]) # (N, 4)
        self.N = input_data.shape[0]
        
        out = input_data # Initialize feed-forward
        for layer in self.layers:
            out = layer(out) # Final layer, (N, 4)

        # Seperate psi and pressure p
        pu, pv, pw, fu, fv, fw, p, fvol = out[:, [0]], out[:, [1]], out[:, [2]], out[:, [3]], out[:, [4]], out[:, [5]], out[:, [6]], out[:, [7]] # (N, 1) each
        
        # Hard constraints for fluid velocities (fu, fv, fw)
        # Velocity inlet at x=0 (left boundary)
        phi_inlet = x  # Distance function for inlet (x=0)
        u_inlet = 0.15  # m/s
        fu = phi_inlet * fu + u_inlet * (1 - phi_inlet)
        fv = phi_inlet * fv
        fw = phi_inlet * fw
        
        u_inlet = 0.15  # m/s
        pu = phi_inlet * pu + u_inlet * (1 - phi_inlet)
        pv = phi_inlet * pv
        pw = phi_inlet * pw

        # No-slip boundary conditions for fluid at other walls
        phi_walls = (x - 0) * (1 - x) * (z - 0) * (1 - z)  # Distance function for walls
        fu = phi_walls * fu
        fv = phi_walls * fv
        fw = phi_walls * fw

        # Hard constraints for pressure (p)
        # Pressure outlet at x=200mm (right boundary)
        phi_outlet = (1 - x)  # Distance function for outlet (x=200)
        p = phi_outlet * p
        
        return pu, pv, pw, fu, fv, fw, p, fvol, #g, f


class NavierStokesPINNLoss2(nn.Module):
    """
    Loss function for Model 2
    Implement PINN loss function for Navier-Stokes PINN
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss() # implement mean square error
        
    def forward(self, pu, pu_pred, pv, pv_pred, pw, pw_pred, fu, fu_pred, fv, fv_pred, fw, fw_pred, p, p_pred, fvol, fvol_pred):
        pu = torch.tensor(pu, dtype=torch.float32).to(device)
        pv = torch.tensor(pv, dtype=torch.float32).to(device)
        pw = torch.tensor(pw, dtype=torch.float32).to(device)
        
        fu = torch.tensor(fu, dtype=torch.float32).to(device)
        fv = torch.tensor(fv, dtype=torch.float32).to(device)
        fw = torch.tensor(fw, dtype=torch.float32).to(device)
        
        p = torch.tensor(p, dtype=torch.float32).to(device)
        fvol = torch.tensor(fvol, dtype=torch.float32).to(device)
        # Loss due to data
        L_data_f = self.mse(fu, fu_pred) + self.mse(fv, fv_pred) + self.mse(fw, fw_pred)
        L_data_p = self.mse(pu, pu_pred) + self.mse(pv, pv_pred) + self.mse(pw, pw_pred)
        L_data =  L_data_f + L_data_p + self.mse(p, p_pred) + self.mse(fvol, fvol_pred)
        
        
        return L_data 