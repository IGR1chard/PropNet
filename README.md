What is this repository for?

PropNet is a Physics-Informed Neural Network for Efficient Prediction of Proppant Transport in Hydraulic Fractures.
It provides tools for loading CFD–DEM simulation data, preprocessing fracture-section flow fields, training a physics-informed neural network (PINN), and predicting proppant volume fraction in vertical fracture sections.

If you use this software in your work, please cite:

PropNet: A Physics-Informed Neural Network for Efficient Prediction of Proppant Transport in Hydraulic Fractures
How do I get set up?

The repository contains:
data_arrays.h5          → Original CFD–DEM simulation dataset  
headercheck.dat         → Header metadata for data validation  
propnet_load_cfddem.py  → Data loader and header checker  
propnet_dataconvert.py  → H5 → NPZ data converter  
propnet.py              → Neural network architecture  
propnet_train.py        → Training script  
propnet_pred.py         → Prediction script  
well-trained.pt         → Example trained model  
Requirements：
GPU Nvidia with 16 GB of memory or higher
Install Python 3.8+ and the dependencies:
numpy
h5py
matplotlib
torch
scipy
zlib
os
sys
pickle
tqdm
