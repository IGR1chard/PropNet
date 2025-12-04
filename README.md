# PropNet

## **What is this repository for?**

**PropNet is a Physics-Informed Neural Network for Efficient Prediction of Proppant Transport in Hydraulic Fractures.**

This repository provides:

- Loading CFD–DEM simulation data  
- Validating data header information  
- Converting data formats (H5 → NPZ)  
- Training the PropNet neural network  
- Predicting proppant volume fraction in hydraulic fracture sections  
- Visualization tools for simulation vs. prediction comparison  

If you use this software, please cite:

> **PropNet: A Physics-Informed Neural Network for Efficient Prediction of Proppant Transport in Hydraulic Fractures**  
> *(journal information will be added upon publication)*

---

## **How do I get set up?**

### **Repository Structure**

```txt
data_arrays.h5          → Original CFD–DEM simulation dataset
headercheck.dat         → Header metadata for format validation
propnet_load_cfddem.py  → Data loader and header checker
propnet_dataconvert.py  → Convert H5 → NPZ
propnet.py              → PropNet neural network architecture
propnet_train.py        → Training script
propnet_pred.py         → Testing / prediction script
well-trained.pt         → Example trained model
```
### **Setup, Requirements, and Usage**

PropNet requires Python ≥ 3.8 and the following dependencies:
```
numpy  
h5py  
matplotlib  
torch  
scipy  
zlib
os
sys
tqdm
```
### **1.Data Loading**
Run:

python propnet_load_cfddem.py

This script automatically loads:

```
Spatial coordinates: x, y, z
Time steps: t (Δt = 0.05 s)
Fluid velocities: fu, fv, fw
Proppant velocities: pu, pv, pw
Fluid pressure: p
Proppant volume fraction: fvol
And validates consistency using headercheck.dat.
```
### **2.Model Training**
Run:

python propnet_train.py

This script will:
```
Load dataset
Initialize PropNet
Train the model
Save model weights as:trained.pt
```
### **3.Model Testing / Prediction**
Run:

python propnet_pred.py

The script loads the trained model, performs prediction for a target time step, and outputs:
```
True proppant volume fraction at time step-XX.png
Predicted proppant volume fraction at time step-XX.png
Absolute error at time step-XX.png
```
