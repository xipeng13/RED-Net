# Xi Peng, Dec 2016
import os, sys

def AddPath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

pylib_path = './pylib'
AddPath(pylib_path)

model_save_path = '/bigdata1/px13/torch/res_det_reg/'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
