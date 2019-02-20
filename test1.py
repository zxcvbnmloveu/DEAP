import numpy as np
# import scipy.io as sio
# import os
# import sys
# import random
# import matplotlib.pyplot as plt
# import pandas as pd
# import pickle
# import mne
# import pyedflib as edf
# import random
# from scipy import signal
# import math
# import shutil
# from sklearn import preprocessing

import numpy as np
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a.shape)
lable = np.arange(3)
permutation = np.random.permutation(lable.shape[0])
data = a[permutation,:]
print(data)

