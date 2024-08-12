from IPython import get_ipython
ipython = get_ipython()
ipython.magic('reset -f')

import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join

import time
from datetime import datetime

# mypath = "c:\\Data\\FindWind20\\"
# mypath = "c:\\Data\\FindWind21\\"
mypath = "c:\\Data\\FindWind22\\"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

mth = np.array([12])

x = 0
for f in mth:
    path = mypath + onlyfiles[f]
    df = pd.read_csv(path, delimiter=',')

    a = df.to_numpy()

    dim = len(a)
    
    res = np.zeros((4034,8))
    # res = np.zeros((4322,8))
    # res = np.zeros((4466,8))
    arw = 0; skp = 0
    for rw in range(dim):
        if rw == 0:
            try:
                uxb = datetime.strptime(a[rw,0], '%d-%m-%Y')
            except:
                uxb = datetime.strptime(a[rw,0], '%d-%m-%Y %H:%M:%S')
            
            uxb = time.mktime(uxb.timetuple())
            res[rw,0:8] = a[rw,1:9]
            arw = arw + 1
            
        else:
            try:
                b = datetime.strptime(a[rw,0], '%d-%m-%Y %H:%M:%S')
            except:
                b = datetime.strptime(a[rw,0], '%d-%m-%Y')
                
            c = time.mktime(b.timetuple())
            
            if c - uxb != 600:
                print(arw,rw)
        
            if c - uxb < 900:
                res[arw,0:8] = a[rw,1:9]
                arw = arw + 1
                
            elif c - uxb == 900 and skp == 1:
                res[arw,0:8] = a[rw,1:9]
                arw = arw + 1
                
            elif c - uxb == 900:
                res[arw:arw+2,0:8] = a[rw,1:9]
                print(98,arw,rw,c,uxb)
                arw = arw + 2
                skp = 1
                
            else:
                miss = int(round((c - uxb) / 600))
                res[arw:int(arw+miss-1),0:8] = 0
                res[arw+miss-1,0:8] = a[rw,1:9]
                print(99,arw,rw,miss,c,uxb)
                arw = arw + miss

            
            uxb = c