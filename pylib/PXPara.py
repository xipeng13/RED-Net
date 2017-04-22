# Xi Peng, Dec 30 2015
import os
import numpy as np

def ReadParaFromTxtRecursive(path,format):
    ct = 0
    list = []
    format = '.txt'
    for root, dirs, files in os.walk(path):
        for fold in dirs:
            files = os.listdir(root+fold)
            for file in sorted(files):
                if file.endswith(format):
                    para = np.loadtxt(root+fold+'/'+file) # (n,)
                    list.append(para)
                    ct = ct + 1
                    #if ct == 1000:
                    #    return list
    return list


if __name__=='__main__':
    print 'Python para process by Xi Peng'

