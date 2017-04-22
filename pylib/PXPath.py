# Xi Peng, Sep 10 2016
import os

def CurWorkPath():
    cwd = os.getcwd()
    token = cwd.split('/')
    work_path = '/'.join(tk for tk in token[:3])
    return work_path

def CurDataPath():
    work_path = CurWorkPath()
    if work_path == '/Users/Xi':        #macbook pro
        data_path = '/Users/Xi/Research/Data'
    elif work_path == '/home/xpeng':    #Ubuntu
        data_path = '/media/xpeng/Data'
    return data_path
    

if __name__=='__main__':
    print 'Python Path Lib by Xi Peng'
