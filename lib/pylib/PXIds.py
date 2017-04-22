# Xi Peng, Jun 16 2016
import os, sys, shutil
import numpy as np

def GenImgListById_casia(img_path,list_save_path):
    for root, dirs, files in os.walk(img_path):
        for id in dirs:
            print id
            files = os.listdir(root+id)
            list = [];
            for file in sorted(files):
                if file.endswith('.jpg'):
                    img_path = root + id + '/' + file
                    list.append(img_path + '\n')

            fd = open(list_save_path+'/'+id+'.txt', 'w')
            for line in list:
                fd.write(line)
            fd.close()

def GenFoldIdDict_casia(root_path):
    dict = {}
    id = 0
    for root, dirs, files in os.walk(root_path):
        for fold in dirs:
            if not dict.has_key(fold):
                dict[fold] = id
                id = id + 1
    return dict

def GenIdDictByImgName_300wlp(NameList):
    dict = {}
    id = 0
    for line in NameList:
        # 300wlp: AFW, HELEN, IBUG, LFPW
        token = line.split('/')
        name_token = token[-1].split('_')
        name = "_".join(tkn for tkn in name_token[:-1])
        if not dict.has_key(name):
            dict[name] = id
            id = id + 1
    print "ids:%d" % id
    return dict



if __name__=='__main__':
    print 'Python IO Lib by Xi Peng'
