# Xi Peng, Sep 2016
# Xi Peng, Feb 2017
import sys, os, random
import numpy as np
sys.path.append('/home/px13/code/lib/pylib/')
import PXIO, PXPts

if __name__ == '__main__':
    img_path = '/bigdata1/px13/dataset/'
    pts_path = '/bigdata1/px13/dataset/'
    
    # list
    #dsource = ['300vw/','300w/','aflw/']
    dsource = ['300W2/']
    lines_train, lines_val = [], []
    for ds in dsource:
        img_list = PXIO.ListFileInFolderRecursive(img_path+ds, '.jpg')
        print '%s: %d' % (ds,len(img_list))
        for one_img in img_list:
            token = one_img.split('/')
            img_name = token[-1]
            fold = token[-2]
            img_fpath = img_path+ds+fold+'/' + img_name
            pts_fpath = pts_path+ds+fold+'/' + img_name[:-4] + '.pts'
            line = '%s %s' % (img_fpath, pts_fpath)
            if one_img.find('test')>0:
                lines_val.append(line)
            else:
                lines_train.append(line)
    print 'Total train images: %d' % len(lines_train)
    print 'Total val images: %d' % len(lines_val)
    random.shuffle(lines_train)
    random.shuffle(lines_val)
    PXIO.WriteLineToFile('train_list.txt', lines_train)
    PXIO.WriteLineToFile('val_list.txt', lines_val)


