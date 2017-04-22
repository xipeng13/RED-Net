# Xi Peng, Sep 11 2016
import os, sys, shutil
from PIL import Image
import PXIO

def RemvSpaceInName(inpath):
    shutil.move(inpath+'ibug/image_092 _01.jpg', inpath+'ibug/image_092_01.jpg')
    shutil.move(inpath+'ibug/image_092 _01.pts', inpath+'ibug/image_092_01.pts')

def RemvTrainTestFold(inpath,outpath):
    PXIO.DeleteThenCreateFolder(outpath+'afw')
    PXIO.DeleteThenCreateFolder(outpath+'ibug')
    PXIO.DeleteThenCreateFolder(outpath+'helen')
    PXIO.DeleteThenCreateFolder(outpath+'lfpw')

    infolds = ['afw/','ibug/','helen/trainset/','helen/testset/','lfpw/trainset/','lfpw/testset/']
    outfolds = ['afw/','ibug/','helen/','helen/','lfpw/','lfpw/']
    prefixs = ['afw','ibug','helen_train','helen_test','lfpw_train','lfpw_test']
    fmts = ['.jpg','.jpg','.jpg','.jpg','.png','.png']

    for f in range(6):
        infold = inpath + infolds[f]
        img_list = PXIO.ListFileInFolder(infold, fmts[f])
        print infold
        print len(img_list)
        for img_path in img_list:
            img_name = img_path.split('/')[-1]
            savepath = outpath + outfolds[f] + prefixs[f] + '_' + img_name[:-4] + '.jpg'
            shutil.copy(img_path[:-4]+'.pts',savepath[:-4]+'.pts')
            if img_name.endswith('.jpg'):
                shutil.copy(img_path, savepath)
            elif img_name.endswith('.png'):
                img = Image.open(img_path)
                img.save(savepath)
 

if __name__=='__main__':
    print 'Python 300W Lib by Xi Peng'

    #inpath = '/Users/Xi/Research/Data/resource/face_align/300W/'
    #outpath = '/Users/Xi/Research/Data/resource/face_align/300W2/'
    inpath = '/media/xpeng/Data/resource/face_align/300W/'
    outpath = '/media/xpeng/Data/resource/face_align/300W2/'
    RemvSpaceInName(inpath)
    RemvTrainTestFold(inpath,outpath)

