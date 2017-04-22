# Xi Peng, Jun 16 2016
import numpy as np
import os
import sys

def NormParaForMultiTask(para, meanstdminmax):
    ps,pe,ss,ie,es,ee = 0, 7, 7, 7+199, 7+199, 7+199+29
    pmean, pstd, pmin, pmax = meanstdminmax[0,:],meanstdminmax[1,:],meanstdminmax[2,:],meanstdminmax[3,:]
    para_pos = NormParaScale( para[ps:pe], pmean[ps:pe], pmin[ps:pe], pmax[ps:pe] )
    para_ids = NormParaScale( para[ss:ie], pmean[ss:ie], pmin[ss:ie], pmax[ss:ie] )
    para_exp = NormParaScale( para[es:ee], pmean[es:ee], pstd[es:ee], pmax[es:ee] )
    #para_ids = NormParaGauss( para[ss:ie], pmean[ss:ie], pstd[ss:ie] )
    #para_exp = NormParaGauss( para[es:ee], pmean[es:ee], pstd[es:ee] )
    return (para_pos, para_ids, para_exp)

def DeNormParaForMultiTask(para, meanstdminmax):
    ps,pe,ss,ie,es,ee = 0, 7, 7, 7+199, 7+199, 7+199+29
    pmean, pstd, pmin, pmax = meanstdminmax[0,:],meanstdminmax[1,:],meanstdminmax[2,:],meanstdminmax[3,:]
    para_pos = DeNormParaScale( para[ps:pe], pmean[ps:pe], pmin[ps:pe], pmax[ps:pe] )
    para_ids = DeNormParaScale( para[ss:ie], pmean[ss:ie], pmin[ss:ie], pmax[ss:ie] )
    para_exp = DeNormParaScale( para[es:ee], pmean[es:ee], pmin[es:ee], pmax[es:ee] )
    #para_ids = DeNormParaGauss( para[ss:ie], pmean[ss:ie], pstd[ss:ie] )
    #para_exp = DeNormParaGauss( para[es:ee], pmean[es:ee], pstd[es:ee] )
    return (para_pos, para_ids, para_exp)

def DeNormParaForMultiTask_pos(para, meanstdminmax):
    ps,pe = 0,7
    pmean, pstd, pmin, pmax = meanstdminmax[0,:],meanstdminmax[1,:],meanstdminmax[2,:],meanstdminmax[3,:]
    para_pos = DeNormParaScale( para[ps:pe], pmean[ps:pe], pmin[ps:pe], pmax[ps:pe] )
    return para_pos

def NormParaScale(para, pmean, pmin, pmax):
    para2 = para - pmean
    para3 = np.divide(para2-pmin, pmax-pmin)
    return para3

def DeNormParaScale(para, pmean, pmin, pmax):
    para2 = np.multiply(para, pmax-pmin) + pmin
    para3 = para2 + pmean
    return para3

def NormParaGauss(para, pmean, pstd):
    para2 = para - pmean
    para3 = np.divide(para2-pmean, pstd)
    return para3

def DeNormParaGauss(para, pmean, pstd):
    para2 = np.multiply(para, pstd) + pmean
    para3 = para2 + pmean
    return para3

def ScalePara(para, pmin, pmax):
    para2 = np.divide(para-pmin, max-pmin)
    return para2

def DeScalePara(para, pmin, pmax):
    para2 = np.multiply(para, pmax-pmin) + pmin
    return para2
 

if __name__=='__main__':
    print 'Python Normalization by Xi Peng'
