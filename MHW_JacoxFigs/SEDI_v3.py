#!/usr/bin/env python
# coding: utf-8
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import warnings
import os
import sys
warnings.filterwarnings("ignore", category=RuntimeWarning) # ignore runtime warnings; here they arise from attempted operations on all-NaN arrays

# calculate TP: true positives
#           TN: true negatives
#           FP: false positives
#           FN: false negatives
#
#           F: false alarm rate = FP/(total observed non-events * M)
#           H: hit rate = TP/(total observed positive events * M)
#           SEDI: (logF-logH-log(1-F)+log(1-H))/(logF+logH+log(1-F)+log(1-H))# ignore ice mask for now; apply in plots

nl = dict(zip( ['CanCM4i', 'COLA-RSMAS-CCSM4', 'GEM-NEMO', 'GFDL-SPEAR', 'NASA-GEOSS2S', 'NCEP-CFSv2', 'CanESM5', 'GEM5.2-NEMO', 'GEM5-NEMO', 'CanCM4i-IC3',],
               [       11,                 11,         11,           11,              8,            9,        11,            11,          11,            11,])) # Max lead time for each model

def SEDIpath(mods,is_detrend,years,il):
    basepath='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/MHW'
    # Input/output directory
    mhwdir = basepath+'/mhw/detrended' if is_detrend else basepath+'/mhw'
    if is_detrend:
        f_save = f'{mhwdir}/SEDI_MME_{"_".join(mods)}_l{il}_detrended_{years[0]}_{years[1]}.nc'
    else:
        f_save = f'{mhwdir}/SEDI_MME_{"_".join(mods)}_l{il}_{years[0]}_{years[1]}.nc'
    return f_save

def calcSEDI_il(mods,is_detrend,years,il,save=True):
    basepath='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/MHW'
    # Input/output directory
    if is_detrend:
        mhwdir = basepath+'/mhw/detrended';
        f_obs = basepath+f'/OISST/mhw_detrended_oisst-avhrr-v02r01.regridded1x1.monthly.{years[0]}_{years[-1]}.nc'
    else:
        mhwdir = basepath+'/mhw';
        f_obs = basepath+f'/OISST/mhw_oisst-avhrr-v02r01.regridded1x1.monthly.{years[0]}_{years[-1]}.nc'
    fobs=xr.open_dataset(f_obs)
        
    # Loop through models
    is_mhw_ens=[]
    mlist=[]
    flist=[]
    for modi in mods:
        if il<nl[modi]:
            # Load MHWs
            if is_detrend:
                f_in = f'{mhwdir}/mhw_{modi}_l{il}_detrended_{years[0]}_{years[1]}.nc'
            else:
                f_in = f'{mhwdir}/mhw_{modi}_l{il}_{years[0]}_{years[1]}.nc'
            flist.append(f_in)
    ffor=xr.open_mfdataset(flist,chunks={'X':10,'Y':10,'M':-1},concat_dim='M',combine='nested',data_vars='minimal',
                       coords='minimal',parallel=True,preprocess=lambda f: f.drop_vars(["sst_an_thr","mhw_prob"]) )

    M0=xr.where(np.isnan(ffor.is_mhw),0,1).sum(dim='M').mean(dim='S')
    
    Ms=np.unique(M0)
    if len(Ms)>1: raise Exception('check M')
    M=Ms[0]
    mhwfor=ffor.is_mhw.data[:(-1*il),...]
    mhwobs=fobs.is_mhw.data[il:,...]
    N_pos=np.sum(mhwfor,axis=1).compute()
    N_neg=np.sum((mhwfor==0).astype(float),axis=1).compute()
    TP=np.where(mhwobs==1,N_pos,0)
    TN=np.where(mhwobs==0,N_neg,0)
    FP=np.where(mhwobs==0,N_pos,0)
    FN=np.where(mhwobs==1,N_neg,0)
    
    # calculate SEDI, summed over time
    Nobs_pos=np.sum(mhwobs,axis=0)
    Nobs_neg=np.sum(1-mhwobs,axis=0)
    F=np.sum(FP,axis=0)/(Nobs_neg*M)
    H=np.sum(TP,axis=0)/(Nobs_pos*M)
    
    SEDI=(np.log(F)-np.log(H)-np.log(1-F)+np.log(1-H))/(np.log(F)+np.log(H)+np.log(1-F)+np.log(1-H))
    lmask=np.logical_or(np.sum(fobs.is_mhw.data,axis=0)==0,M0.data==0)

    f_save=SEDIpath(mods,is_detrend,years,il)
    if save:
        xout=xr.Dataset(data_vars={'lon':(['X',],ffor.lon.values),
                    'lat':(['Y',],ffor.lat.values),
                    'SEDI':(['Y','X'],np.ma.masked_where(lmask,SEDI))},
                    coords=dict(X=ffor.X,Y=ffor.Y),)
        xout.to_netcdf(f_save,mode='w')
    ffor.close()
    return

modict={'Jacox':['CanCM4i', 'COLA-RSMAS-CCSM4', 'GEM-NEMO', 'GFDL-SPEAR', 
                        'NASA-GEOSS2S', 'NCEP-CFSv2'],
        'CanSIPSv2':['CanCM4i','GEM-NEMO'],
        'CanSIPSv21':['CanCM4i-IC3','GEM5-NEMO'],
        'CanSIPSv3':['CanESM5','GEM5.2-NEMO'],
        'GFDLNASA':[ 'GFDL-SPEAR','NASA-GEOSS2S'],}

if __name__=="__main__":
    # python SEDI-v3.py modopt is_detrend lead
    modopt=sys.argv[1] # option defining group of models to run; single model or dict key
    is_detrend=bool(sys.argv[2]) # True or False or 1 or 0
    il=int(sys.argv[3]) # lead, 0-10
    if modopt in modict.keys():
        mods=modict[modopt]
    elif modopt in nl.keys():
        mods=[modopt,]
    else:
        raise Exception(f'nonexistent modopt value:{modopt}')

    years = [1991, 2020]
    basepath='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/MHW'
    
    print(f'options:\n  mods:{mods}\n  detrend:{is_detrend}\n  years:{years}\n  lead:{il}')
    fSEDI=SEDIpath(mods,is_detrend,years,il)
    print('output file:')
    print(fSEDI)
    if not os.path.exists(fSEDI):
        calcSEDI_il(mods,is_detrend,years,il)
    else:
        print('file already exists- skipping')
    print('done')
    
