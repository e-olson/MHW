#!/usr/bin/env python
# coding: utf-8
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import warnings
import os
import sys
import dask.array as da
warnings.filterwarnings("ignore", category=RuntimeWarning) # ignore runtime warnings; here they arise from attempted operations on all-NaN arrays

nl = dict(zip( ['CanCM4i', 'COLA-RSMAS-CCSM4', 'GEM-NEMO', 'GFDL-SPEAR', 'NASA-GEOSS2S', 'NCEP-CFSv2', 'CanESM5', 'GEM5.2-NEMO', 'GEM5-NEMO', 'CanCM4i-IC3',],
               [       11,                 11,         11,           11,              8,            9,        11,            11,          11,            11,])) # Max lead time for each model

def BSSpath(mods,is_detrend,years,il):
    basepath='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/MHW'
    # Input/output directory
    mhwdir = basepath+'/mhw/detrended' if is_detrend else basepath+'/mhw'
    if is_detrend:
        f_save = f'{mhwdir}/BSS_MME_{"_".join(mods)}_l{il}_detrended_{years[0]}_{years[1]}.nc'
    else:
        f_save = f'{mhwdir}/BSS_MME_{"_".join(mods)}_l{il}_{years[0]}_{years[1]}.nc'
    return f_save

def calcBSS_il(mods,is_detrend,years,il,save=True):
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
    mhwfor=ffor.is_mhw.mean(dim='M').data[:(-1*il),...]
    mhwobs=fobs.is_mhw.data[il:,...]
    BrS=da.sum((mhwfor-mhwobs)**2,axis=0)/mhwobs.shape[0]
    BrSref=da.sum((.1-mhwobs)**2,axis=0)/mhwobs.shape[0]
    BSS=1-BrS/BrSref
    lmask=np.logical_or(np.sum(fobs.is_mhw.data,axis=0)==0,M0.data==0)

    f_save=BSSpath(mods,is_detrend,years,il)
    if save:
        xout=xr.Dataset(data_vars={'lon':(['X',],ffor.lon.values),
                    'lat':(['Y',],ffor.lat.values),
                    'BrS':(['Y','X'],np.ma.masked_where(lmask,BrS)),
                    'BrSref':(['Y','X'],np.ma.masked_where(lmask,BrSref)),
                    'BSS':(['Y','X'],np.ma.masked_where(lmask,BSS))},
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
    # python BSS-v3.py modopt is_detrend lead
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
    fBSS=BSSpath(mods,is_detrend,years,il)
    print('output file:')
    print(fBSS)
    if not os.path.exists(fBSS):
        calcBSS_il(mods,is_detrend,years,il)
    else:
        print('file already exists- skipping')
    print('done')
    
