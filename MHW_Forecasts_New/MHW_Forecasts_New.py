import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
import os
import glob
import dask.array as da
import datetime as dt
import sys

def mkdirs(fsave):
    saveloc=os.path.dirname(fsave)
    if not os.path.exists(saveloc):
        try:
            os.makedirs(saveloc)
        except FileExistsError:
            pass # in case other code running at the same time got to it first
    return

sourcepath='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/data/predictions/hindcast'
fpat=lambda yyyy,mm,lead: f'{sourcepath}/{yyyy}{mm:02}_MSC_CanSIPS-Hindcast_WaterTemp_Sfc_LatLon1.0_P{lead:02}M.grib2'
workpath = '/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/MHW/newHindcastMonthly'

def sstfile(lead,years):
    return f'{workpath}/sst_HC_l{il}_{years[0]}_{years[1]}.nc'
def anomfile(mod,lead,years,detr):
    detrstr='_detr' if detr=='detr' else ''
    return f"{workpath}/sstAnom{detrstr}_HC{mod}_l{lead}_{years[0]}_{years[1]}.nc"
def mhwfile(lead,years,detr,qtile):
    detrstr='_detr' if detr=='detr' else ''
    return f"{workpath}/mhw{detrstr}_HC_l{lead}_{years[0]}_{years[1]}_p{int(100*qtile)}.nc"

def convert_grib_nc(il,years):
    # load grib files for a given lead time and save them to single netCDF files
    """
    il = lead (0=.5 months, 1=1.5 months, etc)
    years = [firstyear, lastyear]
    """
    flist=[fpat(yyyy,mm,il) for yyyy in range(years[0],years[1]+1) for mm in range(1,13)]
    print('N files = ',len(flist))
    f1=xr.open_mfdataset(flist,engine='cfgrib',combine='nested',concat_dim='time',parallel=True,decode_times=False)
    t0=dt.datetime(1970,1,1)
    time=np.array([t0 + dt.timedelta(seconds=ii.astype(float)) for ii in f1.time.values])
    months=np.array([t1.month-t0.month+(t1.year-t0.year)*12 for t1 in time])
    f_out = sstfile(il,years)
    mkdirs(f_out)
    xout=xr.Dataset(data_vars={'start_time':(['S'],f1.time.values),
                            'sst':(['S','M','Y','X'],f1.sst.data)},
                coords=dict(lon=f1.longitude.values,lat=f1.latitude.values,M=f1.number.values,S=months),)
    xout.S.attrs['units']='months'
    xout.S.attrs['long_name']='Months since 1970-01'
    print(dt.datetime.now())
    xout.to_netcdf(f_out,mode='w')
    print('done: ',dt.datetime.now())
    f1.close()
    print(f_out)
    return

def _detrend_1d(vec):
    ii=~np.isnan(vec)
    x=np.arange(0,len(vec))
    if np.sum(ii)>0:
        p=np.polyfit(x[ii].flatten(),vec[ii].flatten(),1)
        return vec-p[0]*x
    else:
        return vec # in this case should be all NaNs

def calc_anomalies(il,years): 
    f=xr.open_dataset(sstfile(il,years),chunks={'S':-1,'M':-1,'lat':90,'lon':90})
    # ensemble members 1-20 = GEM5.2-NEMO
    # ensemble members 21-40 = CanESM5
    ens=dict(GN=f.sst.isel(M=slice(0,20)),
             C5=f.sst.isel(M=slice(20,40)))
    modkeys=list(ens.keys())
    nt,nm,ny,nx=np.shape(ens['GN']) # both models have same number of ensemble members
    Emean={imod: ens[imod].mean(dim='M',keepdims=True) for imod in modkeys} # ensemble mean for each model
    clim={imod:da.zeros((12,1,ny,nx)) for imod in modkeys} # initialize climatology arrays
    print(dt.datetime.now())
    # define climatologies and load in memory
    for imod in modkeys:
        print(imod)
        for im in range(0,12):
            clim[imod][im,...]=Emean[imod].isel(S=slice(im,-1,12)).mean(dim='S').compute()
            print(im,dt.datetime.now())
    # anomolies as dask array
    anom={(imod,'base'):ens[imod].data-da.tile(clim[imod],(int(nt/12),nm,1,1)) for imod in modkeys} 
    print(dt.datetime.now())
    for imod in modkeys: # add detrended anomalies to anom dict
        anom[(imod,'detr')]=da.apply_along_axis(_detrend_1d,0,anom[(imod,'base')],
                                                shape=(anom[(imod,'base')].shape[0],),dtype=np.float64)
        print(imod, dt.datetime.now())
    
    akeys=list(anom.keys()) # list of all combinations of models with 'base' and 'detr'
    # write anomaly files (to trigger computation and save intermediate step)
    for ikey in akeys:
        mod=ikey[0]
        detr=ikey[1]
        f_out=anomfile(mod,il,years,detr)
        dsout=xr.Dataset(data_vars={'start_time':f.start_time,
                            'anom':(['S','M','Y','X'],anom[ikey])},
                coords=dict(lon=f.lon,lat=f.lat,M=ens[mod].M,S=f.S),)
        dsout.to_netcdf(f_out,mode='w')
        print(dt.datetime.now())
    
    f.close()
    print(dt.datetime.now())
    print('done anomalies')
    return

def find_MHW(il,years,qtile):
    akeys=[('GN', 'base'), ('C5', 'base'), ('GN', 'detr'), ('C5', 'detr')]
    modkeys=['GN','C5']
    ff={};anom={}
    for imod in modkeys:
        for idet in ('base','detr'):
            ff[(imod,idet)]=xr.open_dataset(anomfile(imod,il,years,idet),chunks={'S':-1,'M':-1,'lat':90,'lon':90})
            anom[(imod,idet)]=ff[(imod,idet)].anom.data.rechunk({2:90,3:90})
    nt,nm,ny,nx=np.shape(anom[akeys[0]]) # both models have same number of ensemble members
    print('loaded data in find_MHW',dt.datetime.now())
    qtiles=[qtile,]
    mm=np.arange(0,nt)%12
    sst_an_thr={ikey:da.zeros((12,1,ny,nx)) for ikey in akeys} #initialize threshold arrays
    for ikey in akeys:
        for im in range(0,12):
            ind=(mm==(im-1)%12)|(mm==im)|(mm==(im+1)%12) # 3 month groups running mean
            temp=anom[ikey][ind,...].reshape((int(nt/12*3*nm),ny,nx))
            sst_an_thr[ikey][im,...]=da.apply_along_axis(np.quantile,q=qtiles,axis=0,arr=temp,shape=(len(qtiles),),
                                                         dtype=np.float64).compute()
            print(ikey,im,dt.datetime.now())
    is_mhw={ikey:da.zeros(np.shape(anom[ikey])) for ikey in akeys} #initialize mhw arrays
    dsout={}
    for ikey in akeys:
        is_mhw[ikey]=da.where(anom[ikey]>da.tile(sst_an_thr[ikey],(int(nt/12),nm,1,1)),1,0)
        print(is_mhw[ikey].shape)
    for dval in ('base','detr'):
        dsout=xr.Dataset(data_vars={'start_time':ff[('GN',dval)].start_time,
                                    'mhw':(['S','M','Y','X'],
                                        da.concatenate([is_mhw[('GN',dval)],is_mhw[('GN',dval)]],axis=1))},
                        coords=dict(lon=ff[('GN',dval)].lon,lat=ff[('GN',dval)].lat,
                                    M=np.arange(1,41),S=ff[('GN',dval)].S),)
        f_out=mhwfile(il,years,dval,qtile)
        dsout.to_netcdf(f_out,'w')
    for ikey in akeys:
        ff[ikey].close()
    return


if __name__=="__main__":
    il=int(sys.argv[1])
    qtile=0.9
    years = [1991, 2020]
    #convert_grib_nc(il,years)
    calc_anomalies(il,years)
    find_MHW(il,years,qtile)
