import os, sys
import datetime as dt
import cftime
import pandas as pd
import xarray as xr
import numpy as np
from dask.distributed import Client, LocalCluster
import dask.array as da
from mhw_daily_paths import * # break script out into smaller files
from mhw_daily_stats import * # break script out into smaller files

ylimlistobs=[[1991,2000],[2001,2010],[2011,2020],[2021,2024]]

def mkdirs(fsave):
    saveloc=os.path.dirname(fsave)
    if not os.path.exists(saveloc):
        try:
            os.makedirs(saveloc)
        except FileExistsError:
            pass # in case other code running at the same time got to it first
    return

def yd365(tdt):
    ##yd=int((dt.datetime(tdt.year,tdt.month,tdt.day)-dt.datetime(tdt.year-1,12,31)).days) # extra code in case of time components
    ##if yd==366: yd=365 # move leap days to overlap with day 365
    # fail if calendar has not been converted to NoLeap by removing 2/29's
    yd=int((cftime.DatetimeNoLeap(tdt.year,tdt.month,tdt.day)-cftime.DatetimeNoLeap(tdt.year-1,12,31)).days) # extra code in case of time components
    return yd

#def lsqfit_md_detr(data):
#    # linearly detrend along axis 0
#    # assume no NaN values; this is for model results
#    # adapt reshaping code from scipy.signal.detrend
#    # put new dimensions at end
#    data=np.asarray(data)
#    dshape = data.shape
#    N=dshape[0]
#    X=np.concatenate([np.ones((N,1)), np.expand_dims(np.arange(0,N),1)],1)
#    newdata = np.reshape(data,(N, np.prod(dshape, axis=0) // N)).copy() # // is floor division; ensure copy
#    b=np.linalg.lstsq(X,newdata,rcond=None)[0] # res=np.sum((np.dot(X,b)-Y)**2)
#    ydetr=newdata-np.dot(X,b)
#    ydetr=np.reshape(ydetr,dshape)
#    return ydetr

def lsqfit_md_detr_calcb(data):
    # linearly detrend along axis 0
    # assume no NaN values; this is for model results
    # adapt reshaping code from scipy.signal.detrend
    # put new dimensions at end
    data=np.asarray(data)
    dshape = data.shape
    N=dshape[0]
    X=np.concatenate([np.ones((N,1)), np.expand_dims(np.arange(0,N),1)],1)
    newdata = np.reshape(data,(N, np.prod(dshape, axis=0) // N)).copy() # // is floor division; ensure copy
    b=np.linalg.lstsq(X,newdata,rcond=None)[0] # res=np.sum((np.dot(X,b)-Y)**2)
    return b

def lsqfit_md_detr_applyb(data,b,offset):
    # linearly detrend along axis 0
    # assume no NaN values; this is for model results
    # adapt reshaping code from scipy.signal.detrend
    # put new dimensions at end
    # b is fit calculated from lsqfit_md_detr_calcb
    # offset is number of intervals offset between start of array to detrend and array from which trend is derived (loc of y intercept in series)
    #    in case data supplied here and data supplied to lsqfit_md_detr_calcb are different time slices
    data=np.asarray(data)
    dshape = data.shape
    N=dshape[0]
    X=np.concatenate([np.ones((N,1)), np.expand_dims(np.arange(0,N)+offset,1)],1)
    newdata = np.reshape(data,(N, np.prod(dshape, axis=0) // N)).copy() # // is floor division; ensure copy
    b=np.reshape(b,(2, np.prod(dshape[1:]))).copy()
    ydetr=newdata-np.dot(X,b)
    ydetr=np.reshape(ydetr,dshape)
    return ydetr

def lsqfit_md_detrPooled(data): # remove single trend for all ensemble members (per model)
    # reshape so ensemble members are concatenated along first axis
    # linearly detrend along axis 0
    # assume no NaN values; this is for model results
    data=np.asarray(data)
    dshape=data.shape
    N=dshape[0]
    R=dshape[1]
    X=np.concatenate([np.ones((R*N,1)),(np.arange(0,N).reshape((N,1))*np.ones((N,R))).reshape((R*N,-1))],1)
    newdata = np.reshape(data,(R*N, np.prod(dshape, axis=0) // (R*N))).copy() # // is floor division; ensure copy
    b=np.linalg.lstsq(X,newdata,rcond=None)[0] # res=np.sum((np.dot(X,b)-Y)**2)
    ydetr=newdata-np.dot(X,b)
    ydetr=np.reshape(ydetr,dshape)
    return ydetr

def lsqfit_md_detrPooled_saveb(x0,data,climyrs,ilead,istartlat,lats,lons,fout): # remove single trend for all ensemble members (per model)
    # reshape so ensemble members are concatenated along first axis
    # linearly detrend along axis 0
    # assume no NaN values; this is for model results
    data=np.asarray(data)
    dshape=data.shape
    N=dshape[0]
    R=dshape[1]
    X=np.concatenate([np.ones((R*N,1)),(x0.reshape((N,1))*np.ones((N,R))).reshape((R*N,-1))],1)
    newdata = np.reshape(data,(R*N, np.prod(dshape, axis=0) // (R*N))).copy() # // is floor division; ensure copy
    b=np.linalg.lstsq(X,newdata,rcond=None)[0] # res=np.sum((np.dot(X,b)-Y)**2)
    ydetr=newdata-np.dot(X,b)
    b=np.reshape(b,tuple([2]+list(dshape)[2:]))
    dsb=xr.Dataset(data_vars={'fit':(['b','lat','lon'],b),},
                   coords={'b':np.arange(0,2),
                           'lat':lats,
                           'lon':lons})
    dsb.to_netcdf(fout,mode='w')
    return

def _add_dims(arr,tarr):
    while len(np.shape(arr))<len(np.shape(tarr)):
        arr=np.expand_dims(arr,-1)
    return arr

def trismooth(t,vals,L=30,periodic=False):
    # t is values assoc with 1st dim
    # smooths over 1st dim
    # if vector, add dim:
    delt=t[1]-t[0]
    alpha=1
    if len(np.shape(vals))==1:
        vals=np.expand_dims(vals,axis=1)
    fil=np.empty(np.shape(vals))
    for ind, ti in enumerate(t):
        if periodic:
            diff=np.minimum(np.minimum(np.abs(ti-t),np.abs(ti-t-365)),np.abs(ti-t+365))
            Leff=L
        else:
            diff=np.abs(ti-t)
            Leff=min(L,alpha*(ti-t[0]+1)*delt,alpha*(t[-1]-ti+1)*delt)# do not smooth beginning and end asymmetrically
        weight=_add_dims(np.maximum(Leff-diff,0),vals)
        fil[ind,...]=np.divide(np.nansum(weight*vals,0),np.nansum(weight*~np.isnan(vals),0),
                               out=np.nan*da.array(np.ones(np.shape(vals)[1:])),
                               where=np.nansum(weight*~np.isnan(vals),0)>0)
    return fil

def fconvert_CanESM(yyyy,mm,dd,hh):
    fin=fnameCanESMjoined(mdirC5,yyyy,mm,dd,hh)
    fout=fnameCanESMdaily(mdirC5,yyyy,mm,dd,hh)
    if not os.path.exists(fout):
        print(fout,flush=True)
        ff=xr.open_dataset(fin,decode_times=False).chunk({'lat':30,'lon':30})
        ff2=ff.drop_vars(['realization','hcrs']).rename({'record':'r'})
        ff3=ff2.coarsen(leadtime=4).mean()
        ff3.tso.assign_attrs({'postprocess':'daily time average, [(6,12,18,24),...]'})
        ff3.to_netcdf(fout,mode='w')
        for fff in [ff3, ff2, ff]:
            fff.close()
    return
    
def fconvert_CanESM_1d5d(yyyy,mm):
    fin=fnameCanESMdaily(mdirC5,yyyy,mm,1,0)
    fout=fnameCanESM5d(mdirC5,yyyy,mm,1,0)
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    if not os.path.exists(fout):
        print(fout,flush=True)
        ff=xr.open_dataset(fin,decode_times={'leadtime':False,'reftime':True},drop_variables='time').chunk({'lat':30,'lon':30})
        ff3=ff.coarsen(leadtime=5).mean() # 5 days
        ir=pd.Timestamp(ff.reftime.values)
        ff3['reftime']=cftime.DatetimeNoLeap(ir.year,ir.month,ir.day,ir.hour,ir.minute,ir.second)
        ff3['reftime']=ff3['reftime'].assign_attrs({'standard_name':ff['reftime'].standard_name,'long_name':ff['reftime'].long_name})
        ff3['tso']=ff3.tso.assign_attrs({'postprocess':'5d time average, [(6,12,18,24),...]'})
        ff3.to_netcdf(fout,mode='w')
        for fff in [ff3,ff]:
            fff.close()
    return

def calcClim_CanESM5(climyrs,L=1):#,nlead):
    # L can be 1 or 5 days
    for mm in range(1,13): # month loop
        print(f"Month:{mm} {dt.datetime.now()}",flush=True)
        print("ncpu:",ncpu)
        #for ix in range(0,int(nlead/5)):
        if L==1:
            flist=[fnameCanESMdaily(mdirC5,yyyy,mm,1,0) for yyyy in range(climyrs[0],climyrs[-1]+1)] 
        elif L==5:
            flist=[fnameCanESM5d(mdirC5,yyyy,mm,1,0) for yyyy in range(climyrs[0],climyrs[-1]+1)] 
        else: 
            raise Exception(f'unexpected L ({L})')
        fnameclim=fnameCanESMClim(workdir[L],climyrs[0],climyrs[-1],mm,smoothClim=False,L=L)
        mkdirs(fnameclim)
        with LocalCluster(n_workers=ncpu-3,threads_per_worker=1) as cluster, Client(cluster) as client:
            ff=xr.open_mfdataset(flist,parallel=True,combine='nested',concat_dim='reftime',
                           decode_times=False,decode_timedelta=False,drop_variables='time')
            EClim=ff.tso.mean(dim=['reftime','r'],skipna=False)
            EClim=EClim.compute(scheduler="processes")
            EClim.to_netcdf(fnameclim,mode='w')
            ff.close()
        del EClim
            
    return

def smoothClim_CanESM5(climyrs,smoothmethod,window,L=1):
    with LocalCluster(n_workers=ncpu-2,threads_per_worker=1) as cluster, Client(cluster) as client:
        flistclim = [fnameCanESMClim(workdir[L],climyrs[0],climyrs[-1],mm,smoothClim=False,L=L) for mm in range(1,13)]
        fclim=xr.open_mfdataset(flistclim,combine='nested',concat_dim='month',parallel=True,decode_times=False)
        SST=fclim.tso.data.rechunk([1,-1,90,120])
        climS=da.empty_like(SST)
        if smoothmethod=='tri':
            smoothfun=trismooth
        else:
            raise Exception('method not implemented:',smoothmethod)
        for ix in range(0,12):
            climS[ix,...]=da.map_blocks(smoothfun,fclim.leadtime.values/24,SST[ix,...],window,dtype=float)
        for mm in range(1,13):
            climSout=climS[mm-1,...].compute(scheduler='processes')
            fout=fnameCanESMClim(workdir[L],climyrs[0],climyrs[-1],mm,smoothClim=True,method=smoothmethod,window=window,L=L)
            print(fout)
            dsout=xr.Dataset(data_vars={'tso':(['leadtime','lat','lon'],climSout)},
                             coords={'leadtime':fclim.leadtime,
                                     'lat':fclim.lat,
                                     'lon':fclim.lon})
            dsout.to_netcdf(fout,mode='w')
        fclim.close()
    return

def calcAnom_CanESM5(climyrs,mm,smoothClim=False,smoothmethod=None,window=1,L=1):#,nlead):
    # mm 1-12
    print(f"Month:{mm} {dt.datetime.now()}",flush=True)
    #for ix in range(0,int(nlead/5)):
    #fnamelast=fnameCanESMAnom0(workdir[L],climyrs[0],climyrs[-1],climyrs[-1],mm,smoothClim,smoothmethod,window,L)
    #if not os.path.exists(fnamelast): # skip if file at (almost) end already exists
    #    mkdirs(fnamelast)
    #    if L==1:
    #        flist=[fnameCanESMdaily(mdirC5,yyyy,mm,1,0) for yyyy in range(1993,2025) if yyyy<2024 or mm<=6] # stop at Jul 2024
    #    elif L==5:
    #        flist=[fnameCanESM5d(mdirC5,yyyy,mm,1,0) for yyyy in range(1993,2025) if yyyy<2024 or mm<=6] # stop at Jul 2024
    #    else:
    #        raise Exception(f'unexpected L ({L})')
    fnameclim=fnameCanESMClim(workdir[L],climyrs[0],climyrs[-1],mm,smoothClim,smoothmethod,window,L)
    fclim=xr.open_dataset(fnameclim,decode_times=False)
    EClim=fclim['tso']
    for iy in range(1993,2025):
        if iy<2024 or mm<=6: # stop at Jul 2024
            if L==1:
                fn=fnameCanESMdaily(mdirC5,iy,mm,1,0)
            if L==5:
                fn=fnameCanESM5d(mdirC5,iy,mm,1,0)
            else:
                raise Exception(f'unexpected L ({L})')
            ff=xr.open_dataset(fn,decode_times=False,chunks={'leadtime':1,'r':-1,'lat':-1,'lon':-1})
            fname=fnameCanESMAnom0(workdir[L],climyrs[0],climyrs[-1],iy,mm,smoothClim,smoothmethod,window,L)
            print(fname,flush=True)
            mkdirs(fname)
            Anom0=ff.tso-EClim
            Anom0.rename('sst_an').to_netcdf(fname,
                    encoding={'sst_an': {'chunksizes': [Anom0.shape[0],1,30,360]}},mode='w')
            del Anom0
    del EClim
    fclim.close()
    return

def anom_bylead(climyrs,ilead,smoothClim=False,smoothmethod=None,window=1,L=1):
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    for jj in range(0,180,60):
        flist=[fnameCanESMAnom0(workdir[L],climyrs[0],climyrs[-1],yy,mm,smoothClim,smoothmethod,window,L) for yy in range(1993,2025) for mm in range(1,13) if yy<2024 or mm<=6]
        fnamout=fnameCanESMAnomByLead(workdir[L],climyrs[0],climyrs[-1],ilead,jj,smoothClim,smoothTrend=False,meth=smoothmethod,win=window,L=L,detrended=False)
        with LocalCluster(n_workers=ncpu-1,threads_per_worker=1) as cluster, Client(cluster) as client:
            ff= xr.open_mfdataset(flist,parallel=True,combine='nested',concat_dim='reftime',decode_times={'reftime':time_coder,'leadtime':False},
                                    preprocess=lambda ff: ff.isel(leadtime=ilead,lat=slice(jj,jj+60)))
            sst_an2=ff.sst_an.chunk({'reftime':ff.sst_an.shape[0],'r':20,'lat':30,'lon':360})
            # fix time
            reftime=[cftime.DatetimeNoLeap(yy,mm,1,0,0) for yy in range(1993,2025) for mm in range(1,13) if yy<2024 or mm<=6]
            time=[cftime.DatetimeNoLeap(yy,mm,1,0,0)+dt.timedelta(hours=float(ff.leadtime.values)) \
                        for yy in range(1993,2025) for mm in range(1,13) if yy<2024 or mm<=6]
            dout=sst_an2.data.compute(scheduler='processes')
            fout=xr.Dataset(data_vars={'sst_an':(['reftime','r','lat','lon'],dout),
                                       'time':(['reftime',],time,{'long_name':'Real Time'})},
                            coords={'reftime':reftime,
                                    'r':np.arange(0,ff.sst_an.shape[1]),
                                    'lat':ff.lat,
                                    'lon':ff.lon})
            mkdirs(fnamout)
            print(fnamout)
            fout.to_netcdf(fnamout,mode='w') # encoding={'sst_an': {'chunksizes': [Anom0.shape[0],1,20,360]}}
            del sst_an2; del fout;
            ff.close(); del ff;
    return
# restrict times?
def anom_bylead_savetr(climyrs,ilead,jj,smoothClim=False,smoothmethod=None,window=1,L=1):
    fin=fnameCanESMAnomByLead(workdir[L], climyrs[0], climyrs[-1], ilead, jj, smoothClim, False, smoothmethod, window,L,detrended=False)
    print(fin)
    print('smoothmethod:',smoothmethod)
    fout=fnameCanESMDetrFitByLead(workdir[L], climyrs[0], climyrs[-1], ilead, jj, smoothClim,False,meth=smoothmethod,win=window,L=L)
    mkdirs(fout)
    print(fout)
    ff=xr.open_dataset(fin,decode_times=True)
    days=cftime.date2num(ff.reftime.values,f'days since {climyrs[0]:04}-01-01')
    lsqfit_md_detrPooled_saveb(days,ff.sst_an,climyrs,ilead,jj,ff.lat.values,ff.lon.values,fout)
    ff.close()
    return
    
def smoothTrend_CanESM5(yind,climyrs,smoothClim=False,smoothmethod=None,window=1,L=1):
    if not smoothClim: raise Exception('Bad combination: smoothed trend without smoothed climatology')
    if smoothmethod=='tri':
        smoothfun=trismooth
    else:
        raise Exception('method not implemented:',smoothmethod)
    flistbS=[fnameCanESMDetrFitByLead(workdir[L], climyrs[0], climyrs[-1], ilead, yind, smoothClim,smoothTrend=False,meth=smoothmethod,win=window,L=L) \
                for ilead in range(0,int(215/L))]
    fbS=xr.open_mfdataset(flistbS,combine='nested',concat_dim=['leadtime'],parallel=True,decode_times=False)
    borig=fbS.fit.data.rechunk([-1,-1,30,180])
    bsmooth=da.map_blocks(smoothfun,fbS.leadtime.values,borig,window,dtype=float) #here lead time is in days
    for ilead in range(0,215):
        fout=fnameCanESMDetrFitByLead(workdir[L], climyrs[0],climyrs[-1], ilead, yind, smoothClim,smoothTrend=True,meth=smoothmethod, win=window, L=L)
        print(fout)
        dsout=xr.Dataset(data_vars={'fit':(['b','lat','lon'],bsmooth[ilead,...])},
                    coords={'b':fbS.b,
                       'lat':fbS.lat,
                       'lon':fbS.lon})
        dsout.to_netcdf(fout,mode='w')
    fbS.close()
    return

def anom_bylead_detr(climyrs,ilead,jj,smoothedClim=False,smoothedTrend=False,smoothmethod=None,window=1,L=1):
    if smoothedTrend and not smoothedClim: raise Exception('Bad combination: smoothed trend without smoothed climatology')
    # note: smoothedTrend implies smoothedClim, but can load unsmoothed trends from smoothed climatology-based anomalies
    fb=fnameCanESMDetrFitByLead(workdir[L], climyrs[0],climyrs[-1], ilead, jj, smoothedClim,smoothedTrend,smoothmethod, window, L)
    fin =fnameCanESMAnomByLead(workdir[L], climyrs[0], climyrs[-1], ilead, jj, smoothedClim,smoothedTrend,smoothmethod,window,L,detrended=False)
    fout=fnameCanESMAnomByLead(workdir[L], climyrs[0], climyrs[-1], ilead, jj, smoothedClim,smoothedTrend,smoothmethod,window,L,detrended=True)
    # 3 options: no smoothing; smoothed clim and raw trend; smoothed clim and smoothed trend
    mkdirs(fout)
    print(fout)
    ff=xr.open_dataset(fin,decode_times=True)
    days=cftime.date2num(ff.reftime.values,f'days since {climyrs[0]:04}-01-01')
    ftr=xr.open_dataset(fb,decode_times=False)
    trest=ftr.fit.isel(b=0)+days*ftr.fit.isel(b=1)
    sstanomdet=ff.sst_an-trest
    sstanomdet=sstanomdet.rename('sst_an')
    sstanomdet.to_netcdf(fout,mode='w')
    ff.close()
    ftr.close()
    return

#def calc_quantile_CanESM(climyrs,ilead,jj,qtile,detr=True,smoothedClim=False,smoothedTrend=False,smoothmethod=None,window=1,delt=0,L=1):
#     # version 1: 10 day windows in lead time
#     lmax=215
#     def getind(i0):
#         if i0>=1 and i0<=10:
#             return [i0-1,i0,i0+1]
#         elif i0==0:
#             return [11,0,1]
#         elif i0==11:
#             return [10,11,0]
#     #def leadbounds(l0,lmax,delt):
#     #    i0=min(max(l0-delt,0),lmax-(2*delt+1))
#     #    return i0, i0+2*delt+1
#     def leadbounds(l0,lmax,delt):
#         return max(0,l0-delt), min(lmax,l0+delt+1)
#     flist=[fnameCanESMAnomByLead(workdir[L], climyrs[0],climyrs[-1],il,jj,smoothClim=smoothedClim,smoothTrend=smoothedTrend,meth=smoothmethod,win=window,L=L,detrended=detr) \
#                 for il in range(*leadbounds(ilead,215,delt))]
#     print(flist)
#     fqout=fnameCanESMAnomQtile(workdir[L], climyrs[0], climyrs[-1], ilead, jj, qtile, detr, 
#                                smoothClim=smoothedClim,smoothTrend=smoothedTrend,meth=smoothmethod,win=window,delt=delt,L=L)
#     ff=xr.open_mfdataset(flist,combine='nested',concat_dim=['leadtime'],parallel=True,decode_times=False)
#     fc=ff.sst_an.coarsen(reftime=12,boundary='pad').construct(reftime=('year','month'))
#     sh=fc.shape
#     ql1=np.nan*np.ones((12,sh[-2],sh[-1]))
#     ql2=np.nan*np.ones((12,sh[-2],sh[-1]))
#     for ii in range(0,12):
#         pool1=fc.isel(month=ii).values.reshape((sh[0]*sh[1]*sh[3],sh[4],sh[5]))
#         ql1[ii,...]=np.nanquantile(pool1,qtile,axis=0)
#         pool2=fc.sel(month=getind(ii)).values.reshape((sh[0]*sh[1]*3*sh[3],sh[4],sh[5]))
#         ql2[ii,...]=np.nanquantile(pool2,qtile,axis=0)
#     print(fqout)
#     dsqt=xr.Dataset(data_vars={'qt1':(['month','lat','lon'],ql1,{'long_name':f"{100*qtile}th percentile value"}),
#                                'qt2':(['month','lat','lon'],ql2,{'long_name':f"{100*qtile}th percentile value"}),},
#                    coords={'month':np.arange(0,12),
#                            'lat':ff.lat,
#                            'lon':ff.lon})
#     dsqt.to_netcdf(fqout,mode='w')
#     del dsqt; del fc;
#     ff.close()
#     return

def calc_quantile_CanESM(climyrs,ilead,jj,qtile,detr=True,smoothedClim=False,smoothedTrend=False,smoothmethod=None,window=1,delt=0,L=1):
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    lmax=int(215/L)
    assert delt%L==0 # have not decided what to do otherwise
    def leadbounds(l0,lmax,delt):
        return max(0,l0-int(delt/L)), min(lmax,l0+int(delt/L)+1)
    flist=[fnameCanESMAnomByLead(workdir[L], climyrs[0],climyrs[-1],il,jj,smoothClim=smoothedClim,smoothTrend=smoothedTrend,meth=smoothmethod,win=window,L=L,detrended=detr) \
                for il in range(*leadbounds(ilead,lmax,delt))]
    fqout=fnameCanESMAnomQtile(workdir[L], climyrs[0], climyrs[-1], ilead, jj, qtile, detr, 
                               smoothClim=smoothedClim,smoothTrend=smoothedTrend,meth=smoothmethod,win=window,delt=delt,L=L)
    if os.path.exists(fqout): return
    print(flist,flush=True)
    with LocalCluster(n_workers=ncpu-1,threads_per_worker=1) as cluster, Client(cluster) as client:
        ff=xr.open_mfdataset(flist,combine='nested',concat_dim=['leadtime'],parallel=True,decode_times=False)#{'reftime':time_coder,'leadtime':False})
        ff=xr.decode_cf(ff,decode_times=time_coder,decode_timedelta=False)
        ff=ff.where((ff.reftime>=cftime.DatetimeNoLeap(climyrs[0],1,1))&(ff.reftime<cftime.DatetimeNoLeap(climyrs[-1]+1,1,1)),drop=True)
        fc=ff.sst_an.coarsen(reftime=12,boundary='pad').construct(reftime=('year','month'))
        #fc=fc.chunk({'lat':10,'lon':10})
        sh=fc.shape
        ql1=np.nan*np.ones((12,sh[-2],sh[-1]))
        #ql2=np.nan*np.ones((12,sh[-2],sh[-1]))
        for ii in range(0,12):
            #if delt<20:
            #    pool1=fc.isel(month=ii).data.reshape((sh[0]*sh[1]*sh[3],sh[4],sh[5]))#.rechunk((-1,10,10))
            #    ql1[ii,...]=da.apply_along_axis(np.nanquantile,0,pool1,qtile).compute(scheduler='processes')
            #    #pool2=fc.sel(month=getind(ii)).data.reshape((sh[0]*sh[1]*3*sh[3],sh[4],sh[5])).rechunk((-1,10,10))
            #    #ql2[ii,...]=da.apply_along_axis(np.quantile,0,pool2,qtile).compute()
            #else:
            gr=2
            for ij in range(0,int(np.ceil(sh[-2]/gr))):
                pool1=fc.isel(month=ii,lat=slice(ij*gr,(ij+1)*gr)).data.reshape((sh[0]*sh[1]*sh[3],gr,sh[5]))
                ql1[ii,ij*gr:(ij+1)*gr,:]=da.apply_along_axis(np.nanquantile,0,pool1,qtile).compute(scheduler='processes')
    print(fqout,flush=True)
    dsqt=xr.Dataset(data_vars={'qt1':(['month','lat','lon'],ql1,{'long_name':f"{100*qtile}th percentile value"}),},
                               # 'qt2':(['month','lat','lon'],ql2,{'long_name':f"{100*qtile}th percentile value"}),},
                   coords={'month':np.arange(0,12),
                           'lat':ff.lat,
                           'lon':ff.lon})
    dsqt.to_netcdf(fqout,mode='w')
    del dsqt; del fc; del ql1; del ql2; del pool1; #del pool2;
    ff.close()
    return

def MHW_calc(climyrs,ilead,jj,qtile,detr=True,smoothedClim=False,smoothedTrend=False,smoothmethod=None,window=1,delt=0,qtvar='qt1',L=1):
    fanom=fnameCanESMAnomByLead(workdir[L], climyrs[0], climyrs[-1], ilead, jj,smoothClim=smoothedClim,smoothTrend=smoothedTrend,meth=smoothmethod,win=window,L=L,detrended=detr) 
    fqtile=fnameCanESMAnomQtile(workdir[L], climyrs[0], climyrs[-1], ilead, jj, qtile,detr,smoothClim=smoothedClim,
                                smoothTrend=smoothedTrend,meth=smoothmethod,win=window,delt=delt,L=L)
    fMHW=fnameCanESMMHW(workdir[L], climyrs[0], climyrs[-1], ilead, jj,qtile,detr,smoothClim=smoothedClim,
                                smoothTrend=smoothedTrend,meth=smoothmethod,win=window,delt=delt,qtvar=qtvar,L=L)
    print(fMHW,flush=True)
    if os.path.exists(fMHW): return
    ff=xr.open_dataset(fanom,decode_times=False)
    fc=ff.sst_an.coarsen(reftime=12,boundary='pad').construct(reftime=('year','month')).values
    sh=fc.shape
    fq=xr.open_dataset(fqtile,decode_times=False)
    ql=fq[qtvar].values
    qt2=np.expand_dims(ql,[0,2])
    MHW=np.ma.masked_where(np.isnan(fc),np.where(fc>qt2,1,0))
    MHWstack=np.reshape(MHW,(sh[0]*sh[1],sh[2],sh[3],sh[4]))
    del MHW; del ql;
    MHWstack=MHWstack[:ff.sst_an.shape[0],...] # remove appended nans
    dsMHW=xr.Dataset(data_vars={'isMHW':(['reftime','r','lat','lon'],MHWstack),
                                'MHW_prob':(['reftime','lat','lon'],np.mean(MHWstack,axis=1))},
                    coords={'reftime':ff.reftime,'r':ff.r,'lat':ff.lat,'lon':ff.lon,'leadtime':ff.leadtime})
    del MHWstack
    mkdirs(fMHW)
    dsMHW.to_netcdf(fMHW,mode='w')
    del dsMHW;
    ff.close(); fq.close(); del ff; del qt2; 
    return

def regrid_daily_OISST(yrlims,L=1):
    flistD=[]
    for iy in range(yrlims[0],yrlims[1]+1):
        for im in range(1,13):
            if iy<2024 | (iy==2024 and im<7): # data provisional/not downloaded from July on
               flistD.append(fnameOISSTDaily(iy,im))
    fD=xr.open_mfdataset(flistD,parallel=True,decode_times=False)
    data={}
    for ivar in ['sst','ice']:
        data[ivar]=fD[ivar].coarsen({'time':L,'lat':4,'lon':4}).mean().data[:,0,:,:]
    data0=fD['err']**2
    data['err']=data0.coarsen({'lat':4,'lon':4}).mean().data[:,0,:,:]**(1/2)
    data['lat']=fD.lat.coarsen({'lat':4}).mean()
    data['lon']=fD.lon.coarsen({'lon':4}).mean()
    data['time']=fD.time.coarsen({'time':L}).mean()
    dsout=xr.Dataset(data_vars={'sst':(('time','lat','lon'),data['sst'],fD.sst.attrs),
                                'ice':(('time','lat','lon'),data['ice'],fD.ice.attrs),
                                'err':(('time','lat','lon'),data['err'],fD.err.attrs)},
                     coords={'time':data['time'],'lat':data['lat'],'lon':data['lon']})
    fout=fnameOISSTGrid2(yrlims,L)
    mkdirs(fout)
    dsout.to_netcdf(fout,'w')
    fD.close()
    return

def daily_to_5day_OISST(yrlims):
    fin=fnameOISSTGrid2(yrlims,L=1)
    fout=fnameOISSTGrid2(yrlims,L=5)
    if not os.path.exists(fout):
        print(fout,flush=True)
        ff=xr.open_dataset(fin).chunk({'lat':30,'lon':30}).convert_calendar('noleap')
        ff=ff.isel(time=slice(0,int(len(ff.time)/5)*5))
        print(ff)
        ff1=ff.coarsen(time=5).mean() # 5 days
        ff1['sst']=ff1.sst.assign_attrs({'postprocess':'remove leap days and 5d time average'})
        ff1['ice']= ff1.ice.assign_attrs({'postprocess':'remove leap days and 5d time average'})
        mkdirs(fout)
        ff1.to_netcdf(fout,mode='w')
        for fff in [ff1, ff]:
            fff.close()
    return
    
#def calc_OISST_clim(climyrs,L=1):
#    flist=[fnameOISSTGrid2(yrlims,L=L) for yrlims in ylimlistobs]
#    fg2=xr.open_mfdataset(flist,decode_times=False,parallel=True)
#    sst=fg2.sst.data.rechunk((len(fg2.time.values),90,90))
#    tdt=np.array([dt.datetime(1978,1,1,12)+dt.timedelta(days=float(el)) for el in fg2.time.values])
#    yd=np.array([yd365(el) for el in tdt]) # day 366 is returned as 365
#    climsst=np.zeros((365,180,360))
#    for iyd in range(1,366):
#        ind=yd==iyd
#        indyrs=np.array([(el.year>=climyrs[0])&(el.year<=climyrs[-1]) for el in tdt])
#        ind=np.logical_and(ind,indyrs)
#        climsst[iyd-1,:,:]=sst[ind,:,:].mean(axis=0)
#        if iyd%10==0: print(iyd)
#    ds=xr.Dataset(data_vars={'sst':(['yearday','lat','lon'],climsst)},
#                             coords={'yearday':np.arange(1,366),
#                                     'lat':fg2.lat,
#                                     'lon':fg2.lon})
#    fout=fnameOISSTDailyClim(climyrs[0],climyrs[-1],smoothedClim=False,L=L)
#    ds.to_netcdf(fout,mode='w') 
#    return

def calc_OISST_clim(climyrs,L=1):
    flist=[fnameOISSTGrid2(yrlims,L=L) for yrlims in ylimlistobs]
    fg2=xr.open_mfdataset(flist,decode_times=True,parallel=True)
    fg2=fg2.where((fg2.time>=cftime.DatetimeNoLeap(climyrs[0],1,1))&(fg2.time<cftime.DatetimeNoLeap(climyrs[-1]+1,1,1)),drop=True)
    sst=fg2.sst.data.rechunk((len(fg2.time.values),90,90))
    #tdt=np.array([cftime.DatetimeNoLeap(climyrs[0],1,1)+dt.timedelta(days=ii) for ii in np.arange(L/2,len(fg2.time.values)*L,L)])
    yl=int(365/L)
    yd=np.array([yd365(ii) for ii in fg2.time.values])
    climsst=np.zeros((yl,180,360))
    for ix, iyd in enumerate(yd[:yl]):
        ind=yd==iyd
        climsst[ix,:,:]=sst[ind,:,:].mean(axis=0)
        if iyd%10==0: print(iyd)
    ds=xr.Dataset(data_vars={'sst':(['yearday','lat','lon'],climsst)},
            coords={'yearday':yd[:yl],
                                     'lat':fg2.lat,
                                     'lon':fg2.lon})
    fout=fnameOISSTDailyClim(climyrs[0],climyrs[-1],smoothedClim=False,L=L)
    ds.to_netcdf(fout,mode='w') 
    return

def smooth_OISST_clim(climyrs,smoothmeth,windowhalf,L=1):
    fclim=xr.open_dataset(fnameOISSTDailyClim(climyrs[0],climyrs[-1],smoothedClim=False,L=L),decode_times=False)
    climS=da.empty_like(fclim.variables['sst'].values)
    smoothClim=trismooth(np.arange(0,365,L),fclim['sst'].values,L=windowhalf,periodic=True)
    fout=fnameOISSTDailyClim(climyrs[0],climyrs[-1],smoothedClim=True,meth=smoothmeth,win=windowhalf,L=L)
    ds=xr.Dataset(data_vars={'sst':(['yearday','lat','lon'],smoothClim)},
                             coords={'yearday':fclim.yearday,
                                     'lat':fclim.lat,
                                     'lon':fclim.lon})
    ds.to_netcdf(fout,mode='w')    
    return

def OISST_anom(yrlims,climyrs,smoothClim=False, meth=None, win=1,L=1):
    climpath=fnameOISSTDailyClim(climyrs[0],climyrs[-1],smoothClim,meth,win,L=L)
    fclim=xr.open_dataset(climpath)
    ifile=fnameOISSTGrid2(yrlims,L)
    fsst=xr.open_dataset(ifile,decode_times=True,chunks={'time':365,})
    yl=int(365/L)
    yd=np.array([yd365(ii) for ii in fsst.time.values])
    sst_an=np.empty(np.shape(fsst.sst.values))
    # Loop over time
    for ind, iyd in enumerate(yd):
        sst_an[ind,...] = fsst.sst.values[ind,...] - fclim.sst.sel(yearday=iyd).values
    for jj in range(0,180,60):
        fout=fnameOISSTAnom(yrlims,climyrs, jj, smoothClim, meth, win,L)
        dsout=xr.Dataset(data_vars={'sst_an':(['time','lat','lon'],sst_an[:,jj:jj+60,:])},
                         coords={'time':fsst.time,
                                 'lat':fsst.lat.isel(lat=slice(jj,jj+60)),
                                 'lon':fsst.lon})
        dsout.to_netcdf(fout,mode='w')
        print(fout)
    return

def OISST_anom_detr(climyrs,smoothClim=False, meth=None, win=1,L=1):
    for jj in range(0,180,60):
        flist=[fnameOISSTAnom(yrlims, climyrs, jj, smoothClim, meth, win,L) for yrlims in ylimlistobs]
        fanom=xr.open_mfdataset(flist,decode_times=True,parallel=True)
        #fanom2=fanom.sel(time=slice(cftime.DatetimeNoLeap(climyrs[0],1,1),cftime.DatetimeNoLeap(climyrs[-1],12,31,23,59,59)))
        fanom2=fanom.where((fanom.time>=cftime.DatetimeNoLeap(climyrs[0],1,1))&(fanom.time<cftime.DatetimeNoLeap(climyrs[-1]+1,1,1)),drop=True)
        b=lsqfit_md_detr_calcb(fanom2.sst_an.values)
        tref=fanom2.time.values[0]
        fbout=fnameOISSTDetrFit(climyrs,jj,smoothClim,meth,win,L)
        # save fbout
        b=np.reshape(b,tuple([2]+list(fanom2.sst_an.values.shape)[1:]))
        dsb=xr.Dataset(data_vars={'fit':(['b','lat','lon'],b),'tref':tref},
                   coords={'b':np.arange(0,2),
                           'lat':fanom2.lat,
                           'lon':fanom2.lon})
        dsb.to_netcdf(fbout,mode='w')
        offset=(fanom.time.values[0]-tref).total_seconds()/(24*3600*L)
        sst_an=lsqfit_md_detr_applyb(fanom.sst_an.values,b,offset)
        fout=fnameOISSTAnom([ylimlistobs[0][0],ylimlistobs[-1][-1]],climyrs, jj, smoothClim, meth, win,L,detrended=True)
        dsout=xr.Dataset(data_vars={'sst_an':(['time','lat','lon'],sst_an)},
                         coords={'time':fanom.time,
                                 'lat':fanom.lat,
                                 'lon':fanom.lon})
        dsout.to_netcdf(fout,mode='w')
    return

def calc_quantile_OISST(climyrs,jj,qtile,detr=True,smoothClim=False,meth=None,win=1,delt=0,L=1):
    # delt windows in year-day (qt1 and qt2)
    # qt2 is also +/1 1 month
    #def getind(i0):
    #    if i0>=1 and i0<=10:
    #        return [i0-1,i0,i0+1]
    #    elif i0==0:
    #        return [11,0,1]
    #    elif i0==11:
    #        return [10,11,0]
    def _ix(ii,yd):
        return np.logical_or(np.logical_or((yd>=ii-delt)&(yd<=ii+delt),
                                           (yd-365>=ii-delt)&(yd-365<=ii+delt)),
                                           (yd+365>=ii-delt)&(yd+365<=ii+delt))
    # def leadbounds(l0,lmax,delt):
    #     i0=min(max(l0-delt,0),lmax-(2*delt+1))
    #     return i0, i0+2*delt+1
    if detr:
        flist=[fnameOISSTAnom([ylimlistobs[0][0],ylimlistobs[-1][-1]],climyrs, jj, smoothClim, meth, win,L,detrended=True),]
    else:
        flist=[fnameOISSTAnom(yrlims, climyrs, jj, smoothClim, meth, win,L,detrended=False) for yrlims in ylimlistobs]
    print(flist)
    ff=xr.open_mfdataset(flist,parallel=True,decode_times=True)
    ff=ff.where((ff.time>=cftime.DatetimeNoLeap(climyrs[0],1,1))&(ff.time<cftime.DatetimeNoLeap(climyrs[-1]+1,1,1)),drop=True)
    #tdt=np.array([dt.datetime(1978,1,1,12)+dt.timedelta(days=float(el)) for el in ff.time.values])
    #yy=[el.year for el in ff.time.values]
    #iy=int(np.argmax(np.array(yy)>climyrs[-1])) # index of first date outside climatology period
    #ff=ff.isel(time=slice(0,iy))
    vals=ff['sst_an'].values
    #tdt=tdt[:iy]
    yl=int(365/L)
    yd=np.array([yd365(ii) for ii in ff.time.values])
    ql1=np.zeros((yl,)+np.shape(ff.sst_an.values)[1:])
    #ql2=np.zeros((365,)+np.shape(ff.sst_an.values)[1:])
    for ii in range(1,yl+1): # loop through yds
        ix1=_ix(ii,yd)
        pool1=vals[ix1,:,:]
        ql1[ii-1,...]=np.nanquantile(pool1,qtile,axis=0)
        #ix2=np.logical_or(np.logical_or(ix1,_ix(ii-30,yd)),_ix(ii+30,yd)) # use 30 day rather than specfic months;
        #                                                                  # for comparison with Jacox monthly version
        #pool2=vals[ix2,:,:]
        #ql2[ii-1,...]=np.nanquantile(pool2,qtile,axis=0)
    fqout = fnameOISSTQTile(climyrs, jj, qtile, smoothClim, meth, win,detr,delt,L)
    print(fqout,flush=True)
    dsqt=xr.Dataset(data_vars={'qt1':(['yd','lat','lon'],ql1,{'long_name':f"{100*qtile}th percentile value"})},
                               #'qt2':(['yd','lat','lon'],ql2,{'long_name':f"{100*qtile}th percentile value"}),},
                               coords={'yd':yd[:yl],
                                       'lat':ff.lat,
                                       'lon':ff.lon})
    dsqt.to_netcdf(fqout,mode='w')
    del dsqt; 
    ff.close()
    return

def MHW_calc_OISST(climyrs,jj,qtile,detr=True,smoothClim=False,meth=None,win=1,delt=0,qtvar='qt1',L=1):
    if detr:
        flist=[fnameOISSTAnom([ylimlistobs[0][0],ylimlistobs[-1][-1]],climyrs, jj, smoothClim, meth, win,L,detrended=True),]
    else:
        flist=[fnameOISSTAnom(yrlims, climyrs, jj, smoothClim, meth, win,L) for yrlims in ylimlistobs]
    print(flist)
    fanom=xr.open_mfdataset(flist,parallel=True,decode_times=True)
    fqtile= fnameOISSTQTile(climyrs, jj, qtile, smoothClim, meth, win,detr,delt,L)
    fMHW = fnameOISSTMHW(climyrs, jj, qtile, smoothClim, meth, win,detr,delt,qtvar,L)
    print(fMHW,flush=True)
    yl=int(365/L)
    yd=np.array([yd365(ii) for ii in fanom.time.values])
    #tdt=np.array([dt.datetime(1978,1,1,12)+dt.timedelta(days=float(el)) for el in fanom.time.values])
    #yd=[yd365(el) for el in tdt]
    fq=xr.open_dataset(fqtile,decode_times=False)
    qt2=fq[qtvar].sel(yd=yd)
    MHW=np.ma.masked_where(np.logical_or(np.isnan(fanom['sst_an'].values),np.isnan(fanom['sst_an'].values)),
                       np.where(fanom['sst_an'].values>qt2.values,1,0))
    dsMHW=xr.Dataset(data_vars={'isMHW':(['time','lat','lon'],MHW),},
                    coords={'time':fanom.time,'lat':fanom.lat,'lon':fanom.lon})
    mkdirs(fMHW)
    dsMHW.to_netcdf(fMHW,mode='w')
    del dsMHW; del MHW; del qt2; 
    fanom.close(); fq.close();
    del fanom; del fq;
    return

class compstats:
    def __init__(self,forfile,obsfile,leaddays):
        self.forfile=forfile
        self.obsfile=obsfile
        self.ffor=xr.open_dataset(forfile)
        self.fobs=xr.open_dataset(obsfile)
        tsel=self.ffor.reftime.values+np.timedelta64(leaddays,'D')
        tsel=tsel[tsel<self.fobs.time.values[-1]]
        self.mhwfor=self.ffor['isMHW'].isel(reftime=slice(0,len(tsel))).data
        self.mhwobs=self.fobs['isMHW'].sel(time=tsel,method='nearest',tolerance=np.timedelta64(12,'h')).data
    def calcSEDI(self):
        self.SEDI,self.lmask,self.TP,self.TN,self.FP,self.FN = calc_SEDI(self.mhwfor,self.mhwobs)
    def saveSEDI(self,filename):
        dsout=xr.Dataset(data_vars={'SEDI':(['lat','lon'],self.SEDI),
                                    'lmask':(['lat','lon'],self.lmask),
                                    'TP':(['lat','lon'],self.TP),
                                    'FP':(['lat','lon'],self.FP),
                                    'TN':(['lat','lon'],self.TN),
                                    'FN':(['lat','lon'],self.FN),},
                         coords={'lat':self.ffor.lat,'lon':self.ffor.lon},
                         attrs={'forecast file':self.forfile,
                                'obs file':self.obsfile})
        dsout.to_netcdf(filename,mode='w')
    def closefiles(self):
        self.ffor.close()
        self.fobs.close()
    def __repr__(self):
        xx=dir(self)
        xx=[el for el in xx if not el.startswith('__')]
        return 'compstats: '+' '.join(xx)

if __name__=="__main__":
    # argument options:
    # - python MHW_daily_calcs.py fconvert_CanESM startyear endyear
    # - python MHW_daily_calcs.py calcAnom_CanESM5 climfirstyear climlastyear
    funx=sys.argv[1] # what function to execute
    ncpu=len(os.sched_getaffinity(0))
    climyrs=[1993,2022]
    #method='tri'
    halfwin=10
    qtile=.9
    L=5 # 5 days
    smoothclim=True
    smoothedTrend=True
    smoothmethod='tri'
    delt=15 # window for quantile selection
    detr=True # default
    qtvar='qt1'
    if funx=='fconvert_CanESM':
        starty=int(sys.argv[2])
        endy=int(sys.argv[3])
        years=[starty,endy]
        dd=1
        hh=0
        for yyyy in range(years[0],years[1]):
            for mm in range(1,13):
                if yyyy==2024 and mm>6:
                    pass
                else:
                    fconvert_CanESM(yyyy,mm,dd,hh)
    elif funx=='fconvert_CanESM_1d5d': # ~12 hrs 
        yyyy=int(sys.argv[2])
        dd=1
        hh=0
        for mm in range(2,13):#1,13
            if (yyyy==2024 and mm>6) or (yyyy>2024) or (yyyy<1993):
                pass
            else:
                fconvert_CanESM_1d5d(yyyy,mm)
    elif funx=='calcClim_CanESM5': # ~5 min
        smoothclim=False
        calcClim_CanESM5(climyrs,L=L)
    elif funx=='smoothClim_CanESM5': # ~1 min
        # request 7 cpus
        smoothClim_CanESM5(climyrs,smoothmethod,halfwin,L=L)
    elif funx=='calcAnom_CanESM5': # 20 min
        mm=int(sys.argv[2])
        calcAnom_CanESM5(climyrs,mm,smoothclim,smoothmethod,halfwin,L=L)#,nlead)
    elif funx=='anom_bylead': # ~60 min
        ilead=int(sys.argv[2])
        #nleads=range(0,int(215/L)) # calculate for all leads
        #startyr=1993
        anom_bylead(climyrs,ilead,smoothclim,smoothmethod,halfwin,L=L)
    elif funx=='anom_bylead_savetr':
        smoothedTrend=True
        ind=int(sys.argv[2]) # argument should be index, currently in range of 0 to 42
        #nleads=215
        if ind*5<215/L:
            for ilead in range(ind*5,min((ind+1)*5,int(215/L))):
                for jj in range(0,180,60):
                    anom_bylead_savetr(climyrs,ilead,jj,smoothclim,smoothmethod,halfwin,L=L)
    elif funx=='smoothTrend_CanESM5':
        yind=int(sys.argv[2])*60 # 0, 60, or 120
        smoothTrend_CanESM5(yind,climyrs,smoothclim,smoothmethod,halfwin,L=L)
    elif funx=='anom_bylead_detr':
        ind=int(sys.argv[2]) # argument should be index, currently in range of 0 to 42
        #nleads=215
        if ind*5<215/L:
            for ilead in range(ind*5,min((ind+1)*5,int(215/L))):
                for jj in range(0,180,60):
                        anom_bylead_detr(climyrs,ilead,jj,smoothclim,smoothedTrend,smoothmethod,halfwin,L=L)
    elif funx=='calc_quantile_CanESM':
        ind=int(sys.argv[2]) # argument should be index, currently in range of 0 to 42
        #for delt in (15,):#,30): #0,5,10,15,30
        if ind*5<215/L:
            for ilead in range(ind*5,min((ind+1)*5,int(215/L))):
                for jj in range(0,180,60):
                    calc_quantile_CanESM(climyrs,ilead,jj,qtile,detr,smoothclim,smoothedTrend,smoothmethod,halfwin,delt,L=L)
    elif funx=='MHW_calc':
        ind=int(sys.argv[2]) # index, 0 to 42
        smoothedClim=True
        smoothedTrend=True
        detrended=True
        delt=15
        if ind*5<215/L:
            for ilead in range(ind*5,min((ind+1)*5,int(215/L))):
                for jj in range(0,180,60):
                    print(f'start {ilead},{jj},{qtile}')
                    MHW_calc(climyrs,ilead,jj,qtile,detrended,smoothedClim,smoothedTrend,
                                         smoothmethod,halfwin,delt,'qt1',L=L)
        #opt=int(sys.argv[3]) # number referencing option set
        #qtvarname=sys.argv[4] # qt1 or qt2; qt1 is 1 month, qt2 is 3 month (at same lead)
        #delt=int(sys.argv[5]) # delt
        #det=int(sys.argv[6])
        #detr=True if det==1 else False
        #if not delt in {0,5,10,15,30}: raise Exception('check delt')
        #if opt==0: # no smoothing
        #    smoothedClim=False
        #    smoothedTrend=False
        #    smoothmethod=None
        #    window=0
        #elif opt==1: # all smoothing
        #    smoothedClim=True
        #    smoothedTrend=True if detr else False
        #    smoothmethod=smoothmethod
        #    window=halfwin
        #for ilead in range(ind*5,(ind+1)*5):
        #    for jj in range(0,180,60):
        #        print(f'start {ilead},{jj},{qtile}')
        #        MHW_calc(climyrs,ilead,jj,qtile,detr,smoothedClim,smoothedTrend,
        #                                 smoothmethod,window,delt,qtvarname,L=L)
    elif funx=='regrid_daily_OISST':
        # after combining files with MHW_OISST/concatFiles.py
        for yrlims in ylimlistobs:
            regrid_daily_OISST(yrlims,L=L)
    elif funx=='daily_to_5day_OISST':
        for yrlims in ylimlistobs:
            daily_to_5day_OISST(yrlims)
    elif funx=='calc_OISST_clim':
        calc_OISST_clim(climyrs,L=L)
        smooth_OISST_clim(climyrs,smoothmethod,halfwin,L=L)
    elif funx=='OISST_anom':
        seg=int(sys.argv[2])
        if seg>=len(ylimlistobs): 
            raise Exception('seg too high')
        else:
            yrlims=ylimlistobs[seg]
        OISST_anom(yrlims,climyrs,smoothclim, smoothmethod, halfwin,L=L)
    elif funx=='OISST_anom_detr':
        OISST_anom_detr(climyrs,smoothclim, smoothmethod, halfwin,L=L)
    elif funx=='calc_quantile_OISST':
        #for delt in (30,): #(5,10,15,
        print(f"delt={delt}",flush=True)
        print(f"detr={detr}",flush=True)
        for jj in range(0,180,60):
            print(f"jj={jj}",flush=True)
            calc_quantile_OISST(climyrs,jj,qtile,detr=detr,smoothClim=smoothclim,meth=smoothmethod,win=halfwin,delt=delt,L=L)
    elif funx=='MHW_calc_OISST':
        #for delt in (15,30):
        #    for detr in (False,): #False):
        for jj in range(0,180,60):
            MHW_calc_OISST(climyrs,jj,qtile,detr,smoothClim=smoothclim,meth=smoothmethod,win=halfwin,
                                    delt=delt,qtvar=qtvar,L=L)
    elif funx=='IndivCalcs':
        ## anomalies
        #for yrlims in ylimlistobs:
        #    OISST_anom(yrlims,climyrs)
        #print(f'anom saved yrlims:{yrlims}')
        ## remove trend
        #OISST_anom_detr(climyrs)
        # quantiles
        detr=True
        for delt in (15,): #(0,30):
            for jj in range(0,180,60):
                calc_quantile_OISST(climyrs,jj,qtile,detr=detr,delt=delt,L=L)
                MHW_calc_OISST(climyrs,jj,qtile,detr=detr,delt=delt,L=L)
    elif funx=='saveSEDI':
        ind=int(sys.argv[2]) # argument should be index, currently in range of 0 to 42
        smooth=int(sys.argv[3]) # 0 or 1
        delt=int(sys.argv[4]) # delt
        det=int(sys.argv[5])    # 0 or 1
        detr=True if det==1 else False
        if smooth==1:
            smoothedClim=True #False #True
            win=halfwin #0
        else:
            smoothedClim=False #True
            win=0
        qtvar='qt1'
        smoothTrend=True if (smoothedClim and detr) else False
        for ilead in range(ind*5,(ind+1)*5):
            for jj in range(0,180,60):
                print(f'start {detr},{ilead},{jj},{qtile}',flush=True)
                pathobs=fnameOISSTMHW(climyrs,jj,qtile,smoothedClim,smoothmethod,win,detr,delt,qtvar,L=L)
                pathfor=fnameCanESMMHW(workdir[L],climyrs[0],climyrs[-1],ilead,jj,qtile,detr,smoothedClim,smoothTrend,smoothmethod,win,delt,qtvar,L=L)
                fout=fnameSEDI_OISST_CanESM_daily(ilead,climyrs, smoothedClim, smoothmethod, halfwin, detr, qtile, delt, qtvar, jj,L=L)
                if os.path.exists(fout):
                    pass
                else:
                    iSEDI=compstats(pathfor,pathobs,ilead,L=L)
                    iSEDI.calcSEDI()
                    iSEDI.saveSEDI(fout)
                    iSEDI.closefiles()
                    print(fout,flush=True)
    elif funx=='saveReli':
        ind=int(sys.argv[2]) # index from job array, should be adjusted to range of leadlist
        leadlist=[50,75,100,125,200]#[0, 1, 3, 6, 10, 15, 20, 30]
        detr=True
        smoothClim=True
        smoothTrend=True
        meth=method
        win=halfwin
        delt=15 
        qtvar='qt1'
        region='global'
        mcount, ocount, ps = reliability1(climyrs,leadlist[ind],qtile,detr,smoothClim,smoothTrend,
                                        meth,win,delt,qtvar,region,L=L)
        np.savez(fnameReli(leadlist[ind],climyrs, smoothClim, meth, win, detr, qtile, delt,qtvar,region,L=L),
                 mcount=mcount,ocount=ocount,ps=ps)
    print('Done')

