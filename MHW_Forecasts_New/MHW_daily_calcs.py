import os, sys
import datetime as dt
import xarray as xr
import numpy as np
from dask.distributed import Client, LocalCluster
import dask.array as da
from mhw_daily_paths import * # break script out into smaller files
from mhw_daily_stats import * # break script out into smaller files

ylimlistobs=[[1991,2000],[2001,2010],[2011,2020],[2021,2024]]
method='tri'
halfwin=10
qtile=.9

def mkdirs(fsave):
    saveloc=os.path.dirname(fsave)
    if not os.path.exists(saveloc):
        try:
            os.makedirs(saveloc)
        except FileExistsError:
            pass # in case other code running at the same time got to it first
    return

def yd365(tdt):
    yd=int((dt.datetime(tdt.year,tdt.month,tdt.day)-dt.datetime(tdt.year-1,12,31)).days) # extra code in case of time components
    if yd==366: yd=365 # move leap days to overlap with day 365
    return yd

def lsqfit_md_detr(data):
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

def calcClim_CanESM5(climyrs):#,nlead):
    for mm in range(1,13): # month loop
        print(f"Month:{mm} {dt.datetime.now()}",flush=True)
        #for ix in range(0,int(nlead/5)):
        flist=[fnameCanESMdaily(mdirC5,yyyy,mm,1,0) for yyyy in range(climyrs[0],climyrs[-1]+1)] 
        fnameclim=fnameCanESMClim(workdir,climyrs[0],climyrs[-1],mm)
        mkdirs(fnameclim)
        with LocalCluster(n_workers=ncpu-1,threads_per_worker=1) as cluster, Client(cluster) as client:
            ff=xr.open_mfdataset(flist,parallel=True,combine='nested',concat_dim='reftime',
                           chunks={'reftime':-1,'leadtime':-1,'r':-1,'lat':90,'lon':180})
            EClim=ff.tso.mean(dim='r',skipna=False).mean(dim='reftime',skipna=False)
            EClim.to_netcdf(fnameclim,mode='w')
            del EClim
            ff.close()
    return

def smoothClim_CanESM5(climyrs,smoothmethod,window):
    with LocalCluster(n_workers=6,threads_per_worker=1) as cluster, Client(cluster) as client:
        flistclim = [fnameCanESMClim(workdir,climyrs[0],climyrs[-1],mm) for mm in range(1,13)]
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
            fout=fnameCanESMClimSmooth(workdir,climyrs[0],climyrs[-1],mm,smoothmethod,window)
            print(fout)
            dsout=xr.Dataset(data_vars={'tso':(['leadtime','lat','lon'],climS[mm-1,...])},
                             coords={'leadtime':fclim.leadtime,
                                     'lat':fclim.lat,
                                     'lon':fclim.lon})
            dsout.to_netcdf(fout,mode='w')
        fclim.close()
    return

def calcAnom_CanESM5(climyrs,smoothClim=False,smoothmethod=None,window=1):#,nlead):
    for mm in range(1,13): # month loop
        print(f"Month:{mm} {dt.datetime.now()}",flush=True)
        #for ix in range(0,int(nlead/5)):
        if smoothClim:
            fnamelast=fnameCanESMAnomSClim(workdir,climyrs[0],climyrs[-1],climyrs[-1],mm,smoothmethod,window)
        else:
            fnamelast=fnameCanESMAnom(workdir,climyrs[0],climyrs[-1],climyrs[-1],mm)
        if not os.path.exists(fnamelast): # skip if file at (almost) end already exists
            flist=[fnameCanESMdaily(mdirC5,yyyy,mm,1,0) for yyyy in range(1993,2025) if yyyy<2024 or mm<=6] # stop at Jul 2024
            if smoothClim:
                fnameclim=fnameCanESMClimSmooth(workdir,climyrs[0],climyrs[-1],mm,smoothmethod,window)
            else:
                fnameclim=fnameCanESMClim(workdir,climyrs[0],climyrs[-1],mm)
            with LocalCluster(n_workers=ncpu-1,threads_per_worker=1) as cluster, Client(cluster) as client:
                with xr.open_mfdataset(flist,parallel=True,combine='nested',concat_dim='reftime',
                               chunks={'reftime':-1,'leadtime':-1,'r':-1,'lat':-1,'lon':-1}) as ff:
                    # if not os.path.exists(fnameclim):
                    #     EClim=ff.tso.sel(reftime=slice(np.datetime64(f'{climyrs[0]:04}-01-01'),
                    #         np.datetime64(f'{climyrs[-1]:04}-12-31'))).mean(dim='r').mean(dim='reftime')
                    #     mkdirs(fnameclim)
                    #     EClim.to_netcdf(fnameclim,mode='w')
                    #     del EClim
                    fclim=xr.open_dataset(fnameclim)
                    EClim=fclim['tso']
                    for iy in range(1993,2025):
                        if iy<2024 or mm<=6: # stop at Jul 2024
                            if smoothClim:
                                fname=fnameCanESMAnomSClim(workdir,climyrs[0],climyrs[-1],iy,mm,smoothmethod,window)
                            else:
                                fname=fnameCanESMAnom(workdir,climyrs[0],climyrs[-1],iy,mm)
                            print(fname,flush=True)
                            if mm==1 and iy==1993: 
                                mkdirs(fname)
                            Anom0=ff.tso.sel(reftime=np.datetime64(f'{iy:04}-{mm:02}-01'))-EClim
                            #Anom0.chunk(chunks={'leadtime':1}).rename('sst_an').to_netcdf(fname,
                            #        encoding={'sst_an': {'chunksizes': [Anom0.shape[0],1,30,360]}},mode='w')
                            Anom0.rename('sst_an').to_netcdf(fname,
                                    encoding={'sst_an': {'chunksizes': [Anom0.shape[0],1,30,360]}},mode='w')
                            del Anom0
                    del EClim
                    fclim.close()
    return

def anom_bylead(climyrs,nleads,smoothClim=False,smoothmethod=None,window=1):
    for ilead in nleads:
        for jj in range(0,180,60):
            if smoothClim:
                flist=[fnameCanESMAnomSClim(workdir,climyrs[0],climyrs[-1],yy,mm,smoothmethod,window) for yy in range(1993,2025) for mm in range(1,13) if yy<2024 or mm<=6]
            else:
                flist=[fnameCanESMAnom(workdir,climyrs[0],climyrs[-1],yy,mm) for yy in range(1993,2025) for mm in range(1,13) if yy<2024 or mm<=6]
            with LocalCluster(n_workers=ncpu-1,threads_per_worker=1) as cluster, Client(cluster) as client:
                ff= xr.open_mfdataset(flist,parallel=True,combine='nested',concat_dim='reftime',
                                        preprocess=lambda ff: ff.isel(leadtime=ilead,lat=slice(jj,jj+60)),decode_times=False)
                sst_an2=ff.sst_an.chunk({'reftime':ff.sst_an.shape[0],'r':20,'lat':30,'lon':360})
                # fix time
                reftime=[dt.datetime(yy,mm,1,0,0) for yy in range(1993,2025) for mm in range(1,13) if yy<2024 or mm<=6]
                time=[dt.datetime(yy,mm,1,0,0)+dt.timedelta(hours=float(ff.leadtime.values)) \
                            for yy in range(1993,2025) for mm in range(1,13) if yy<2024 or mm<=6]
                fout=xr.Dataset(data_vars={'sst_an':(['reftime','r','lat','lon'],sst_an2.data),
                                           'time':(['reftime',],time,{'long_name':'Real Time'})},
                                coords={'reftime':reftime,
                                        'r':np.arange(0,ff.sst_an.shape[1]),
                                        'lat':ff.lat,
                                        'lon':ff.lon})
                fnamout=fnameCanESMAnomByLeadNoDetr(workdir,climyrs[0],climyrs[-1],ilead,jj,smoothClim,smoothmethod,window)
                mkdirs(fnamout)
                print(fnamout)
                fout.to_netcdf(fnamout,mode='w') # encoding={'sst_an': {'chunksizes': [Anom0.shape[0],1,20,360]}}
                del sst_an2; del fout;
                ff.close(); del ff;
    return

def anom_bylead_savetr(climyrs,ilead,jj,smoothClim=False,smoothmethod=None,window=1):
    fin=fnameCanESMAnomByLeadNoDetr(workdir, climyrs[0], climyrs[-1], ilead, jj,smoothClim, smoothmethod, window)
    fout=fnameCanESMDetrFitByLead(workdir, climyrs[0], climyrs[-1], ilead, jj, smoothClim,smoothmethod,window)
    mkdirs(fout)
    print(fout)
    ff=xr.open_dataset(fin,decode_times=False)
    lsqfit_md_detrPooled_saveb(ff.reftime.values,ff.sst_an,climyrs,ilead,jj,ff.lat.values,ff.lon.values,fout)
    ff.close()
    return
    
def smoothTrend_CanESM5(yind,climyrs,smoothmethod,window):
    desstr=f'_ClimS{smoothmethod}{window}'
    if smoothmethod=='tri':
        smoothfun=trismooth
    else:
        raise Exception('method not implemented:',smoothmethod)
    flistbS=[fnameCanESMDetrFitByLead(workdir, climyrs[0],climyrs[-1], ilead, yind, smoothClim=False,smoothmethod=None,window=1) \
                for ilead in range(0,215)]
    fbS=xr.open_mfdataset(flistbS,combine='nested',concat_dim=['leadtime'],parallel=True,decode_times=False)
    borig=fbS.fit.data.rechunk([-1,-1,30,180])
    bsmooth=da.map_blocks(smoothfun,fbS.leadtime.values,borig,window,dtype=float) #here lead time is in days
    for ilead in range(0,215):
        fout=fnameCanESMDetrFitByLeadS(workdir, climyrs[0],climyrs[-1], ilead, yind, smoothmethod, window, desstr)
        print(fout)
        dsout=xr.Dataset(data_vars={'fit':(['b','lat','lon'],bsmooth[ilead,...])},
                    coords={'b':fbS.b,
                       'lat':fbS.lat,
                       'lon':fbS.lon})
        dsout.to_netcdf(fout,mode='w')
    fbS.close()
    return
# def smoothTrend_CanESM5(climyrs,smoothmethod,window):
#     with LocalCluster(n_workers=6,threads_per_worker=1) as cluster, Client(cluster) as client:
#         desstr=f'_ClimS{smoothmethod}{window}'
#         flistbS=[[fnameCanESMDetrFitByLead(workdir, climyrs[0],climyrs[-1], ilead, yind,desstr) \
#                 for yind in [0,60,120]] for ilead in range(0,215)]
#         fbS=xr.open_mfdataset(flistbS,combine='nested',concat_dim=['leadtime','lat'],parallel=True,decode_times=False)
#         borig=fbS.fit.data.rechunk([-1,-1,60,-1])
#         if smoothmethod=='tri':
#             smoothfun=trismooth
#         else:
#             raise Exception('method not implemented:',smoothmethod)
#         bsmooth=da.map_blocks(smoothfun,fbS.leadtime.values,borig,window,dtype=float) #here lead time is in days
#         for ilead in range(0,215):
#             for yind in [0,60,120]:
#                 fout=fnameCanESMDetrFitByLeadS(workdir, climyrs[0],climyrs[-1], ilead, yind, smoothmethod, window, desstr)
#                 print(fout)
#                 dsout=xr.Dataset(data_vars={'fit':(['b','lat','lon'],bsmooth[ilead,:,yind:yind+60,:])},
#                             coords={'b':fbS.b,
#                            'lat':fbS.lat.isel(lat=slice(yind,yind+60)),
#                            'lon':fbS.lon})
#                 dsout.to_netcdf(fout,mode='w')
#         fbS.close()
#     return

def anom_bylead_detr(climyrs,ilead,jj,smoothedClim=False,smoothedTrend=False,smoothmethod=None,window=1):
    # note: smoothedTrend implies smoothedClim, but can load unsmoothed trends from smoothed climatology-based anomalies
    if smoothedTrend:
        fb=fnameCanESMDetrFitByLeadS(workdir, climyrs[0],climyrs[-1], ilead, jj, smoothmethod, window, desstr)
    else:
        fb=fnameCanESMDetrFitByLead(workdir, climyrs[0],climyrs[-1], ilead, jj, smoothedClim,smoothmethod,window)
    fin=fnameCanESMAnomByLeadNoDetr(workdir, climyrs[0], climyrs[-1], ilead, jj, smoothedClim,smoothmethod,window)
    if smoothedTrend and not smoothedClim: raise Exception('Bad combination: smoothed trend without smoothed climatology')
    fout=fnameCanESMAnomDetrByLead(workdir, climyrs[0], climyrs[-1], ilead, jj,smoothClim=True,smoothTrend=True,meth=smoothmethod,win=window)
    # 3 options: no smoothing; smoothed clim and raw trend; smoothed clim and smoothed trend
    mkdirs(fout)
    ff=xr.open_dataset(fin,decode_times=False)
    ftr=xr.open_dataset(fb,decode_times=False)
    trest=ftr.fit.isel(b=0)+ff.reftime*ftr.fit.isel(b=1)
    sstanomdet=ff.sst_an-trest
    sstanomdet=sstanomdet.rename('sst_an')
    sstanomdet.to_netcdf(fout,mode='w')
    ff.close()
    ftr.close()
    return

def calc_quantile_CanESM(climyrs,ilead,jj,qtile,detr=True,smoothedClim=False,smoothedTrend=False,smoothmethod=None,window=1,delt=0):
     # version 1: 10 day windows in lead time
     lmax=215
     def getind(i0):
         if i0>=1 and i0<=10:
             return [i0-1,i0,i0+1]
         elif i0==0:
             return [11,0,1]
         elif i0==11:
             return [10,11,0]
     #def leadbounds(l0,lmax,delt):
     #    i0=min(max(l0-delt,0),lmax-(2*delt+1))
     #    return i0, i0+2*delt+1
     def leadbounds(l0,lmax,delt):
         return max(0,l0-delt), min(lmax,l0+delt+1)
     if detr:
         flist=[fnameCanESMAnomDetrByLead(workdir, climyrs[0],climyrs[-1],il,jj,smoothClim=smoothedClim,smoothTrend=smoothedTrend,meth=smoothmethod,win=window) \
                 for il in range(*leadbounds(ilead,215,delt))]
     else:
         flist=[fnameCanESMAnomByLeadNoDetr(workdir, climyrs[0], climyrs[-1], il, jj,smoothClim=smoothedClim,meth=smoothmethod,win=window) \
                 for il in range(*leadbounds(ilead,215,delt))]
     print(flist)
     fqout=fnameCanESMAnomQtile(workdir, climyrs[0], climyrs[-1], ilead, jj, qtile, detr, 
                                smoothClim=smoothedClim,smoothTrend=smoothedTrend,meth=smoothmethod,win=window,delt=delt)
     ff=xr.open_mfdataset(flist,combine='nested',concat_dim=['leadtime'],parallel=True,decode_times=False)
     fc=ff.sst_an.coarsen(reftime=12,boundary='pad').construct(reftime=('year','month'))
     sh=fc.shape
     ql1=np.nan*np.ones((12,sh[-2],sh[-1]))
     ql2=np.nan*np.ones((12,sh[-2],sh[-1]))
     for ii in range(0,12):
         pool1=fc.isel(month=ii).values.reshape((sh[0]*sh[1]*sh[3],sh[4],sh[5]))
         ql1[ii,...]=np.nanquantile(pool1,qtile,axis=0)
         pool2=fc.sel(month=getind(ii)).values.reshape((sh[0]*sh[1]*3*sh[3],sh[4],sh[5]))
         ql2[ii,...]=np.nanquantile(pool2,qtile,axis=0)
     print(fqout)
     dsqt=xr.Dataset(data_vars={'qt1':(['month','lat','lon'],ql1,{'long_name':f"{100*qtile}th percentile value"}),
                                'qt2':(['month','lat','lon'],ql2,{'long_name':f"{100*qtile}th percentile value"}),},
                    coords={'month':np.arange(0,12),
                            'lat':ff.lat,
                            'lon':ff.lon})
     dsqt.to_netcdf(fqout,mode='w')
     del dsqt; del fc;
     ff.close()
     return

def calc_quantile_CanESM30(climyrs,ilead,jj,qtile,detr=True,smoothedClim=False,smoothedTrend=False,smoothmethod=None,window=1,delt=0):
    lmax=215
    def getind(i0):
        if i0>=1 and i0<=10:
            return [i0-1,i0,i0+1]
        elif i0==0:
            return [11,0,1]
        elif i0==11:
            return [10,11,0]
    def leadbounds(l0,lmax,delt):
        return max(0,l0-delt), min(lmax,l0+delt+1)
    if detr:
        flist=[fnameCanESMAnomDetrByLead(workdir, climyrs[0],climyrs[-1],il,jj,smoothClim=smoothedClim,smoothTrend=smoothedTrend,meth=smoothmethod,win=window) \
                for il in range(*leadbounds(ilead,215,delt))]
    else:
        flist=[fnameCanESMAnomByLeadNoDetr(workdir, climyrs[0], climyrs[-1], il, jj,smoothClim=smoothedClim,meth=smoothmethod,win=window) \
                for il in range(*leadbounds(ilead,215,delt))]
    fqout=fnameCanESMAnomQtile(workdir, climyrs[0], climyrs[-1], ilead, jj, qtile, detr, 
                               smoothClim=smoothedClim,smoothTrend=smoothedTrend,meth=smoothmethod,win=window,delt=delt)
    if os.path.exists(fqout): return
    print(flist,flush=True)
    ff=xr.open_mfdataset(flist,combine='nested',concat_dim=['leadtime'],parallel=True,decode_times=False)
    fc=ff.sst_an.coarsen(reftime=12,boundary='pad').construct(reftime=('year','month'))
    fc=fc.chunk({'lat':10,'lon':10})
    sh=fc.shape
    ql1=np.nan*np.ones((12,sh[-2],sh[-1]))
    ql2=np.nan*np.ones((12,sh[-2],sh[-1]))
    for ii in range(0,12):
        if delt<20:
            pool1=fc.isel(month=ii).data.reshape((sh[0]*sh[1]*sh[3],sh[4],sh[5])).rechunk((-1,10,10))
            ql1[ii,...]=da.apply_along_axis(np.nanquantile,0,pool1,qtile).compute()
            #pool2=fc.sel(month=getind(ii)).data.reshape((sh[0]*sh[1]*3*sh[3],sh[4],sh[5])).rechunk((-1,10,10))
            #ql2[ii,...]=da.apply_along_axis(np.quantile,0,pool2,qtile).compute()
        else:
            gr=20
            for ij in range(0,int(np.ceil(sh[-2]/gr))):
                pool1=fc.isel(month=ii,lat=slice(ij*gr,(ij+1)*gr)).data.reshape((sh[0]*sh[1]*sh[3],gr,sh[5]))
                ql1[ii,ij*gr:(ij+1)*gr,:]=da.apply_along_axis(np.nanquantile,0,pool1,qtile).compute()
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

def MHW_calc(climyrs,ilead,jj,qtile,detr=True,smoothedClim=False,smoothedTrend=False,smoothmethod=None,window=1,delt=0,qtvar='qt1'):
    if detr:    
        fanom=fnameCanESMAnomDetrByLead(workdir, climyrs[0], climyrs[-1], ilead, jj,smoothClim=smoothedClim,smoothTrend=smoothedTrend,meth=smoothmethod,win=window) 
    else:
        fanom=fnameCanESMAnomByLeadNoDetr(workdir, climyrs[0], climyrs[-1], ilead, jj,smoothClim=smoothedClim,meth=smoothmethod,win=window)
    fqtile=fnameCanESMAnomQtile(workdir, climyrs[0], climyrs[-1], ilead, jj, qtile,detr,smoothClim=smoothedClim,
                                smoothTrend=smoothedTrend,meth=smoothmethod,win=window,delt=delt)
    fMHW=fnameCanESMMHW(workdir, climyrs[0], climyrs[-1], ilead, jj,qtile,detr,smoothClim=smoothedClim,
                                smoothTrend=smoothedTrend,meth=smoothmethod,win=window,delt=delt,qtvar=qtvar)
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

#def MHW_calc(climyrs,ilead,jj,qtile,detr=True):
#    if detr: # set path-defining fxns for detrended or non-detrended versions of calculation
#        ffunanom=fnameCanESMAnomDetrByLead # or fnameCanESMAnomDetrByLeadIndiv if switch method
#        ffunqtile=fnameCanESMAnomDetrQtile
#        ffunMHW=fnameCanESMMHWDetr
#    else:
#        ffunanom=fnameCanESMAnomByLead
#        ffunqtile=fnameCanESMAnomQtile
#        ffunMHW=fnameCanESMMHW
#    ff=xr.open_dataset(ffunanom(workdir, climyrs[0], climyrs[-1], ilead, jj),decode_times=False)
#    fc=ff.sst_an.coarsen(reftime=12,boundary='pad').construct(reftime=('year','month'))
#    sh=fc.shape
#    def getind(i0):
#        if i0>=1 and i0<=10:
#            return [i0-1,i0,i0+1]
#        elif i0==0:
#            return [11,0,1]
#        elif i0==11:
#            return [10,11,0]
#    ql=np.nan*np.ones((12,sh[-2],sh[-1]))
#    for ii in range(0,12):
#        pool=fc.sel(month=getind(ii)).values.reshape((sh[0]*3*sh[2],sh[3],sh[4]))
#        ql[ii,...]=np.nanquantile(pool,0.9,axis=0)
#    fqout=ffunqtile(workdir, climyrs[0], climyrs[-1], ilead, jj,qtile)
#    dsqt=xr.Dataset(data_vars={'qt':(['month','lat','lon'],ql,{'long_name':f"{100*qtile}th percentile value"}),},
#                   coords={'month':np.arange(0,12),
#                           'lat':ff.lat,
#                           'lon':ff.lon})
#    dsqt.to_netcdf(fqout,mode='w')
#    del dsqt
#    qt2=np.expand_dims(ql,[0,2])
#    del ql
#    MHW=np.ma.masked_where(np.isnan(fc),np.where(fc>qt2,1,0))
#    del fc; del qt2;
#    MHWstack=np.reshape(MHW,(sh[0]*sh[1],sh[2],sh[3],sh[4]))
#    del MHW
#    MHWstack=MHWstack[:ff.sst_an.shape[0],...] # remove appended nans
#    dsMHW=xr.Dataset(data_vars={'isMHW':(['reftime','r','lat','lon'],MHWstack),
#                                'MHW_prob':(['reftime','lat','lon'],np.mean(MHWstack,axis=1))},
#                    coords={'reftime':ff.reftime,'r':ff.r,'lat':ff.lat,'lon':ff.lon,'leadtime':ff.leadtime})
#    del MHWstack
#    fMHWout=ffunMHW(workdir, climyrs[0], climyrs[-1], ilead, jj,qtile)
#    mkdirs(fMHWout)
#    dsMHW.to_netcdf(fMHWout,mode='w')
#    del dsMHW;
#    ff.close(); del ff
#    return

def regrid_daily_OISST(yrlims):
    flistD=[]
    for iy in range(yrlims[0],yrlims[1]+1):
        for im in range(1,13):
            if iy<2024 | (iy==2024 and im<7): # data provisional/not downloaded from July on
               flistD.append(fnameOISSTDaily(iy,im))
    fD=xr.open_mfdataset(flistD,parallel=True,decode_times=False)
    data={}
    for ivar in ['sst','ice']:
        data[ivar]=fD[ivar].coarsen({'lat':4,'lon':4}).mean().data[:,0,:,:]
    data0=fD['err']**2
    data['err']=data0.coarsen({'lat':4,'lon':4}).mean().data[:,0,:,:]**(1/2)
    data['lat']=fD.lat.coarsen({'lat':4}).mean()
    data['lon']=fD.lon.coarsen({'lon':4}).mean()
    dsout=xr.Dataset(data_vars={'sst':(('time','lat','lon'),data['sst'],fD.sst.attrs),
                                'ice':(('time','lat','lon'),data['ice'],fD.ice.attrs),
                                'err':(('time','lat','lon'),data['err'],fD.err.attrs)},
                     coords={'time':fD.time,'lat':data['lat'],'lon':data['lon']})
    fout=fnameOISSTDailyGrid2(yrlims)
    mkdirs(fout)
    dsout.to_netcdf(fout,'w')
    fD.close()
    return

def calc_OISST_clim(climyrs):
    flist=[fnameOISSTDailyGrid2(yrlims) for yrlims in ylimlistobs]
    fg2=xr.open_mfdataset(flist,decode_times=False,parallel=True)
    sst=fg2.sst.data.rechunk((len(fg2.time.values),90,90))
    tdt=np.array([dt.datetime(1978,1,1,12)+dt.timedelta(days=float(el)) for el in fg2.time.values])
    yd=np.array([yd365(el) for el in tdt]) # day 366 is returned as 365
    climsst=np.zeros((365,180,360))
    for iyd in range(1,366):
        ind=yd==iyd
        indyrs=np.array([(el.year>=climyrs[0])&(el.year<=climyrs[-1]) for el in tdt])
        ind=np.logical_and(ind,indyrs)
        climsst[iyd-1,:,:]=sst[ind,:,:].mean(axis=0)
        if iyd%10==0: print(iyd)
    ds=xr.Dataset(data_vars={'sst':(['yearday','lat','lon'],climsst)},
                             coords={'yearday':np.arange(1,366),
                                     'lat':fg2.lat,
                                     'lon':fg2.lon})
    fout=fnameOISSTDailyClim(climyrs[0],climyrs[-1])
    ds.to_netcdf(fout,mode='w') 
    return

def smooth_OISST_clim(climyrs,smoothmeth,windowhalf):
    fclim=xr.open_dataset(fnameOISSTDailyClim(climyrs[0],climyrs[-1]),decode_times=False)
    climS=da.empty_like(fclim.variables['sst'].values)
    smoothClim=trismooth(np.arange(0,365),fclim['sst'].values,L=windowhalf,periodic=True)
    fout=fnameOISSTDailyClimSmooth(climyrs[0],climyrs[-1],smoothmeth,windowhalf)
    ds=xr.Dataset(data_vars={'sst':(['yearday','lat','lon'],smoothClim)},
                             coords={'yearday':np.arange(1,366),
                                     'lat':fclim.lat,
                                     'lon':fclim.lon})
    ds.to_netcdf(fout,mode='w')    
    return

def OISST_anom(yrlims,climyrs,smoothClim=False, meth=None, win=1):
    if smoothClim:
        climpath=fnameOISSTDailyClimSmooth(climyrs[0],climyrs[-1],meth,win)
    else:
        print('in OISST_anom, smoothClim=False')
        climpath=fnameOISSTDailyClim(climyrs[0],climyrs[-1])
    fclim=xr.open_dataset(climpath)
    print(yrlims)
    ifile=fnameOISSTDailyGrid2(yrlims)
    fsst=xr.open_dataset(ifile,decode_times=False,chunks={'time':365,})
    tdt=[dt.datetime(1978,1,1)+dt.timedelta(days=float(ii)) for ii in fsst.time.values]
    yd=[yd365(el) for el in tdt] # max 365
    sst_an=np.empty(np.shape(fsst.sst.values))
    # Loop over time
    for ind, iyd in enumerate(yd):
        if ind%100==0: print(ind)
        sst_an[ind,...] = fsst.sst.values[ind,...] - fclim.sst.values[iyd-1,...]
    for jj in range(0,180,60):
        fout=fnameOISSTAnom(yrlims,climyrs, jj, smoothClim, meth, win)
        dsout=xr.Dataset(data_vars={'sst_an':(['time','lat','lon'],sst_an[:,jj:jj+60,:])},
                         coords={'time':fsst.time,
                                 'lat':fsst.lat.isel(lat=slice(jj,jj+60)),
                                 'lon':fsst.lon})
        dsout.to_netcdf(fout,mode='w')
        print(fout)
    return

def OISST_anom_detr(climyrs,smoothClim=False, meth=None, win=1):
    for jj in range(0,180,60):
        flist=[fnameOISSTAnom(yrlims, climyrs, jj, smoothClim, meth, win) for yrlims in ylimlistobs]
        fanom=xr.open_mfdataset(flist,decode_times=False,parallel=True)
        sst_an=lsqfit_md_detr(fanom.sst_an.values)
        fout=fnameOISSTAnomDetr([ylimlistobs[0][0],ylimlistobs[-1][-1]],climyrs, jj, smoothClim, meth, win)
        dsout=xr.Dataset(data_vars={'sst_an':(['time','lat','lon'],sst_an)},
                         coords={'time':fanom.time,
                                 'lat':fanom.lat,
                                 'lon':fanom.lon})
        dsout.to_netcdf(fout,mode='w')
    return

def calc_quantile_OISST(climyrs,jj,qtile,detr=True,smoothClim=False,meth=None,win=1,delt=0):
    # delt windows in year-day (qt1 and qt2)
    # qt2 is also +/1 1 month
    def getind(i0):
        if i0>=1 and i0<=10:
            return [i0-1,i0,i0+1]
        elif i0==0:
            return [11,0,1]
        elif i0==11:
            return [10,11,0]
    def _ix(ii,yd):
        return np.logical_or(np.logical_or((yd>=ii-delt)&(yd<=ii+delt),
                                           (yd-365>=ii-delt)&(yd-365<=ii+delt)),
                                           (yd+365>=ii-delt)&(yd+365<=ii+delt))
    # def leadbounds(l0,lmax,delt):
    #     i0=min(max(l0-delt,0),lmax-(2*delt+1))
    #     return i0, i0+2*delt+1
    if detr:
        flist=[fnameOISSTAnomDetr([ylimlistobs[0][0],ylimlistobs[-1][-1]],climyrs, jj, smoothClim, meth, win),]
    else:
        flist=[fnameOISSTAnom(yrlims, climyrs, jj, smoothClim, meth, win) for yrlims in ylimlistobs]
    print(flist)
    ff=xr.open_mfdataset(flist,parallel=True,decode_times=False)
    tdt=np.array([dt.datetime(1978,1,1,12)+dt.timedelta(days=float(el)) for el in ff.time.values])
    yy=[el.year for el in tdt]
    iy=int(np.argmax(np.array(yy)>climyrs[-1])) # index of first date outside climatology period
    ff=ff.isel(time=slice(0,iy))
    vals=ff['sst_an'].values
    tdt=tdt[:iy]
    yd=np.array([(dt.datetime(el.year,el.month,el.day)-dt.datetime(el.year-1,12,31)).days for el in tdt])
    ql1=np.zeros((365,)+np.shape(ff.sst_an.values)[1:])
    ql2=np.zeros((365,)+np.shape(ff.sst_an.values)[1:])
    for ii in range(1,366):
        ix1=_ix(ii,yd)
        pool1=vals[ix1,:,:]
        ql1[ii-1,...]=np.nanquantile(pool1,qtile,axis=0)
        ix2=np.logical_or(np.logical_or(ix1,_ix(ii-30,yd)),_ix(ii+30,yd)) # use 30 day rather than specfic months;
                                                                          # for comparison with Jacox monthly version
        pool2=vals[ix2,:,:]
        ql2[ii-1,...]=np.nanquantile(pool2,qtile,axis=0)
    fqout = fnameOISSTQTile(climyrs, jj, qtile, smoothClim, meth, win,detr,delt)
    print(fqout,flush=True)
    dsqt=xr.Dataset(data_vars={'qt1':(['yd','lat','lon'],ql1,{'long_name':f"{100*qtile}th percentile value"}),
                               'qt2':(['yd','lat','lon'],ql2,{'long_name':f"{100*qtile}th percentile value"}),},
                   coords={'yd':np.arange(1,366),
                           'lat':ff.lat,
                           'lon':ff.lon})
    dsqt.to_netcdf(fqout,mode='w')
    del dsqt; 
    ff.close()
    return

def MHW_calc_OISST(climyrs,jj,qtile,detr=True,smoothClim=False,meth=None,win=1,delt=0,qtvar='qt1'):
    if detr:
        flist=[fnameOISSTAnomDetr([ylimlistobs[0][0],ylimlistobs[-1][-1]],climyrs, jj, smoothClim, meth, win),]
    else:
        flist=[fnameOISSTAnom(yrlims, climyrs, jj, smoothClim, meth, win) for yrlims in ylimlistobs]
    print(flist)
    fanom=xr.open_mfdataset(flist,parallel=True,decode_times=False)
    fqtile= fnameOISSTQTile(climyrs, jj, qtile, smoothClim, meth, win,detr,delt)
    fMHW = fnameOISSTMHW(climyrs, jj, qtile, smoothClim, meth, win,detr,delt,qtvar)
    print(fMHW,flush=True)
    # real data has leap years, so coarsen won't work; need to account for 366 day years as well
    tdt=np.array([dt.datetime(1978,1,1,12)+dt.timedelta(days=float(el)) for el in fanom.time.values])
    yd=[yd365(el) for el in tdt]
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
    climyrs=[1993,2023]
    windowhalfwid=halfwin
    smoothmethod=method
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
    elif funx=='calcClim_CanESM5':
        climstart=int(sys.argv[2])
        climend=int(sys.argv[3])
        calcClim_CanESM5([climstart,climend])
    elif funx=='smoothClim_CanESM5':
        # request 7 cpus
        smoothClim_CanESM5(climyrs,smoothmethod,windowhalfwid)
    elif funx=='calcAnom_CanESM5':
        smoothclim=int(sys.argv[2])
        #nlead=215
        if smoothclim==1:
            calcAnom_CanESM5(climyrs,True,smoothmethod,windowhalfwid)#,nlead)
        else:
            calcAnom_CanESM5(climyrs)
    elif funx=='anom_bylead':
        climyrs=[1993,2023]
        smoothclim=int(sys.argv[2])
        nleads=range(0,215) # calculate for all leads
        startyr=1993
        if smoothclim==1:
            anom_bylead(climyrs,nleads,True,smoothmethod,windowhalfwid)
        else:
            anom_bylead([climstart,climend],nleads)
    elif funx=='anom_bylead_savetr':
        ind=int(sys.argv[2]) # argument should be index, currently in range of 0 to 42
        smoothclim=int(sys.argv[3])
        climyrs=[1993,2023]
        #nleads=215
        for ilead in range(ind*5,(ind+1)*5):
            for jj in range(0,180,60):
                if smoothclim==1:
                    print(climyrs,ilead,jj,True,smoothmethod,windowhalfwid)
                    anom_bylead_savetr(climyrs,ilead,jj,True,smoothmethod,windowhalfwid)
                else:
                    print(climyrs,ilead,jj)
                    anom_bylead_savetr(climyrs,ilead,jj)
    elif funx=='smoothTrend_CanESM5':
        yind=int(sys.argv[2])*60 # 0, 60, or 120
        smoothTrend_CanESM5(yind,climyrs,smoothmethod,windowhalfwid)
    elif funx=='anom_bylead_detr':
        ind=int(sys.argv[2]) # argument should be index, currently in range of 0 to 42
        smoothedClim=True
        smoothedTrend=True
        #nleads=215
        for ilead in range(ind*5,(ind+1)*5):
            for jj in range(0,180,60):
                if smoothedTrend:
                    anom_bylead_detr(climyrs,ilead,jj,True,True,smoothmethod,windowhalfwid)
                elif smoothedClim:
                    anom_bylead_detr(climyrs,ilead,jj,True,False,smoothmethod,windowhalfwid)
                else:
                    anom_bylead_detr(climyrs,ilead,jj)
    elif funx=='calc_quantile_CanESM':
        ind=int(sys.argv[2]) # argument should be index, currently in range of 0 to 42
        opt=int(sys.argv[3]) # 0 for no smoothing, 1 for all smoothing
        det=int(sys.argv[4]) # 0 for no detrend, 1 for detrend
        detr=True if det==1 else False
        print(ind,opt,det,flush=True)
        for delt in (15,):#,30): #0,5,10,15,30
            print(f"detr:{detr}, delt:{delt}",flush=True)
            if opt==0: # no smoothing
                smoothedClim=False
                smoothedTrend=False
                smoothmethod=None
                window=0
            elif opt==1: # all smoothing
                smoothedClim=True
                smoothedTrend=True if detr else False
                smoothmethod=smoothmethod
                window=windowhalfwid
            for ilead in range(ind*5,(ind+1)*5):
                print(f"ilead:{ilead}",flush=True)
                for jj in range(0,180,60):
                    print(f"jj:{jj}",flush=True)
                    calc_quantile_CanESM30(climyrs,ilead,jj,qtile,detr,smoothedClim,smoothedTrend,
                                             smoothmethod,window,delt)
    elif funx=='MHW_calc':
        ind=int(sys.argv[2]) # index, 0 to 42
        opt=int(sys.argv[3]) # number referencing option set
        qtvarname=sys.argv[4] # qt1 or qt2; qt1 is 1 month, qt2 is 3 month (at same lead)
        delt=int(sys.argv[5]) # delt
        det=int(sys.argv[6])
        detr=True if det==1 else False
        if not delt in {0,5,10,15,30}: raise Exception('check delt')
        if opt==0: # no smoothing
            smoothedClim=False
            smoothedTrend=False
            smoothmethod=None
            window=0
        elif opt==1: # all smoothing
            smoothedClim=True
            smoothedTrend=True if detr else False
            smoothmethod=smoothmethod
            window=windowhalfwid
        for ilead in range(ind*5,(ind+1)*5):
            for jj in range(0,180,60):
                print(f'start {ilead},{jj},{qtile}')
                MHW_calc(climyrs,ilead,jj,qtile,detr,smoothedClim,smoothedTrend,
                                         smoothmethod,window,delt,qtvarname)
    #elif funx=='MHW_calc':  #old
    #    ind=int(sys.argv[2]) # argument should be index, currently in range of 0 to 42
    #    qtile=0.9
    #    climyrs=[1993,2023]
    #    for detr in (True,False):
    #        for ilead in range(ind*5,(ind+1)*5):
    #            for jj in range(0,180,60):
    #                print(f'start {detr},{ilead},{jj},{qtile}')
    #                MHW_calc(climyrs,ilead,jj,qtile,detr)
    elif funx=='regrid_daily_OISST':
        # after combining files with MHW_OISST/concatFiles.py
        for yrlims in ylimlistobs:
            regrid_daily_OISST(yrlims)
    elif funx=='calc_OISST_clim':
        calc_OISST_clim(climyrs)
        smooth_OISST_clim(climyrs,smoothmethod,windowhalfwid)
    elif funx=='OISST_anom':
        seg=int(sys.argv[2])
        if seg>=len(ylimlistobs): 
            raise Exception('seg too high')
        else:
            yrlims=ylimlistobs[seg]
        smoothedClim=True
        OISST_anom(yrlims,climyrs,smoothedClim, smoothmethod, windowhalfwid)
    elif funx=='OISST_anom_detr':
        smoothedClim=True
        OISST_anom_detr(climyrs,smoothedClim, smoothmethod, windowhalfwid)
    elif funx=='calc_quantile_OISST':
        smoothedClim=True
        detr=False # True
        for delt in (30,): #(5,10,15,
            print(f"delt={delt}",flush=True)
            for jj in range(0,180,60):
                print(f"jj={jj}",flush=True)
                calc_quantile_OISST(climyrs,jj,qtile,detr=detr,smoothClim=smoothedClim,meth=smoothmethod,win=windowhalfwid,delt=delt)
    elif funx=='MHW_calc_OISST':
        smoothedClim=True
        qtvar='qt1'
        for delt in (15,30):
            for detr in (False,): #False):
                for jj in range(0,180,60):
                    MHW_calc_OISST(climyrs,jj,qtile,detr,smoothClim=smoothedClim,meth=smoothmethod,win=windowhalfwid,
                                    delt=delt,qtvar=qtvar)
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
                calc_quantile_OISST(climyrs,jj,qtile,detr=detr,delt=delt)
                MHW_calc_OISST(climyrs,jj,qtile,detr=detr,delt=delt)
    elif funx=='saveSEDI':
        ind=int(sys.argv[2]) # argument should be index, currently in range of 0 to 42
        smooth=int(sys.argv[3]) # 0 or 1
        delt=int(sys.argv[4]) # delt
        det=int(sys.argv[5])    # 0 or 1
        detr=True if det==1 else False
        if smooth==1:
            smoothedClim=True #False #True
            win=windowhalfwid #0
        else:
            smoothedClim=False #True
            win=0
        qtvar='qt1'
        smoothTrend=True if (smoothedClim and detr) else False
        for ilead in range(ind*5,(ind+1)*5):
            for jj in range(0,180,60):
                print(f'start {detr},{ilead},{jj},{qtile}',flush=True)
                pathobs=fnameOISSTMHW(climyrs,jj,qtile,smoothedClim,smoothmethod,win,detr,delt,qtvar)
                pathfor=fnameCanESMMHW(workdir,climyrs[0],climyrs[-1],ilead,jj,qtile,detr,smoothedClim,smoothTrend,smoothmethod,win,delt,qtvar)
                fout=fnameSEDI_OISST_CanESM_daily(ilead,climyrs, smoothedClim, smoothmethod, windowhalfwid, detr, qtile, delt, qtvar, jj)
                if os.path.exists(fout):
                    pass
                else:
                    iSEDI=compstats(pathfor,pathobs,ilead)
                    iSEDI.calcSEDI()
                    iSEDI.saveSEDI(fout)
                    iSEDI.closefiles()
                    print(fout,flush=True)
    elif funx=='saveReli':
        ind=int(sys.argv[2]) # index from job array, should be adjusted to range of leadlist
        leadlist=[50,75,100,125,200]#[0, 1, 3, 6, 10, 15, 20, 30]
        climyrs=[1993,2023]
        detr=True
        smoothClim=True
        smoothTrend=True
        meth=method
        win=halfwin
        delt=15 
        qtvar='qt1'
        region='global'
        mcount, ocount, ps = reliability1(climyrs,leadlist[ind],qtile,detr,smoothClim,smoothTrend,
                                        meth,win,delt,qtvar,region)
        np.savez(fnameReli(leadlist[ind],climyrs, smoothClim, meth, win, detr, qtile, delt,qtvar,region),
                 mcount=mcount,ocount=ocount,ps=ps)
    print('Done')


