import os, sys
import datetime as dt
import xarray as xr
import numpy as np
from dask.distributed import Client, LocalCluster
import dask.array as da

# run with two arguments: first year to process and first year not to process
# should add up to 1993, 2024
workdir='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/MHW_daily'
mdirC5='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/data/predictions/cansipsv3_daily/CanESM5'
osrcdir='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/data/obs/NOAA_OISST/combined'

ylimlistobs=[[1991,2000],[2001,2010],[2011,2020],[2021,2024]]
method='tri'
halfwin=10
qtile=.9

fnameCanESMjoined=lambda mdir, yyyy, mm, dd, hh: \
       f"{mdir}/joined/cwao_CanESM5.1p1bc-v20240611_hindcast_S{yyyy:04}{mm:02}{dd:02}{hh:02}_ocean_6hr_surface_tso.nc"
fnameCanESMdaily=lambda mdir, yyyy, mm, dd, hh: \
       f"{mdir}/joined/cwao_CanESM5.1p1bc-v20240611_hindcast_S{yyyy:04}{mm:02}{dd:02}{hh:02}_ocean_1d_surface_tso.nc"
fnameCanESMClim=lambda mdir, climyfirst, climylast, mm: \
       f"{mdir}/clim/clim_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"Mon{mm:02}_ocean_1d_surface_tso.nc"
fnameCanESMClimSmooth=lambda mdir, climyfirst, climylast, mm, method, window: \
       f"{mdir}/clim/clim_smooth_{method}{window}cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"Mon{mm:02}_ocean_1d_surface_tso.nc"
#fnameCanESMAnom=lambda mdir, climyfirst,climylast,lfirst, llast, mm: \
#       f"{mdir}/anom/anom_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_SMon{mm:02}_"\
#       f"L_{lfirst:03}_{llast:03}_ocean_1d_surface_tso.nc"
#fnameCanESMAnomSClim=lambda mdir, climyfirst,climylast,lfirst,llast,mm,meth,win:\
#       f"{mdir}/anom/anom_sclim{meth}{win}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_SMon{mm:02}_"\
#       f"L_{lfirst:03}_{llast:03}_ocean_1d_surface_tso.nc"
fnameCanESMAnom=lambda mdir, climyfirst, climylast, yyyy, mm: \
       f"{mdir}/anom/anom_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"SYr{yyyy:04}Mon{mm:02}_ocean_1d_surface_tso.nc"
fnameCanESMAnomSClim=lambda mdir, climyfirst, climylast, yyyy, mm, meth, win: \
       f"{mdir}/anom/anom_sclim{meth}{win}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"SYr{yyyy:04}Mon{mm:02}_ocean_1d_surface_tso.nc"
fnameCanESMAnomByLead=lambda mdir, climyfirst, climylast, ilead, istartlat: \
       f"{mdir}/byLead/anomByLead_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
fnameCanESMAnomByLeadSClim=lambda mdir, climyfirst, climylast, ilead, istartlat,meth,win: \
       f"{mdir}/byLead/anomByLead_sclim{meth}{win}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
fnameCanESMAnomDetrByLeadIndiv=lambda mdir, climyfirst, climylast, ilead, istartlat: \
       f"{mdir}/byLeadDetrIndiv2/anomDetrByLead_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
#fnameCanESMAnomDetrByLead=lambda mdir, climyfirst, climylast, ilead, istartlat: \
#       f"{mdir}/byLeadDetr/anomDetrByLead_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
#       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
def fnameCanESMDetrFitByLead(mdir, climyfirst, climylast, ilead, istartlat, sourcedesig=''):
    subdir='byLeadDetrIndiv2' if sourcedesig=='' else 'byLeadDetr'
    return f"{mdir}/{subdir}/fitDetrByLead{sourcedesig}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
fnameCanESMDetrFitByLeadS=lambda mdir, climyfirst, climylast, ilead, istartlat, meth, win, sourcedesig='': \
       f"{mdir}/byLeadDetr/fitDetrByLead{sourcedesig}_smoothed{meth}{win}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
def fnameCanESMAnomDetrByLead(mdir, climyfirst, climylast, ilead, istartlat, smoothClim=False,smoothTrend=False,meth=None,win=1): 
    subdir='byLeadDetr' if (smoothClim or smoothTrend) else 'byLeadDetrIndiv2'
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    strSTrend=f'_TrS{meth}{win}' if smoothClim else ''
    return f"{mdir}/{subdir}/anomDetrByLead{strSClim}{strSTrend}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
def fnameCanESMAnomDetrQtile(mdir, climyfirst, climylast, ilead, istartlat, qt, smoothClim=False,smoothTrend=False,meth=None,win=1,delt=0): 
    subdir='byLeadDetr' if (smoothClim or smoothTrend) else 'byLeadDetrIndiv2'
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    strSTrend=f'_TrS{meth}{win}' if smoothClim else ''
    strdelt=f'_delt{delt}' # reflects number of lead time days to pool together
    qstr='{:.2f}'.format(qt).replace('.','_')
    return f"{mdir}/{subdir}/qtileDetrByLead{strSClim}{strSTrend}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
            f"L{ilead:03}{strdelt}_j{istartlat:03}_q{qstr}_ocean_1d_surface_tso.nc"
#fnameCanESMAnomQtile=lambda mdir, climyfirst, climylast, ilead, istartlat, qt: \
#       f"{mdir}/byLead/qtileByLead_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
#       f"L{ilead:03}_j{istartlat:03}_q{'{:.2f}'.format(qt).replace('.','_')}_ocean_1d_surface_tso.nc"
def fnameCanESMMHWDetr(mdir, climyfirst, climylast, ilead, istartlat, qt, smoothClim=False,smoothTrend=False,meth=None,win=1,delt=0,qtvar='qt1'): 
    subdir='byLeadDetr' if (smoothClim or smoothTrend) else 'byLeadDetrIndiv2'
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    strSTrend=f'_TrS{meth}{win}' if smoothClim else ''
    strdelt=f'_delt{delt}' # reflects number of lead time days to pool together
    qstr='{:.2f}'.format(qt).replace('.','_')
    qvstr='_'+qtvar
    return f"{mdir}/{subdir}/MHWDetrByLead{strSClim}{strSTrend}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
            f"L{ilead:03}{strdelt}_j{istartlat:03}{qvstr}_q{qstr}_ocean_1d_surface_tso.nc"
#fnameCanESMMHW=lambda mdir, climyfirst, climylast, ilead, istartlat, qt: \
#       f"{mdir}/byLeadMHW/MHWByLead_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
#       f"L{ilead:03}_j{istartlat:03}_q{'{:.2f}'.format(qt).replace('.','_')}_ocean_1d_surface_tso.nc"
fnameOISSTDaily = lambda iy, im:\
       f"{osrcdir}/oisst-avhrr-v02r01.{iy}{im:02}_daily.nc"
fnameOISSTDailyGrid2 = lambda yrlims: \
       f"{workdir}/OISST/oisst-avhrr-v02r01.regridded1x1g2.daily.{yrlims[0]}_{yrlims[-1]}.nc"
fnameOISSTDailyClim=lambda climyfirst, climylast: \
       f"{workdir}/OISST/climSST_oisst-avhrr-v02r01.regridded1x1g2.daily_C{climyfirst:04}_{climylast:04}.nc"

def mkdirs(fsave):
    saveloc=os.path.dirname(fsave)
    if not os.path.exists(saveloc):
        try:
            os.makedirs(saveloc)
        except FileExistsError:
            pass # in case other code running at the same time got to it first
    return

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
    #fout=fnameCanESMDetrFitByLead(workdir, climyrs[0], climyrs[-1], ilead, istartlat)
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

def trismooth(t,vals,L=30):
    # t is values assoc with 1st dim
    # smooths over 1st dim
    # if vector, add dim:
    delt=t[1]-t[0]
    alpha=1
    if len(np.shape(vals))==1:
        vals=np.expand_dims(vals,axis=1)
    fil=np.empty(np.shape(vals))
    for ind, ti in enumerate(t):
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
                time=[dt.datetime(yy,mm,1,0,0)+dt.timedelta(hours=float(ff.leadtime.values)) for yy in range(1993,2025) for mm in range(1,13) if yy<2024 or mm<=6]
                fout=xr.Dataset(data_vars={'sst_an':(['reftime','r','lat','lon'],sst_an2.data),
                                           'time':(['reftime',],time,{'long_name':'Real Time'})},
                                coords={'reftime':reftime,
                                        'r':np.arange(0,ff.sst_an.shape[1]),
                                        'lat':ff.lat,
                                        'lon':ff.lon})
                if smoothClim:
                    fnamout=fnameCanESMAnomByLeadSClim(workdir,climyrs[0],climyrs[-1],ilead,jj,smoothmethod,window)
                else:
                    fnamout=fnameCanESMAnomByLead(workdir,climyrs[0],climyrs[-1],ilead,jj)
                mkdirs(fnamout)
                print(fnamout)
                fout.to_netcdf(fnamout,mode='w') # encoding={'sst_an': {'chunksizes': [Anom0.shape[0],1,20,360]}}
                del sst_an2; del fout;
                ff.close(); del ff;
    return

def anom_bylead_savetr(climyrs,ilead,jj,smoothClim=False,smoothmethod=None,window=1):
    if smoothClim:
        fin=fnameCanESMAnomByLeadSClim(workdir, climyrs[0], climyrs[-1], ilead, jj,smoothmethod,window)
        desstr=f'_ClimS{smoothmethod}{window}'
        fout=fnameCanESMDetrFitByLead(workdir, climyrs[0], climyrs[-1], ilead, jj,desstr)
    else:
        fin=fnameCanESMAnomByLead(workdir, climyrs[0], climyrs[-1], ilead, jj)
        fout=fnameCanESMDetrFitByLead(workdir, climyrs[0], climyrs[-1], ilead, jj)
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
    flistbS=[fnameCanESMDetrFitByLead(workdir, climyrs[0],climyrs[-1], ilead, yind,desstr) for ilead in range(0,215)]
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
    if smoothedClim:
        desstr=f'_ClimS{smoothmethod}{window}'
        fin=fnameCanESMAnomByLeadSClim(workdir, climyrs[0], climyrs[-1], ilead, jj,smoothmethod,window)
    if smoothedTrend:
        fb=fnameCanESMDetrFitByLeadS(workdir, climyrs[0],climyrs[-1], ilead, jj, smoothmethod, window, desstr)
        fout=fnameCanESMAnomDetrByLead(workdir, climyrs[0], climyrs[-1], ilead, jj,smoothClim=True,smoothTrend=True,meth=smoothmethod,win=window)
    elif smoothedClim: # (and not smoothedTrend)
        fb=fnameCanESMDetrFitByLead(workdir, climyrs[0],climyrs[-1], ilead, jj, desstr)
        fout=fnameCanESMAnomDetrByLead(workdir, climyrs[0], climyrs[-1], ilead, jj,smoothClim=True,smoothTrend=False,meth=smoothmethod,win=window)
    else: # no smoothing
        fin=fnameCanESMAnomByLead(workdir, climyrs[0], climyrs[-1], ilead, jj)
        fb=fnameCanESMDetrFitByLead(workdir, climyrs[0], climyrs[-1], ilead, jj)
        fout=fnameCanESMAnomDetrByLead(workdir, climyrs[0], climyrs[-1], ilead, jj)
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

def calc_quantile_detr_A(climyrs,ilead,jj,qtile,detr=True,smoothedClim=False,smoothedTrend=False,smoothmethod=None,window=1,delt=0):
    # version 1: 10 day windows in lead time
    lmax=215
    def getind(i0):
        if i0>=1 and i0<=10:
            return [i0-1,i0,i0+1]
        elif i0==0:
            return [11,0,1]
        elif i0==11:
            return [10,11,0]
    def leadbounds(l0,lmax,delt):
        i0=min(max(l0-delt,0),lmax-(2*delt+1))
        return i0, i0+2*delt+1
    flist=[fnameCanESMAnomDetrByLead(workdir, climyrs[0], climyrs[-1], il, jj,smoothClim=smoothedClim,smoothTrend=smoothedTrend,meth=smoothmethod,win=window) \
            for il in range(*leadbounds(ilead,215,delt))]
    print(flist)
    ff=xr.open_mfdataset(flist,combine='nested',concat_dim=['leadtime'],parallel=True,decode_times=False)
    fc=ff.sst_an.coarsen(reftime=12,boundary='pad').construct(reftime=('year','month'))
    sh=fc.shape
    ql1=np.nan*np.ones((12,sh[-2],sh[-1]))
    ql2=np.nan*np.ones((12,sh[-2],sh[-1]))
    for ii in range(0,12):
        pool1=fc.isel(month=ii).values.reshape((sh[0]*sh[1]*sh[3],sh[4],sh[5]))
        ql1[ii,...]=np.nanquantile(pool1,0.9,axis=0)
        pool2=fc.sel(month=getind(ii)).values.reshape((sh[0]*sh[1]*3*sh[3],sh[4],sh[5]))
        ql2[ii,...]=np.nanquantile(pool2,0.9,axis=0)
    fqout=fnameCanESMAnomDetrQtile(workdir, climyrs[0], climyrs[-1], ilead, jj, qtile, 
                                   smoothClim=smoothedClim,smoothTrend=smoothedTrend,meth=smoothmethod,win=window,delt=delt)
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

def MHW_calc(climyrs,ilead,jj,qtile,detr=True,smoothedClim=False,smoothedTrend=False,smoothmethod=None,window=1,delt=0,qtvar='qt1'):
    if detr: # set path-defining fxns for detrended or non-detrended versions of calculation
        fanom=fnameCanESMAnomDetrByLead(workdir, climyrs[0], climyrs[-1], ilead, jj,smoothClim=smoothedClim,smoothTrend=smoothedTrend,meth=smoothmethod,win=window) 
        fqtile=fnameCanESMAnomDetrQtile(workdir, climyrs[0], climyrs[-1], ilead, jj, qtile,smoothClim=smoothedClim,smoothTrend=smoothedTrend,meth=smoothmethod,win=window,delt=delt)
        fMHW=fnameCanESMMHWDetr(workdir, climyrs[0], climyrs[-1], ilead, jj,qtile,smoothClim=smoothedClim,smoothTrend=smoothedTrend,meth=smoothmethod,win=window,delt=delt,qtvar=qtvar)
    else:
        raise Exception('not implemented yet')
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

def calc_OISST_clim(yrlims):
    flist=[fnameOISSTDailyGrid2(yrlims) for yrlims in ylimlistobs]
    fg2=xr.open_mfdataset(flist,decode_times=False,parallel=True)
    sst=fg2.sst.data.rechunk((len(fg2.time.values),90,90))
    tdt=np.array([dt.datetime(1978,1,1,12)+dt.timedelta(days=float(el)) for el in fg2.time.values])
    yd=np.array([(dt.datetime(el.year,el.month,el.day)-dt.datetime(el.year-1,12,31)).days for el in tdt])
    climsst=np.zeros((365,180,360))
    for iyd in range(1,366):
        ind=yd==iyd
        if iyd==365: ind=np.logical_or(ind,yd==366)
        indyrs=np.array([(el.year>=climyrs[0])&(el.year<=climyrs[-1]) for el in tdt])
        ind=np.logical_and(ind,indyrs)
        climsst[iyd-1,:,:]=sst.mean(axis=0)
        if iyd%10==0: print(iyd)
    ds=xr.Dataset(data_vars={'sst':(['yearday','lat','lon'],climsst)},
                             coords={'yearday':np.arange(1,366),
                                     'lat':fg2.lat,
                                     'lon':fg2.lon})
    fout=fnameOISSTDailyClim(climyrs[0],climyrs[-1])
    ds.to_netcdf(fout,mode='w') 
    return
#def process_daily_OISST():
#    
#    return

if __name__=="__main__":
    # argument options:
    # - python MHW_daily_calcs.py fconvert_CanESM startyear endyear
    # - python MHW_daily_calcs.py calcAnom_CanESM5 climfirstyear climlastyear
    funx=sys.argv[1] # what function to execute
    ncpu=len(os.sched_getaffinity(0))
    climyrs=[1993,2023]
    windowhalfwid=10
    smoothmethod='tri'
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
    elif funx=='calc_quantile_detr_A':
        ind=int(sys.argv[2]) # argument should be index, currently in range of 0 to 42
        opt=int(sys.argv[3]) # numer referencing option set
        if opt==0: # no smoothing
            smoothedClim=False
            smoothedTrend=False
            smoothmethod=None
            window=0
            delt=0
        elif opt==1: # all smoothing
            smoothedClim=True
            smoothedTrend=True
            smoothmethod=smoothmethod
            window=windowhalfwid
            delt=15
        for ilead in range(ind*5,(ind+1)*5):
            for jj in range(0,180,60):
                calc_quantile_detr_A(climyrs,ilead,jj,qtile,True,smoothedClim,smoothedTrend,
                                         smoothmethod,window,delt)
    elif funx=='MHW_calc':
        ind=int(sys.argv[2]) # index, 0 to 42
        opt=int(sys.argv[3]) # numer referencing option set
        qtvarname=sys.argv[4] # qt1 or qt2; qt1 is 1 month, qt2 is 3 month (at same lead)
        if opt==0: # no smoothing
            smoothedClim=False
            smoothedTrend=False
            smoothmethod=None
            window=0
            delt=0
        elif opt==1: # all smoothing
            smoothedClim=True
            smoothedTrend=True
            smoothmethod=smoothmethod
            window=windowhalfwid
            delt=5
        for ilead in range(ind*5,(ind+1)*5):
            for jj in range(0,180,60):
                print(f'start {ilead},{jj},{qtile}')
                MHW_calc(climyrs,ilead,jj,qtile,True,smoothedClim,smoothedTrend,
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
    print('Done')
