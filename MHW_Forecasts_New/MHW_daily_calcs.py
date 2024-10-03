import os, sys
import datetime as dt
import xarray as xr
import numpy as np
from dask.distributed import Client, LocalCluster

# run with two arguments: first year to process and first year not to process
# should add up to 1993, 2024
workdir='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/MHW_daily/'
mdirC5='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/data/predictions/cansipsv3_daily/CanESM5'
fnameCanESMjoined=lambda mdir, yyyy, mm, dd, hh: \
       f"{mdir}/joined/cwao_CanESM5.1p1bc-v20240611_hindcast_S{yyyy:04}{mm:02}{dd:02}{hh:02}_ocean_6hr_surface_tso.nc"
fnameCanESMdaily=lambda mdir, yyyy, mm, dd, hh: \
       f"{mdir}/joined/cwao_CanESM5.1p1bc-v20240611_hindcast_S{yyyy:04}{mm:02}{dd:02}{hh:02}_ocean_1d_surface_tso.nc"
fnameCanESMAnom=lambda mdir, climyfirst,climylast,lfirst, llast, mm: \
       f"{mdir}/anom/anom_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_SMon{mm:02}_"\
       f"L_{lfirst:03}_{llast:03}_ocean_1d_surface_tso.nc"
fnameCanESMClim=lambda mdir, climyfirst, climylast, mm: \
       f"{mdir}/clim/clim_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"Mon{mm:02}_ocean_1d_surface_tso.nc"
fnameCanESMAnom=lambda mdir, climyfirst, climylast, yyyy, mm: \
       f"{mdir}/anom/anom_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"SYr{yyyy:04}Mon{mm:02}_ocean_1d_surface_tso.nc"
fnameCanESMAnomByLead=lambda mdir, climyfirst, climylast, ilead, istartlat: \
       f"{mdir}/byLead/anomByLead_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
fnameCanESMAnomDetrByLeadIndiv=lambda mdir, climyfirst, climylast, ilead, istartlat: \
       f"{mdir}/byLeadDetrIndiv/anomDetrByLead_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
fnameCanESMAnomDetrByLead=lambda mdir, climyfirst, climylast, ilead, istartlat: \
       f"{mdir}/byLeadDetr/anomDetrByLead_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
fnameCanESMDetrFitByLead=lambda mdir, climyfirst, climylast, ilead, istartlat: \
       f"{mdir}/byLeadDetr/fitDetrByLead_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
fnameCanESMAnomDetrByLead=lambda mdir, climyfirst, climylast, ilead, istartlat: \
       f"{mdir}/byLeadDetr/anomDetrByLead_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
fnameCanESMAnomDetrQtile=lambda mdir, climyfirst, climylast, ilead, istartlat, qt: \
       f"{mdir}/byLeadDetr/qtileDetrByLead_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_q{'{:.2f}'.format(qt).replace('.','_')}_ocean_1d_surface_tso.nc"
fnameCanESMAnomQtile=lambda mdir, climyfirst, climylast, ilead, istartlat, qt: \
       f"{mdir}/byLead/qtileByLead_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_q{'{:.2f}'.format(qt).replace('.','_')}_ocean_1d_surface_tso.nc"
fnameCanESMMHWDetr=lambda mdir, climyfirst, climylast, ilead, istartlat, qt: \
       f"{mdir}/byLeadMHWDetr/MHWDetrByLead_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_q{'{:.2f}'.format(qt).replace('.','_')}_ocean_1d_surface_tso.nc"
fnameCanESMMHW=lambda mdir, climyfirst, climylast, ilead, istartlat, qt: \
       f"{mdir}/byLeadMHW/MHWByLead_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_q{'{:.2f}'.format(qt).replace('.','_')}_ocean_1d_surface_tso.nc"
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

def lsqfit_md_detrPooled_saveb(data,climyrs,ilead,istartlat): # remove single trend for all ensemble members (per model)
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
    b=np.reshape(b,tuple([2]+list(dshape)[2:]))
    fout=fnameCanESMDetrFitByLead(workdir, climyrs[0], climyrs[-1], ilead, istartlat)
    dsb=xr.dataset(data_vars={'fit':(['b','lat','lon'],b),},
                   coords={'b':np.arange(0,2),
                           'lat':ff.lat,
                           'lon':ff.lon})
    dsb.to_netcdf(fout,mode='w')
    ydetr=np.reshape(ydetr,dshape)
    return ydetr

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

def calcAnom_CanESM5(climyrs):#,nlead):
    for mm in range(1,13): # month loop
        print(f"Month:{mm} {dt.datetime.now()}",flush=True)
        #for ix in range(0,int(nlead/5)):
        fnamelast=fnameCanESMAnom(workdir,climyrs[0],climyrs[-1],climyrs[-1],mm)
        if not os.path.exists(fnamelast): # skip if file at (almost) end already exists
            flist=[fnameCanESMdaily(mdirC5,yyyy,mm,1,0) for yyyy in range(1993,2025) if yyyy<2024 or mm<=6] # stop at Jul 2024
            fnameclim=fnameCanESMClim(workdir,climyrs[0],climyrs[-1],mm)
            with LocalCluster(n_workers=ncpu-1,threads_per_worker=1) as cluster, Client(cluster) as client:
                with xr.open_mfdataset(flist,parallel=True,combine='nested',concat_dim='reftime',
                               chunks={'reftime':-1,'leadtime':-1,'r':-1,'lat':-1,'lon':-1}) as ff:
                    if not os.path.exists(fnameclim):
                        EClim=ff.tso.sel(reftime=slice(np.datetime64(f'{climyrs[0]:04}-01-01'),
                            np.datetime64(f'{climyrs[-1]:04}-12-31'))).mean(dim='r').mean(dim='reftime')
                        mkdirs(fnameclim)
                        EClim.to_netcdf(fnameclim,mode='w')
                        del EClim
                    fclim=xr.open_dataset(fnameclim)
                    EClim=fclim['tso']
                    for iy in range(1993,2025):
                        if iy<2024 or mm<=6: # stop at Jul 2024
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

def anom_bylead(climyrs,nleads):
    for ilead in nleads:
        for jj in range(0,180,60):
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
                fnamout=fnameCanESMAnomByLead(workdir,climyrs[0],climyrs[-1],ilead,jj)
                mkdirs(fnamout)
                fout.to_netcdf(fnamout,mode='w') # encoding={'sst_an': {'chunksizes': [Anom0.shape[0],1,20,360]}}
                del sst_an2; del fout;
                ff.close(); del ff;
    return

def anom_bylead_detr(climyrs,ilead,jj):
    fin=fnameCanESMAnomByLead(workdir, climyrs[0], climyrs[-1], ilead, jj)
    fout=fnameCanESMAnomDetrByLead(workdir, climyrs[0], climyrs[-1], ilead, jj)
    mkdirs(fout)
    ff=xr.open_dataset(fin,decode_times=False)
    out=lsqfit_md_detrPooled(ff.sst_an)
    #daout=ff.sst_an.copy(deep=True,data=out)
    #daout.to_netcdf(fout,mode='w')
    ff.close()
    return

def MHW_calc(climyrs,ilead,jj,qtile,detr=True):
    if detr: # set path-defining fxns for detrended or non-detrended versions of calculation
        ffunanom=fnameCanESMAnomDetrByLead # or fnameCanESMAnomDetrByLeadIndiv if switch method
        ffunqtile=fnameCanESMAnomDetrQtile
        ffunMHW=fnameCanESMMHWDetr
    else:
        ffunanom=fnameCanESMAnomByLead
        ffunqtile=fnameCanESMAnomQtile
        ffunMHW=fnameCanESMMHW
    ff=xr.open_dataset(ffunanom(workdir, climyrs[0], climyrs[-1], ilead, jj),decode_times=False)
    fc=ff.sst_an.coarsen(reftime=12,boundary='pad').construct(reftime=('year','month'))
    sh=fc.shape
    def getind(i0):
        if i0>=1 and i0<=10:
            return [i0-1,i0,i0+1]
        elif i0==0:
            return [11,0,1]
        elif i0==11:
            return [10,11,0]
    ql=np.nan*np.ones((12,sh[-2],sh[-1]))
    for ii in range(0,12):
        pool=fc.sel(month=getind(ii)).values.reshape((sh[0]*3*sh[2],sh[3],sh[4]))
        ql[ii,...]=np.nanquantile(pool,0.9,axis=0)
    fqout=ffunqtile(workdir, climyrs[0], climyrs[-1], ilead, jj,qtile)
    dsqt=xr.Dataset(data_vars={'qt':(['month','lat','lon'],ql,{'long_name':f"{100*qtile}th percentile value"}),},
                   coords={'month':np.arange(0,12),
                           'lat':ff.lat,
                           'lon':ff.lon})
    dsqt.to_netcdf(fqout,mode='w')
    del dsqt
    qt2=np.expand_dims(ql,[0,2])
    del ql
    MHW=np.ma.masked_where(np.isnan(fc),np.where(fc>qt2,1,0))
    del fc; del qt2;
    MHWstack=np.reshape(MHW,(sh[0]*sh[1],sh[2],sh[3],sh[4]))
    del MHW
    MHWstack=MHWstack[:ff.sst_an.shape[0],...] # remove appended nans
    dsMHW=xr.Dataset(data_vars={'isMHW':(['reftime','r','lat','lon'],MHWstack),
                                'MHW_prob':(['reftime','lat','lon'],np.mean(MHWstack,axis=1))},
                    coords={'reftime':ff.reftime,'r':ff.r,'lat':ff.lat,'lon':ff.lon,'leadtime':ff.leadtime})
    del MHWstack
    fMHWout=ffunMHW(workdir, climyrs[0], climyrs[-1], ilead, jj,qtile)
    mkdirs(fMHWout)
    dsMHW.to_netcdf(fMHWout,mode='w')
    del dsMHW;
    ff.close(); del ff
    return

if __name__=="__main__":
    # argument options:
    # - python MHW_daily_calcs.py fconvert_CanESM startyear endyear
    # - python MHW_daily_calcs.py calcAnom_CanESM5 climfirstyear climlastyear
    funx=sys.argv[1] # what function to execute
    ncpu=len(os.sched_getaffinity(0))
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
    elif funx=='calcAnom_CanESM5':
        climstart=int(sys.argv[2])
        climend=int(sys.argv[3])
        #nlead=215
        calcAnom_CanESM5([climstart,climend])#,nlead)
    elif funx=='anom_bylead':
        climstart=1993
        climend=2023
        nleads=range(0,215) # calculate for all leads
        startyr=1993
        anom_bylead([climstart,climend],nleads)
    elif funx=='anom_bylead_detr':
        ind=int(sys.argv[2]) # argument should be index, currently in range of 0 to 42
        #nleads=215
        for ilead in range(ind*5,(ind+1)*5):
            for jj in range(0,180,60):
                anom_bylead_detr([1993,2023],ilead,jj)
    elif funx=='MHW_calc':
        ind=int(sys.argv[2]) # argument should be index, currently in range of 0 to 42
        qtile=0.9
        climyrs=[1993,2023]
        for detr in (True,False):
            for ilead in range(ind*5,(ind+1)*5):
                for jj in range(0,180,60):
                    print(f'start {detr},{ilead},{jj},{qtile}')
                    MHW_calc(climyrs,ilead,jj,qtile,detr)
    print('Done')
