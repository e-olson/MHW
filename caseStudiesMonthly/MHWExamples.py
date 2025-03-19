import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from dask.distributed import Client, LocalCluster
import dask.array as da
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import warnings
import os
import sys
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import pandas as pd

def make_movie_Pac():
    nl = dict(zip( ['CanCM4i', 'COLA-RSMAS-CCSM4', 'GEM-NEMO', 'GFDL-SPEAR', 'NASA-GEOSS2S', 
                            'NCEP-CFSv2', 'CanESM5', 'GEM5.2-NEMO', 'GEM5-NEMO', 'CanCM4i-IC3',],
                   [       11,                 11,         11,           11,              8,
                                       9,        11,            11,          11,            11,])) 
    modict={'Jacox':['CanCM4i', 'COLA-RSMAS-CCSM4', 'GEM-NEMO', 'GFDL-SPEAR', 
                            'NASA-GEOSS2S', 'NCEP-CFSv2'],
            'CanSIPSv2':['CanCM4i','GEM-NEMO'],
            'CanSIPSv21':['CanCM4i-IC3','GEM5-NEMO'],
            'CanSIPSv3':['CanESM5','GEM5.2-NEMO'],
            'GFDLNASA':[ 'GFDL-SPEAR','NASA-GEOSS2S'],}
    
    basepath='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/MHW'
    clim_years = [1991, 2020]
    years=clim_years
    mhwdir = basepath+'/mhw/detrended';
    #f_obs = basepath+f'/OISST/mhw_detrended_oisst-avhrr-v02r01.regridded1x1.monthly.{years[0]}_{years[-1]}.nc'
    f_obs = basepath+f'/OISST/mhw_detrended_oisst-avhrr-v02r01.regridded1x1.monthly.{clim_years[0]}_{clim_years[-1]}.nc'
    fobs=xr.open_dataset(f_obs)
    mods=modict['CanSIPSv3']
    il=3 # 3 months lead time
    flist=[]
    for modi in mods:
        if il<nl[modi]:
            # Load MHWs
            f_in = f'{mhwdir}/mhw_{modi}_l{il}_detrended_{clim_years[0]}_{clim_years[1]}.nc'
            flist.append(f_in)
    ffor=xr.open_mfdataset(flist,chunks={'time':1,'X':-1,'Y':-1,'M':-1},concat_dim='M',
                        combine='nested',data_vars='minimal',coords='minimal',parallel=True,
                        preprocess=lambda f: f.drop_vars(["sst_an_thr","mhw_prob"]))
    M=40
    mhwfor=ffor.is_mhw.data[:(-1*il),...].sum(axis=1)/M if il>0 else ffor.is_mhw.data.sum(axis=1)/M
    mhwobs=fobs.is_mhw.data[il:,...]
    lm=np.sum(fobs.is_mhw.data,axis=0)==0
    # ice mask
    f_ice=basepath+f'/OISST/oisst-avhrr-v02r01.regridded1x1.monthly.{years[0]}_{years[-1]}.nc'
    fice=xr.open_dataset(f_ice)
    icemask1=np.nanmax(fice.ice.data,axis=0)>0
    Athresh=.50
    Tthresh=.9
    lost=np.where(fice.ice>Athresh,1,0).sum(axis=0)
    imask=lost>(1-Tthresh)*np.shape(fice.ice)[0]
    cmap=plt.get_cmap('Reds')
    cmap.set_bad('w',alpha=0)
    
    # Define the meta data for the movie
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='NPac', artist='Matplotlib',
                    comment='MHW')
    writer = FFMpegWriter(fps=1, metadata=metadata)
    
    cmap=plt.get_cmap('Reds')
    cmap.set_bad('w',alpha=0)
    proj=ccrs.Orthographic(central_longitude=-136, central_latitude=52, globe=None)

    fig = plt.figure(figsize=[8, 4])
    ax1 = fig.add_subplot(1, 2, 1, projection=proj)
    ax2 = fig.add_subplot(1, 2, 2, projection=proj)
    axcb = fig.add_axes(rect=[.97,.15,.02,.7])
    fig.subplots_adjust(bottom=0.05, top=0.95,left=0.04, right=0.9, wspace=0.05)
    ax1.set_title('Observations')
    ax2.set_title(f'Hindcast (Lead {il})')
    for iax in (ax1,ax2):
        iax.set_extent([-150, -123, 47, 61], ccrs.PlateCarree())
    
    # Update the frames for the movie
    with writer.saving(fig, "mhw_NPac.mp4", dpi=100):
        for i in range(mhwobs.shape[0]):
            print(f"{i/mhwobs.shape[0]}: {dt.datetime.now()}",flush=True)
            date=dt.datetime(years[0]+int((i+il)/12),(i+il)%12+1,15)
            fig.suptitle(f"{date.year}-{date.month:02}")
            ax1.pcolormesh(fobs.lon.values,fobs.lat.values,np.ma.masked_where(imask|lm,mhwobs[i,...]),
                                        vmin=0,vmax=1,cmap=cmap,transform=ccrs.PlateCarree())
            m=ax2.pcolormesh(fobs.lon.values,fobs.lat.values,np.ma.masked_where(imask|lm,mhwfor[i,...]),
                                        vmin=0,vmax=1,cmap=cmap,transform=ccrs.PlateCarree())
            fig.colorbar(m,cax=axcb)
            
            for iax in (ax1,ax2):
                iax.add_feature(cfeature.LAND,zorder=1,color='lightgray')
                gl = iax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                                  linewidth=1, color='gray', alpha=0.5, linestyle=':')
                #gl.xlocator = mticker.FixedLocator([-90, -70, -50,-30,-10,10])
                #gl.ylocator = mticker.FixedLocator([20, 30, 40, 50, 60,70])
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                if iax==ax1:
                    gl.top_labels = False
                    gl.right_labels = False
                elif iax==ax2:
                    gl.top_labels = False
                    gl.left_labels = False
                writer.grab_frame()
    fobs.close()
    ffor.close()
    fice.close()
    return

def byleadcalcsNAtl():
    nl = dict(zip( ['CanCM4i', 'COLA-RSMAS-CCSM4', 'GEM-NEMO', 'GFDL-SPEAR', 'NASA-GEOSS2S', 
                                'NCEP-CFSv2', 'CanESM5', 'GEM5.2-NEMO', 'GEM5-NEMO', 'CanCM4i-IC3',],
                   [       11,                 11,         11,           11,              8,
                                           9,        11,            11,          11,            11,])) 
    modict={'Jacox':['CanCM4i', 'COLA-RSMAS-CCSM4', 'GEM-NEMO', 'GFDL-SPEAR', 
                            'NASA-GEOSS2S', 'NCEP-CFSv2'],
            'CanSIPSv2':['CanCM4i','GEM-NEMO'],
            'CanSIPSv21':['CanCM4i-IC3','GEM5-NEMO'],
            'CanSIPSv3':['CanESM5','GEM5.2-NEMO'],
            'GFDLNASA':[ 'GFDL-SPEAR','NASA-GEOSS2S'],}
    basepath='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/MHW'
    clim_years = [1991, 2020]
    years=clim_years
    mhwdir = basepath+'/mhw/detrended';
    #f_obs = basepath+f'/OISST/mhw_detrended_oisst-avhrr-v02r01.regridded1x1.monthly.{years[0]}_{years[-1]}.nc'
    f_obs = basepath+f'/OISST/mhw_detrended_oisst-avhrr-v02r01.regridded1x1.monthly.{clim_years[0]}_{clim_years[-1]}.nc'
    fobs=xr.open_dataset(f_obs)
    mods=modict['CanSIPSv3']
    
    ffor={};mhwfor={}
    for il in range(0,9):
        flist=[]
        for modi in mods:
            if il<nl[modi]:
                # Load MHWs
                f_in = f'{mhwdir}/mhw_{modi}_l{il}_detrended_{clim_years[0]}_{clim_years[1]}.nc'
                flist.append(f_in)
        ffor[il]=xr.open_mfdataset(flist,chunks={'time':1,'X':-1,'Y':-1,'M':-1},
                                    concat_dim='M',combine='nested',data_vars='minimal',
                           coords='minimal',parallel=True,preprocess=lambda f: f.drop_vars(["sst_an_thr","mhw_prob"]))
        M=40
        mhwfor[il]=ffor[il].is_mhw.data[:(-1*il),...].sum(axis=1)/M if il>0 else ffor[il].is_mhw.data.sum(axis=1)/M
    mhwobs=fobs.is_mhw.data
    lm=np.sum(fobs.is_mhw.data,axis=0)==0
    # ice mask
    f_ice=basepath+f'/OISST/oisst-avhrr-v02r01.regridded1x1.monthly.{years[0]}_{years[-1]}.nc'
    #with xr.open_dataset(f_ice) as fice:
    fice=xr.open_dataset(f_ice)
    icemask1=np.nanmax(fice.ice.data,axis=0)>0
    Athresh=.50
    Tthresh=.9
    lost=np.where(fice.ice>Athresh,1,0).sum(axis=0)
    imask=lost>(1-Tthresh)*np.shape(fice.ice)[0]
    dates=np.array([dt.datetime(years[0]+int((i)/12),(i)%12+1,15) for i in range(0,mhwobs.shape[0])])
    ix = lambda yr,mn,lead: int(yr-1991)*12+int(mn-1)-lead
    fields=['year','month','A1_obs','A2_obs']
    for lag in range(0,9):
        fields.append(f'A1_lag{lag}_weighted')
        fields.append(f'A2_lag{lag}_weighted')
        for th in range(1,41):
            fields.append(f'A1_lag{lag}_gt{th}')
            fields.append(f'A2_lag{lag}_gt{th}')
    dfA=pd.DataFrame(columns=fields)
    area=np.cos(np.pi/180*np.ones(np.shape(mhwobs[0,...]))*np.expand_dims(fobs.lat.values,axis=1))*111.3**2
    lonind = lambda lon,tol=.1:int(np.argwhere((np.abs(fobs.lon.values-lon)<tol)|\
                                           (np.abs(fobs.lon.values+360-lon)<tol)|\
                                           (np.abs(fobs.lon.values-360-lon)<tol))[0][0])
    latind = lambda lat,tol=.1:int(np.argwhere(np.abs(fobs.lat.values-lat)<tol)[0][0])
    box1=[-67,-43,42,50]
    box2=[-64,-43,51,65]
    area1=np.zeros(np.shape(area))
    area2=np.zeros(np.shape(area))
    area1[latind(box1[2]):latind(box1[3])+1,lonind(box1[0]):lonind(box1[1])+1]=area[latind(box1[2]):latind(box1[3])+1,lonind(box1[0]):lonind(box1[1])+1]
    area2[latind(box2[2]):latind(box2[3])+1,lonind(box2[0]):lonind(box2[1])+1]=area[latind(box2[2]):latind(box2[3])+1,lonind(box2[0]):lonind(box2[1])+1]
    for ind in range(0,24):
        yy=2011+int((ind+5)/12)
        mm=(ind+5)%12+1
        print(ind,yy,mm,dt.datetime.now())
        dfA.loc[ind,['year','month']]=[yy,mm]
        dfA.loc[ind,'A1_obs']=np.nansum(area1*np.ma.masked_where(imask|lm,mhwobs[ix(yy,mm,0),...]))
        dfA.loc[ind,'A2_obs']=np.nansum(area2*np.ma.masked_where(imask|lm,mhwobs[ix(yy,mm,0),...]))
        
        for il in range(0,9):
            print(dt.datetime.now(),il)
            dfA.loc[ind,[f'A1_lag{il}_weighted']]=np.nansum(area1*np.ma.masked_where(imask|lm,
                                                                    mhwfor[il][ix(yy,mm,il),...]))
            dfA.loc[ind,[f'A2_lag{il}_weighted']]=np.nansum(area2*np.ma.masked_where(imask|lm,
                                                                    mhwfor[il][ix(yy,mm,il),...]))
            for th in range(1,41):
                dfA.loc[ind,[f'A1_lag{il}_gt{th}']]=np.nansum(area1*np.where(np.ma.masked_where(imask|lm,
                                                            mhwfor[il][ix(yy,mm,il),...])>th/40-1e-5,1,0))
                dfA.loc[ind,[f'A2_lag{il}_gt{th}']]=np.nansum(area2*np.where(np.ma.masked_where(imask|lm,
                                                            mhwfor[il][ix(yy,mm,il),...])>th/40-1e-5,1,0))
    dfA.to_csv('figs/dfA.csv')
    fobs.close()
    for el in ffor.keys():
        ffor[el].close()
    fice.close()
    return

def byleadcalcs(yy,region):
    if region=='NPac':
        lonW=-170+360
        lonE=-110+360
        latS=35
        latN=65
    elif region=='NAtl':
        lonW=-110+360
        lonE=0+360
        latS=20
        latN=80
    else:
        raise Exception(f'region "{region}" not implemented; should be NPac or NAtl')
    nl = dict(zip( ['CanCM4i', 'COLA-RSMAS-CCSM4', 'GEM-NEMO', 'GFDL-SPEAR', 'NASA-GEOSS2S', 
                                'NCEP-CFSv2', 'CanESM5', 'GEM5.2-NEMO', 'GEM5-NEMO', 'CanCM4i-IC3',],
                   [       11,                 11,         11,           11,              8,
                                           9,        11,            11,          11,            11,])) 
    modict={'Jacox':['CanCM4i', 'COLA-RSMAS-CCSM4', 'GEM-NEMO', 'GFDL-SPEAR', 
                            'NASA-GEOSS2S', 'NCEP-CFSv2'],
            'CanSIPSv2':['CanCM4i','GEM-NEMO'],
            'CanSIPSv21':['CanCM4i-IC3','GEM5-NEMO'],
            'CanSIPSv3':['CanESM5','GEM5.2-NEMO'],
            'GFDLNASA':[ 'GFDL-SPEAR','NASA-GEOSS2S'],}

    basepath='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/MHW'
    clim_years = [1991, 2020]
    years=clim_years
    mhwdir = basepath+'/mhw/detrended';
    f_obs = basepath+f'/OISST/mhw_detrended_oisst-avhrr-v02r01.regridded1x1.monthly.{clim_years[0]}_{clim_years[-1]}.nc'
    fobs=xr.open_dataset(f_obs,chunks={'time':1,'X':-1,'Y':-1}).sel(X=slice(lonW,lonE),Y=slice(latS,latN))
    mods=modict['CanSIPSv3']
    ffor={};mhwfor={}
    for il in range(0,9):
        flist=[]
        for modi in mods:
            if il<nl[modi]:
                # Load MHWs
                f_in = f'{mhwdir}/mhw_{modi}_l{il}_detrended_{clim_years[0]}_{clim_years[1]}.nc'
                flist.append(f_in)
        ffor[il]=xr.open_mfdataset(flist,chunks={'time':1,'X':-1,'Y':-1,'M':-1},
                                    concat_dim='M',combine='nested',data_vars='minimal', coords='minimal',parallel=True,
                    preprocess=lambda f: f.drop_vars(["sst_an_thr","mhw_prob"])).sel(X=slice(lonW,lonE),Y=slice(latS,latN))
        M=40
        mhwfor[il]=ffor[il].is_mhw.data[:(-1*il),...].sum(axis=1)/M if il>0 else ffor[il].is_mhw.data.sum(axis=1)/M
    mhwobs=fobs.is_mhw.data
    lm=mhwobs.sum(axis=0)==0
    # ice mask
    f_ice=basepath+f'/OISST/oisst-avhrr-v02r01.regridded1x1.monthly.{years[0]}_{years[-1]}.nc'
    #with xr.open_dataset(f_ice) as fice:
    fice=xr.open_dataset(f_ice).sel(X=slice(lonW,lonE),Y=slice(latS,latN))
    icemask1=np.nanmax(fice.ice.data,axis=0)>0
    Athresh=.50
    Tthresh=.9
    lost=np.where(fice.ice>Athresh,1,0).sum(axis=0)
    imask=lost>(1-Tthresh)*np.shape(fice.ice)[0]
    lonind = lambda lon,tol=.1:int(np.argwhere((np.abs(fobs.lon.values-lon)<tol)|\
                                       (np.abs(fobs.lon.values+360-lon)<tol)|\
                                       (np.abs(fobs.lon.values-360-lon)<tol))[0][0])
    latind = lambda lat,tol=.1:int(np.argwhere(np.abs(fobs.lat.values-lat)<tol)[0][0])
    if region=='NPac':
        # region 1
        imin=lonind(-150)
        jmin=latind(45)
        A1mask=~lm&~imask
        A1mask[:jmin,:]=0
        A1mask[:,:imin]=0
        # region 2
        feez=xr.open_dataset('eezmask.nc').sel(X=slice(lonW,lonE),Y=slice(latS,latN))
        eezmask=feez.eezmask.astype(int)
        feez.close()
        A2mask=eezmask&~lm&~imask # ocean points only
    elif region=='NAtl':
        i1s=lonind(-67)
        i1e=lonind(-42)+1
        j1s=latind(42)
        j1e=latind(50)+1
        i2s=lonind(-64)
        i2e=lonind(-43)+1
        j2s=latind(51)
        j2e=latind(65)+1
        A1mask=~lm
        A1mask[:j1s,:]=0
        A1mask[j1e:,:]=0
        A1mask[:,:i1s]=0
        A1mask[:,i1e:]=0
        A2mask=~lm
        A2mask[:j2s,:]=0
        A2mask[j2e:,:]=0
        A2mask[:,:i2s]=0
        A2mask[:,i2e:]=0
    dates=np.array([dt.datetime(years[0]+int((i)/12),(i)%12+1,15) for i in range(0,mhwobs.shape[0])])
    ix = lambda yr,mn,lead: int(yr-1991)*12+int(mn-1)-lead
    fields=['year','month','A1_obs','A2_obs']
    for lag in range(0,9):
        fields.append(f'A1_lag{lag}_weighted')
        fields.append(f'A2_lag{lag}_weighted')
        for th in range(1,41,4):
            fields.append(f'A1_lag{lag}_gt{th}')
            fields.append(f'A2_lag{lag}_gt{th}')
    area=np.cos(np.pi/180*np.ones(np.shape(mhwobs[0,...]))*np.expand_dims(fobs.lat.values,axis=1))*111.3**2
    area1=np.array(area*A1mask)
    area2=np.array(area*A2mask.compute())
    dfA=pd.DataFrame(columns=fields)
    for ind in range(0,12): #ind,date in enumerate(dates):
        mm=ind+1
        print(ind,yy,mm,dt.datetime.now())
        dfA.loc[ind,['year','month']]=[yy,mm]
        dfA.loc[ind,'A1_obs']=(area1*mhwobs[ix(yy,mm,0),...]).sum().compute()
        dfA.loc[ind,'A2_obs']=(area2*mhwobs[ix(yy,mm,0),...]).sum().compute()
        
        for il in range(0,9):
            iix=ix(yy,mm,il)
            imhwfor=np.array(mhwfor[il][iix,...].compute())
            print('    ',dt.datetime.now(),il)
            if (yy==clim_years[0])&(ind<il): # data for date ind does not exist at lag il
                dfA.loc[ind,[f'A1_lag{il}_weighted']]=np.nan
                dfA.loc[ind,[f'A2_lag{il}_weighted']]=np.nan
                for th in range(1,41,4):
                    dfA.loc[ind,[f'A1_lag{il}_gt{th}']]=np.nan
                    dfA.loc[ind,[f'A2_lag{il}_gt{th}']]=np.nan
            else: # data at lag exists for date
                dfA.loc[ind,[f'A1_lag{il}_weighted']]=(area1*imhwfor).sum()
                dfA.loc[ind,[f'A2_lag{il}_weighted']]=(area2*imhwfor).sum()
                for th in range(1,41,4):
                    dfA.loc[ind,[f'A1_lag{il}_gt{th}']]=(area1*np.where(imhwfor>th/40-1e-5,1,0)).sum()
                    dfA.loc[ind,[f'A2_lag{il}_gt{th}']]=(area2*np.where(imhwfor>th/40-1e-5,1,0)).sum()
    dfA.to_csv(f'figs/dfA_{region}_y{yy}.csv')
    fobs.close()
    for el in ffor.keys():
        ffor[el].close()
    fice.close()
    return

if __name__=="__main__":
    # argument options:
    funx=sys.argv[1] # what function to execute
    ncpu=len(os.sched_getaffinity(0))
    if funx=='make_movie_Pac':
        make_movie_Pac()
    if funx=='byleadcalcsNAtl':
        yy=int(sys.argv[2])
        byleadcalcs(yy,'NAtl')
    if funx=='byleadcalcsNPac':
        yy=int(sys.argv[2])
        byleadcalcs(yy,'NPac')
    print('Done')
