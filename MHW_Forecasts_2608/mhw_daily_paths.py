
# run with two arguments: first year to process and first year not to process
# should add up to 1993, 2024
workdir={1:'/fs/site7/eccc/crd/cccma/users/reo000/work/MHW_daily', # new version replacing MHW_daily #MHW_1d
         5:'/fs/site7/eccc/crd/cccma/users/reo000/work/MHW_5d'}
mdirC5='/fs/site7/eccc/crd/cccma/users/reo000/data/predictions/cansipsv3_daily/CanESM5'
osrcdir='/fs/site7/eccc/crd/cccma/users/reo000/data/obs/NOAA_OISST/combined'

fnameCanESMjoined=lambda mdir, yyyy, mm, dd, hh: \
       f"{mdir}/joined/cwao_CanESM5.1p1bc-v20240611_hindcast_S{yyyy:04}{mm:02}{dd:02}{hh:02}_ocean_6hr_surface_tso.nc"
fnameCanESMdaily=lambda mdir, yyyy, mm, dd, hh: \
       f"{mdir}/joined/cwao_CanESM5.1p1bc-v20240611_hindcast_S{yyyy:04}{mm:02}{dd:02}{hh:02}_ocean_1d_surface_tso.nc"
fnameCanESM5d=lambda mdir, yyyy, mm, dd, hh: \
       f"{mdir}/joined/cwao_CanESM5.1p1bc-v20240611_hindcast_S{yyyy:04}{mm:02}{dd:02}{hh:02}_ocean_5d_surface_tso.nc"
def fnameCanESMClim(mdir, climyfirst, climylast, mm, smoothClim=False,method=None,window=1,L=1):
    smstr=f'_smooth_{method}{window}' if smoothClim else ''
    return f"{mdir}/clim/clim{smstr}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
            f"Mon{mm:02}_ocean_{L}d_surface_tso.nc"
#fnameCanESMAnom=lambda mdir, climyfirst,climylast,lfirst, llast, mm: \
#       f"{mdir}/anom/anom_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_SMon{mm:02}_"\
#       f"L_{lfirst:03}_{llast:03}_ocean_1d_surface_tso.nc"
#fnameCanESMAnomSClim=lambda mdir, climyfirst,climylast,lfirst,llast,mm,meth,win:\
#       f"{mdir}/anom/anom_sclim{meth}{win}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_SMon{mm:02}_"\
#       f"L_{lfirst:03}_{llast:03}_ocean_1d_surface_tso.nc"
## before reorg by lead:
def fnameCanESMAnom0(mdir,climyfirst,climylast,yyyy,mm,smoothedClim=False,meth=None,win=1,L=1):
    strSClim=f'_sclim{meth}{win}' if smoothedClim else ''
    return f"{mdir}/anom/anom{strSClim}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
        f"SYr{yyyy:04}Mon{mm:02}_ocean_{L}d_surface_tso.nc"
## after reorg by lead:
#def fnameCanESMAnomByLeadNoDetr(mdir, climyfirst, climylast, ilead, istartlat,  smoothClim=False,meth=None,win=1,L=1):
#    strSClim=f'_sclim{meth}{win}' if smoothClim else ''
#    return f"{mdir}/byLead/anomByLead{strSClim}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
#       f"L{ilead:03}_j{istartlat:03}_ocean_{L}d_surface_tso.nc"
#def fnameCanESMAnomDetrByLead(mdir, climyfirst, climylast, ilead, istartlat, smoothClim=False,smoothTrend=False,meth=None,win=1,L=1): 
#    subdir='byLeadDetr' if (smoothClim or smoothTrend) else 'byLeadDetrIndiv2'
#    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
#    strSTrend=f'_TrS{meth}{win}' if smoothClim else ''
#    return f"{mdir}/{subdir}/anomDetrByLead{strSClim}{strSTrend}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
#       f"L{ilead:03}_j{istartlat:03}_ocean_{L}d_surface_tso.nc"
def fnameCanESMAnomByLead(mdir, climyfirst, climylast, ilead, istartlat,  smoothClim=False,smoothTrend=False,meth=None,win=1,L=1,detrended=False):
    if detrended:
        subdir='byLeadDetr' if (smoothClim or smoothTrend) else 'byLeadDetrIndiv2'
        strSTrend=f'_DetrTrS{meth}{win}' if smoothTrend else '_Detr'
    else:
        subdir='byLead'
        strSTrend=f''
    strSClim=f'_sclim{meth}{win}' if smoothClim else ''
    return f"{mdir}/{subdir}/anombylead{strSClim}{strSTrend}_cwao_canesm5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
            f"l{ilead:03}_j{istartlat:03}_ocean_{L}d_surface_tso.nc"
# lineary fit:
def fnameCanESMDetrFitByLead(mdir, climyfirst, climylast, ilead, istartlat, smoothClim=False,smoothTrend=False,meth=None,win=1,L=1):
    sourcedesig = f'_ClimS{meth}{win}' if smoothClim else ''
    trdesig = f'_smoothed{meth}{win}' if smoothTrend else ''
    subdir='byLeadDetr'
    #subdir='byLeadDetrIndiv2' if sourcedesig=='' else 'byLeadDetr'
    return f"{mdir}/{subdir}/fitDetrByLead{sourcedesig}{trdesig}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_ocean_{L}d_surface_tso.nc"
## smoothed linear fit:
#fnameCanESMDetrFitByLeadS=lambda mdir, climyfirst, climylast, ilead, istartlat, meth, win, sourcedesig='': \
#       f"{mdir}/byLeadDetr/fitDetrByLead{sourcedesig}_smoothed{meth}{win}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
#       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
def fnameCanESMAnomQtile(mdir, climyfirst, climylast, ilead, istartlat, qt, detrend=False, smoothClim=False,smoothTrend=False,meth=None,win=1,delt=0,L=1): 
    if detrend: 
        subdir='byLeadDetr' if (smoothClim or smoothTrend) else 'byLeadDetrIndiv2'
    else:
        subdir='byLead' if (smoothClim or smoothTrend) else 'byLeadIndiv2'
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    strSTrend=f'_TrS{meth}{win}' if smoothTrend else ''
    strdelt=f'_delt{delt}' # reflects number of lead time days to pool together
    qstr='{:.2f}'.format(qt).replace('.','_')
    detrstr=f"Detr" if detrend else ""
    return f"{mdir}/{subdir}/qtile{detrstr}ByLead{strSClim}{strSTrend}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
            f"L{ilead:03}{strdelt}_j{istartlat:03}_q{qstr}_ocean_{L}d_surface_tso.nc"
def fnameCanESMMHW(mdir, climyfirst, climylast, ilead, istartlat, qt, detrend=False, smoothClim=False,smoothTrend=False,meth=None,win=1,delt=0,qtvar='qt1',L=1):
    if detrend: 
        subdir='byLeadDetrMHW' if (smoothClim or smoothTrend) else 'byLeadDetrIndiv2MHW'
    else:
        subdir='byLeadMHW' if (smoothClim or smoothTrend) else 'byLeadIndiv2MHW'
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    strSTrend=f'_TrS{meth}{win}' if smoothTrend else ''
    strdelt=f'_delt{delt}' # reflects number of lead time days to pool together
    qstr='{:.2f}'.format(qt).replace('.','_')
    detrstr=f"Detr" if detrend else ""
    qvstr='_'+qtvar
    return f"{mdir}/{subdir}/MHW{detrstr}ByLead{strSClim}{strSTrend}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
            f"L{ilead:03}{strdelt}_j{istartlat:03}{qvstr}_q{qstr}_ocean_{L}d_surface_tso.nc"
#fnameCanESMMHW=lambda mdir, climyfirst, climylast, ilead, istartlat, qt: \
#       f"{mdir}/byLeadMHW/MHWByLead_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
#       f"L{ilead:03}_j{istartlat:03}_q{'{:.2f}'.format(qt).replace('.','_')}_ocean_1d_surface_tso.nc"
fnameOISSTDaily = lambda iy, im:\
       f"{osrcdir}/oisst-avhrr-v02r01.{iy}{im:02}_daily.nc"
def fnameOISSTGrid2(yrlims,L=1): 
    return f"{workdir[L]}/OISST/oisst-avhrr-v02r01.regridded1x1g2.{'daily' if L==1 else str(L)+'d'}.{yrlims[0]}_{yrlims[-1]}.nc"
def fnameOISSTDailyClim(climyfirst, climylast,smoothedClim=False,meth=None,win=1,L=1):
    sclimstr=f'_smooth_{meth}{win}' if smoothedClim else ''
    return f"{workdir[L]}/OISST/climSST{sclimstr}_oisst-avhrr-v02r01.regridded1x1g2.{'daily' if L==1 else str(L)+'d'}_C{climyfirst:04}_{climylast:04}.nc"
def fnameOISSTAnom(yrlims, climyrs, istartlat, smoothClim=False, meth=None, win=1,L=1,detrended=False):
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    return f"{workdir[L]}/OISST/oisst_anom{'_detr' if detrended else ''}{strSClim}_C{climyrs[0]:04}_{climyrs[-1]:04}-avhrr-v02r01.regridded1x1g2.{'daily' if L==1 else str(L)+'day'}.{yrlims[0]}_{yrlims[-1]}_j{istartlat}.nc"
def fnameOISSTDetrFit(climyrs, istartlat, smoothClim=False,meth=None,win=1,L=1):
    sourcedesig = f'_ClimS{meth}{win}' if smoothClim else ''
    return f"{workdir[L]}/OISST/fitDetr{sourcedesig}_oisst-avhrr-v02r01.regridded1x1g2.{'daily' if L==1 else str(L)+'d'}_C{climyrs[0]:04}_{climyrs[1]:04}.nc"
def fnameOISSTQTile(climyrs, istartlat, qt, smoothClim=False, meth=None, win=1,detr=True,delt=0,L=1):
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    strdelt=f'_delt{delt}' # reflects number of lead time days to pool together
    qstr='{:.2f}'.format(qt).replace('.','_')
    return f"{workdir[L]}/OISST/oisst_qtile{'_detr' if detr else ''}{strSClim}_C{climyrs[0]:04}_{climyrs[-1]:04}_q{qstr}{strdelt}-avhrr-v02r01.regridded1x1g2.{'daily' if L==1 else str(L)+'day'}.j{istartlat}.nc"
def fnameOISSTMHW(climyrs, istartlat, qt, smoothClim=False, meth=None, win=1, detr=True, delt=0,qtvar='qt1',L=1):
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    strdelt=f'_delt{delt}' # reflects number of lead time days to pool together
    qstr='{:.2f}'.format(qt).replace('.','_')
    qvstr='_'+qtvar
    return f"{workdir[L]}/OISST/oisst_MHW_{'_detr' if detr else ''}{strSClim}_C{climyrs[0]:04}_{climyrs[-1]:04}_q{qstr}{strdelt}-avhrr-v02r01.regridded1x1g2.{'daily' if L==1 else str(L)+'day'}.j{istartlat}{qvstr}.nc"
def fnameSEDI_OISST_CanESM_daily(lead,climyrs, smoothClim, meth, win, detr, qt, delt, qtvar, jj,L=1):
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    strdelt=f'_delt{delt}' # reflects number of lead time days to pool together
    detrstr='_detr' if detr else ''
    qstr='_{:.2f}'.format(qt).replace('.','_')
    qvstr='_'+qtvar
    return f"{workdir[L]}/stats/SEDI_OISST_CanESM_{'daily' if L==1 else str(L)+'day'}_L{lead:03}_C{climyrs[0]:04}_{climyrs[-1]:04}{strSClim}{detrstr}{qstr}{strdelt}_j{jj:03}{qvstr}.nc"
def fnameReli(lead,climyrs, smoothClim, meth, win, detr, qt, delt,qtvar,region,L=1):
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    strdelt=f'_delt{delt}' # reflects number of lead time days to pool together
    detrstr='_detr' if detr else ''
    qstr='_{:.2f}'.format(qt).replace('.','_')
    qvstr='_'+qtvar
    return f"{workdir[L]}/stats/Reli_OISST_CanESM_{'daily' if L==1 else str(L)+'day'}_L{lead:03}_C{climyrs[0]:04}_{climyrs[-1]:04}{strSClim}{detrstr}{qstr}{strdelt}_{qvstr}_{region}.npz"
