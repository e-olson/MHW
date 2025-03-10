
# run with two arguments: first year to process and first year not to process
# should add up to 1993, 2024
workdir='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/MHW_daily'
mdirC5='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/data/predictions/cansipsv3_daily/CanESM5'
osrcdir='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/data/obs/NOAA_OISST/combined'

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
## before reorg by lead:
fnameCanESMAnom=lambda mdir, climyfirst, climylast, yyyy, mm: \
       f"{mdir}/anom/anom_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"SYr{yyyy:04}Mon{mm:02}_ocean_1d_surface_tso.nc"
fnameCanESMAnomSClim=lambda mdir, climyfirst, climylast, yyyy, mm, meth, win: \
       f"{mdir}/anom/anom_sclim{meth}{win}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"SYr{yyyy:04}Mon{mm:02}_ocean_1d_surface_tso.nc"
## after reorg by lead:
def fnameCanESMAnomByLeadNoDetr(mdir, climyfirst, climylast, ilead, istartlat,  smoothClim=False,meth=None,win=1):
    strSClim=f'_sclim{meth}{win}' if smoothClim else ''
    return f"{mdir}/byLead/anomByLead{strSClim}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
#fnameCanESMAnomByLead=lambda mdir, climyfirst, climylast, ilead, istartlat: \
#       f"{mdir}/byLead/anomByLead_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
#       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
#fnameCanESMAnomByLeadSClim=lambda mdir, climyfirst, climylast, ilead, istartlat,meth,win: \
#       f"{mdir}/byLead/anomByLead_sclim{meth}{win}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
#       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
#fnameCanESMAnomDetrByLeadIndiv=lambda mdir, climyfirst, climylast, ilead, istartlat: \
#       f"{mdir}/byLeadDetrIndiv2/anomDetrByLead_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
#       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
#fnameCanESMAnomDetrByLead=lambda mdir, climyfirst, climylast, ilead, istartlat: \
#       f"{mdir}/byLeadDetr/anomDetrByLead_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
#       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
# lineary fit:
def fnameCanESMDetrFitByLead(mdir, climyfirst, climylast, ilead, istartlat, smoothClim=False,meth=None,win=1):
    sourcedesig = f'_ClimS{smoothmethod}{window}' if smoothClim else ''
    subdir='byLeadDetr'
    #subdir='byLeadDetrIndiv2' if sourcedesig=='' else 'byLeadDetr'
    return f"{mdir}/{subdir}/fitDetrByLead{sourcedesig}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
# smoothed linear fit:
fnameCanESMDetrFitByLeadS=lambda mdir, climyfirst, climylast, ilead, istartlat, meth, win, sourcedesig='': \
       f"{mdir}/byLeadDetr/fitDetrByLead{sourcedesig}_smoothed{meth}{win}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
def fnameCanESMAnomDetrByLead(mdir, climyfirst, climylast, ilead, istartlat, smoothClim=False,smoothTrend=False,meth=None,win=1): 
    subdir='byLeadDetr' if (smoothClim or smoothTrend) else 'byLeadDetrIndiv2'
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    strSTrend=f'_TrS{meth}{win}' if smoothClim else ''
    return f"{mdir}/{subdir}/anomDetrByLead{strSClim}{strSTrend}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
       f"L{ilead:03}_j{istartlat:03}_ocean_1d_surface_tso.nc"
def fnameCanESMAnomQtile(mdir, climyfirst, climylast, ilead, istartlat, qt, detrend=False, smoothClim=False,smoothTrend=False,meth=None,win=1,delt=0): 
    if detrend: 
        subdir='byLeadDetr' if (smoothClim or smoothTrend) else 'byLeadDetrIndiv2'
    else:
        subdir='byLead' if (smoothClim or smoothTrend) else 'byLeadIndiv2'
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    strSTrend=f'_TrS{meth}{win}' if smoothClim else ''
    strdelt=f'_delt{delt}' # reflects number of lead time days to pool together
    qstr='{:.2f}'.format(qt).replace('.','_')
    detrstr=f"Detr" if detrend else ""
    return f"{mdir}/{subdir}/qtile{detrstr}ByLead{strSClim}{strSTrend}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
            f"L{ilead:03}{strdelt}_j{istartlat:03}_q{qstr}_ocean_1d_surface_tso.nc"
def fnameCanESMMHW(mdir, climyfirst, climylast, ilead, istartlat, qt, detrend=False, smoothClim=False,smoothTrend=False,meth=None,win=1,delt=0,qtvar='qt1'):
    if detrend: 
        subdir='byLeadDetrMHW' if (smoothClim or smoothTrend) else 'byLeadDetrIndiv2MHW'
    else:
        subdir='byLeadMHW' if (smoothClim or smoothTrend) else 'byLeadIndiv2MHW'
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    strSTrend=f'_TrS{meth}{win}' if smoothClim else ''
    strdelt=f'_delt{delt}' # reflects number of lead time days to pool together
    qstr='{:.2f}'.format(qt).replace('.','_')
    detrstr=f"Detr" if detrend else ""
    qvstr='_'+qtvar
    return f"{mdir}/{subdir}/MHW{detrstr}ByLead{strSClim}{strSTrend}_cwao_CanESM5.1p1bc-v20240611_hindcast_C{climyfirst:04}_{climylast:04}_"\
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
fnameOISSTDailyClimSmooth=lambda climyfirst, climylast,method,window: \
       f"{workdir}/OISST/climSST_smooth_{method}{window}_oisst-avhrr-v02r01.regridded1x1g2.daily_C{climyfirst:04}_{climylast:04}.nc"
def fnameOISSTAnom(yrlims, climyrs, istartlat, smoothClim=False, meth=None, win=1):
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    return f"{workdir}/OISST/oisst_anom{strSClim}_C{climyrs[0]:04}_{climyrs[-1]:04}-avhrr-v02r01.regridded1x1g2.daily.{yrlims[0]}_{yrlims[-1]}_j{istartlat}.nc"
def fnameOISSTAnomDetr(yrlims, climyrs, istartlat, smoothClim=False, meth=None, win=1):
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    return f"{workdir}/OISST/oisst_anom_detr{strSClim}_C{climyrs[0]:04}_{climyrs[-1]:04}-avhrr-v02r01.regridded1x1g2.daily.{yrlims[0]}_{yrlims[-1]}_j{istartlat}.nc"
def fnameOISSTQTile(climyrs, istartlat, qt, smoothClim=False, meth=None, win=1,detr=True,delt=0):
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    strdelt=f'_delt{delt}' # reflects number of lead time days to pool together
    qstr='{:.2f}'.format(qt).replace('.','_')
    return f"{workdir}/OISST/oisst_qtile{'_detr' if detr else ''}{strSClim}_C{climyrs[0]:04}_{climyrs[-1]:04}_q{qstr}{strdelt}-avhrr-v02r01.regridded1x1g2.daily.j{istartlat}.nc"
def fnameOISSTMHW(climyrs, istartlat, qt, smoothClim=False, meth=None, win=1, detr=True, delt=0,qtvar='qt1'):
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    strdelt=f'_delt{delt}' # reflects number of lead time days to pool together
    qstr='{:.2f}'.format(qt).replace('.','_')
    qvstr='_'+qtvar
    return f"{workdir}/OISST/oisst_MHW_{'_detr' if detr else ''}{strSClim}_C{climyrs[0]:04}_{climyrs[-1]:04}_q{qstr}{strdelt}-avhrr-v02r01.regridded1x1g2.daily.j{istartlat}{qvstr}.nc"
def fnameSEDI_OISST_CanESM_daily(lead,climyrs, smoothClim, meth, win, detr, qt, delt, qtvar, jj):
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    strdelt=f'_delt{delt}' # reflects number of lead time days to pool together
    detrstr='_detr' if detr else ''
    qstr='_{:.2f}'.format(qt).replace('.','_')
    qvstr='_'+qtvar
    return f"{workdir}/stats/SEDI_OISST_CanESM_daily_L{lead:03}_C{climyrs[0]:04}_{climyrs[-1]:04}{strSClim}{detrstr}{qstr}{strdelt}_j{jj:03}{qvstr}.nc"
def fnameReli(lead,climyrs, smoothClim, meth, win, detr, qt, delt,qtvar,region):
    strSClim=f'_ClimS{meth}{win}' if smoothClim else ''
    strdelt=f'_delt{delt}' # reflects number of lead time days to pool together
    detrstr='_detr' if detr else ''
    qstr='_{:.2f}'.format(qt).replace('.','_')
    qvstr='_'+qtvar
    return f"{workdir}/stats/Reli_OISST_CanESM_daily_L{lead:03}_C{climyrs[0]:04}_{climyrs[-1]:04}{strSClim}{detrstr}{qstr}{strdelt}_{qvstr}_{region}.npz"
