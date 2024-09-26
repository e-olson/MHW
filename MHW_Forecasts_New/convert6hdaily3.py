import os, sys
import subprocess
import datetime as dt
import xarray as xr
import dask
from dask.distributed import Client, LocalCluster

# run with two arguments: first year to process and first year not to process
# should add up to 1993, 2024
mdir='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/data/predictions/cansipsv3_daily/CanESM5'
fnameCanESMjoined=lambda mdir, yyyy, mm, dd, hh:f"{mdir}/joined/cwao_CanESM5.1p1bc-v20240611_hindcast_S{yyyy:04}{mm:02}{dd:02}{hh:02}_ocean_6hr_surface_tso.nc"
fnameCanESMdaily=lambda mdir, yyyy, mm, dd, hh:f"{mdir}/joined/cwao_CanESM5.1p1bc-v20240611_hindcast_S{yyyy:04}{mm:02}{dd:02}{hh:02}_ocean_1d_surface_tso.nc"

def fconvert(yyyy,mm,dd,hh):
    fin=fnameCanESMjoined(mdir,yyyy,mm,dd,hh)
    fout=fnameCanESMdaily(mdir,yyyy,mm,dd,hh)
    print(fout,flush=True)
    ff=xr.open_dataset(fin,decode_times=False).chunk({'lat':30,'lon':30})
    ff2=ff.drop_vars(['realization','hcrs']).rename({'record':'r'})
    ff3=ff2.coarsen(leadtime=4).mean()
    ff3.tso.assign_attrs({'postprocess':'daily time average, [(6,12,18,24),...]'})
    ff3.to_netcdf(fout,mode='w')
    del ff3; del ff2; ff.close(); del ff;
    return

#mdir='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/data/predictions/cansipsv3_daily/CanESM5'
#fnameCanESMjoined=lambda mdir, yyyy, mm, dd, hh:f"{mdir}/joined/cwao_CanESM5.1p1bc-v20240611_hindcast_S{yyyy:04}{mm:02}{dd:02}{hh:02}_ocean_6hr_surface_tso.nc"
#fnameCanESMdaily=lambda mdir, yyyy, mm, dd, hh:f"{mdir}/joined/cwao_CanESM5.1p1bc-v20240611_hindcast_S{yyyy:04}{mm:02}{dd:02}{hh:02}_ocean_1d_surface_tso.nc"
#
#def fconvert(yyyy,mm,dd,hh):
#    fin=fnameCanESMjoined(mdir,yyyy,mm,dd,hh)
#    fout=fnameCanESMdaily(mdir,yyyy,mm,dd,hh)
#    #if not os.path.exists(fout):
#    print(fout,flush=True)
#    ff=xr.open_dataset(fin,decode_times=False)
#    #ff0=ff.chunk(chunks={'record':-1,'leadtime':16,'lat':-1,'lon':-1})
#    ff2=ff.drop_vars(['realization','hcrs']).rename({'record':'r'})
#    ff3=ff2.coarsen(leadtime=4).mean()
#    #ff3.tso.assign_attrs({'postprocess':'daily time average, [(6,12,18,24),...]'})
#    ff3.to_netcdf(fout,mode='w')
#    del ff3; del ff2; 
#    ff.close(); del ff
#    return


if __name__=="__main__":
    dd=1
    hh=0
    yyyy=int(sys.argv[1])
    mm=int(sys.argv[2])
    fconvert(yyyy,mm,dd,hh)
    print('done:',yyyy,mm)
