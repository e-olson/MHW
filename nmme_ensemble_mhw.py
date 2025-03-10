#!/usr/bin/env python
# coding: utf-8
import xarray as xr
import sys

il=int(sys.argv[1])
is_detrend=int(sys.argv[2]) #1 to use detrended anomalies (default), 0 otherwise

# Years to process and years to use as climatology for thresholds
years = [1991, 2020]
clim_years = [1991, 2020]

basepath='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/MHW'
# Input/output directory
if is_detrend:
    dir = basepath+'/mhw/detrended';
else:
    dir = basepath+'/mhw';

# Model names
#           1              2               3            4              5             6
mods = ['CanCM4i', 'COLA-RSMAS-CCSM4', 'GEM-NEMO', 'GFDL-SPEAR', 'NASA-GEOSS2S', 'NCEP-CFSv2']#
nmod = len(mods)
nl = dict(zip(mods,[11, 11, 11, 11, 8, 9])) # Max lead time for each model

# Loop through lead times
print('\nBuilding multimodel ensemble for NMME MHW forecasts...\n')
print('Lead')
print(' ',il)
    
# Loop through models
is_mhw_ens=[]
mlist=[]
flist=[]
for modi in mods:

    if il<nl[modi]:
        
        # Load MHWs
        if is_detrend:
            f_in = f'{dir}/mhw_{modi}_l{il}_detrended_{years[0]}_{years[1]}.nc'
        else:
            f_in = f'{dir}/mhw_{modi}_l{il}_{years[0]}_{years[1]}.nc'
            print(f_in)
        flist.append(f_in)
fin=xr.open_mfdataset(flist,chunks={'X':10,'Y':10,'M':-1},concat_dim='M',combine='nested',data_vars='minimal',
                   coords='minimal',parallel=True,preprocess=lambda f: f.drop_vars(["sst_an_thr","mhw_prob"]) )
fin2=fin.chunk({'X':10,'Y':10,'M':-1})
out=fin2.mean(dim='M').rename({'is_mhw':'mhw_prob'})
# Save to file
if is_detrend:
    f_out = f'{dir}/mhw_MME_l{il}_detrended_{years[0]}_{years[1]}.nc'
    print(f_out)
else:
    f_out = f'{dir}/mhw_MME_l{il}_{years[0]}_{years[1]}.nc'
out.to_netcdf(f_out,mode='w')
fin.close()
print('\nDone\n\n')


