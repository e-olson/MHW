#!/bin/bash

#modopt=Jacox
#modopt=CanSIPSv3
#modopt=CanSIPSv2
#modopt=CanSIPSv21
#modopt=GFDLNASA
modopt=NCEP-CFSv2
detr=1
#for modopt in CanCM4i CanESM5 GFDL-SPEAR ; do
#for modopt in GEM-NEMO GEM5-NEMO GEM5.2-NEMO ; do
for modopt in  COLA-RSMAS-CCSM4 NASA-GEOSS2S NCEP-CFSv2 ; do
  for il in 1 3 6 10 ; do
      qsub -v modopt=$modopt,detr=$detr,il=$il -N calcSEDI_${il}${modopt} -o out_calcSEDI_${il}${modopt}.txt -e err_calcSEDI_${il}${modopt}.txt subCalcSEDI.pbs
  done
done
