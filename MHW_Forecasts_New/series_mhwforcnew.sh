#!/bin/bash

for il in {0..11}; do
    qsub -v il=$il -N mhwforcnew_${il} -o out_mhwforcnew_${il}.txt -e err_mhwforcnew_${il}.txt MHW_Forecasts_New.pbs
done
