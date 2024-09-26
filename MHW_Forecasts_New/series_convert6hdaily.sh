#!/bin/bash

#for t0 in {1993..2003..2}
#for t0 in {2005..2011..2}
#for t0 in {2017..2023..2}
for t0 in {1993..2024}
do
    t1=$((t0 + 1))
    echo $t1
    qsub -v t0=$t0,t1=$t1 -N df6h_${t0}_${t1} -o out_df6h_${t0}_${t1}.txt -e err_df6h_${t0}_${t1}.txt convert6hdaily.pbs
done
