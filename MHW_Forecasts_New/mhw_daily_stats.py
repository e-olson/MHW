import numpy as np
import xarray as xr
from mhw_daily_paths import * 
import datetime as dt

def calc_SEDI(mhwfor,mhwobs):
    # dim 0 must be time
    # mhwfor has extra dim at axis 1: ensemble member
    M=np.shape(mhwfor)[1]
    N_pos=np.sum(mhwfor,axis=1)
    N_neg=np.sum((mhwfor==0).astype(float),axis=1)
    TP=np.where(mhwobs==1,N_pos,0).sum(axis=0)
    TN=np.where(mhwobs==0,N_neg,0).sum(axis=0)
    FP=np.where(mhwobs==0,N_pos,0).sum(axis=0)
    FN=np.where(mhwobs==1,N_neg,0).sum(axis=0)
    # calculate SEDI, summed over time
    Nobs_pos=np.sum(mhwobs,axis=0)
    Nobs_neg=np.sum(1-mhwobs,axis=0)
    F=FP/(Nobs_neg*M)
    H=TP/(Nobs_pos*M)
    SEDI=(np.log(F)-np.log(H)-np.log(1-F)+np.log(1-H))/(np.log(F)+np.log(H)+np.log(1-F)+np.log(1-H))
    lmask=np.logical_or(np.sum(mhwfor[:,0,:,:],axis=0)==0,np.sum(mhwobs,axis=0)==0)
    return SEDI,lmask,TP,TN,FP,FN

def calcdur(MHWseries,dayson=1,daysoff=1):
    # for now, count MHWs that maybe cut off by sampling period
    # assume for now that MHWseries is 1d array, dimension is time in days
    oncount=0
    offcount=0
    MHWdur=0
    istarts=[]
    iends=[]
    durs=[]
    inMHW=False
    # iterate through elements in time series
    for ii, el in enumerate(MHWseries):
        if inMHW: # inMHW=True
            if el:   # MHW+ day while inMHW: reset offcount, increment MHWdur
                MHWdur=MHWdur+1 
                offcount=0
            else:    # MHW- day while inMHW: increment offcount, don't increment MHWdur
                offcount=offcount+1
            if offcount>=daysoff: # offcount surpassed; exit MHW; record iend, MHWdur; reset oncount
                inMHW=False
                iends.append(ii) # index of 1st point not in MHW
                durs.append(MHWdur)
                MHWdur=0
                oncount=0
        else:     # inMHW=False
            if el:   # MHW+ day while not inMHW: increment oncount
                oncount=oncount+1 # previously not in MHW, MHW+ day
            else:    # MHW- day while not inMHW: oncount=0
                oncount=0
            if oncount>=dayson: # if oncount surpassed, enter MHW; increment MHWdur; record istart; reset offcount
                inMHW=True # switch to inMHW=True
                MHWdur=oncount # oncount MHWdays have occurred
                istarts.append(ii-oncount+1)
                offcount=0
    # if series ended in MHW, record last (partial) MHW:
    if inMHW:
        iends.append(ii+1) # index has not been incremented beyond final value; index of 1st point assumed not in MHW
        durs.append(MHWdur)
    return np.array(istarts), np.array(iends), np.array(durs)

def intIntens(thresh,val,onlyPos=False):
    diff=val-thresh
    if onlyPos:
        diff=np.where(diff>0,diff,0.0)
    return np.sum(diff)

def eventIntens(istarts,iends,thresh,val,onlyPos=False):
    Intens=[]
    for ist, ien in zip(istarts,iends):
        Intens.append(intIntens(thresh[ist:ien],val[ist:ien],onlyPos))
    return Intens

def eventInd(istarts,iends,arraylen):
    arr=np.zeros(arraylen).astype(int)
    for ist, ien in zip(istarts,iends):
        arr[ist:ien]=1
    return arr

def cumEventIntens(thresh,val,inEvent,I0=0,onlyPos=False):
    Iout=[]
    I=I0
    for ith,iv,iMHW in zip(thresh,val,inEvent):
        if iMHW==1:
            if onlyPos:
                I=I+max(iv-ith,0)
            else:
                I=I+iv-ith
        else:
            I=0
        Iout.append(I)
    return np.array(Iout)


def reliability1(climyrs,ilead,qtile,detr=True,smoothClim=False,smoothTrend=False,meth=None,win=1,delt=0,qtvar='qt1',region='global'):
    if region=='global':
        fex=xr.open_mfdataset([fnameSEDI_OISST_CanESM_daily(0,climyrs, smoothClim, meth, 
                                                        win, detr, qtile, delt, qtvar, jj) \
                                        for jj in (0,60,120)],combine='nested')
        fullmask=np.isnan(fex.SEDI)
        fex.close()
    else:
        raise Exception('not yet implemented')

    fMHWO=xr.open_mfdataset([fnameOISSTMHW(climyrs, jj, qtile, smoothClim=True, meth=meth, 
                                            win=win,detr=True,delt=delt,qtvar='qt1') for jj in (0,60,120)],
                            combine='nested',parallel=True,decode_times=False,chunks={'time':10,'lat':60,'lon':360})
    ot=len(fMHWO.time)
    fMHWM=xr.open_mfdataset([fnameCanESMMHW(workdir, climyrs[0], climyrs[-1], ilead, jj,
                                            qt=qtile,detrend=detr,smoothClim=True, smoothTrend=True,
                                            meth=meth,win=win,delt=delt,qtvar=qtvar) for jj in (0,60,120)],
                            combine='nested',concat_dim='lat',parallel=True,decode_times=False)
    mcount=[]
    ocount=[]
    ips=np.arange(0,21)
    for ip in ips:
        msum=0
        osum=0
        for irt in range(0,len(fMHWM.reftime.values)):
            if irt%30==0: print(irt, dt.datetime.now())
            iii=ilead+int(fMHWM.reftime.values[irt])
            if (dt.datetime(1993,1,1)-dt.datetime(1991,1,1)).days+iii<ot: # make sure time has not exceeded observational max
                iMHWp=np.ma.masked_where(fullmask,fMHWM.isMHW.isel(reftime=irt).sum(dim='r').values)
                ind=iMHWp==ip
                del iMHWp
                msum=msum+np.sum(ind.filled(False))
                omhw=fMHWO.isel(time=(dt.datetime(1993,1,1)-dt.datetime(1991,1,1)).days+iii).isMHW.values
                osum=osum+np.sum(omhw[ind])
                del omhw
        mcount.append(float(msum))
        ocount.append(float(osum))
    mcount=np.array(mcount)
    ocount=np.array(ocount)
    fMHWO.close()
    fMHWM.close()
    return mcount, ocount, ips
