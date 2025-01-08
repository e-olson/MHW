def trismooth(t,vals,L=30):
    # t is values assoc with 1st dim
    # smooths over 1st dim
    # if vector, add dim:
    delt=t[1]-t[0]
    alpha=1
    if len(np.shape(vals))==1:
        vals=np.expand_dims(vals,axis=1)
    fil=np.empty(np.shape(vals))
    for ind, ti in enumerate(t):
        diff=np.abs(ti-t)
        Leff=min(L,alpha*(ti-t[0]+1)*delt,alpha*(t[-1]-ti+1)*delt)# do not smooth beginning and end asymmetrically
        weight=_add_dims(np.maximum(Leff-diff,0),vals)
        fil[ind,...]=np.divide(np.nansum(weight*vals,0),np.nansum(weight*~np.isnan(vals),0),
                               out=np.nan*da.array(np.ones(np.shape(vals)[1:])),
                               where=np.nansum(weight*~np.isnan(vals),0)>0)
    return fil

def calcClim_CanESM5(climyrs):#,nlead):
    for mm in range(1,13): # month loop
        print(f"Month:{mm} {dt.datetime.now()}",flush=True)
        #for ix in range(0,int(nlead/5)):                                                                                                   
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