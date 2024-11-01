import os, sys
import subprocess
import datetime as dt
import itertools

# run with single argument: number of processors requested (maxproc)
# creates 2 sets of monthly files: concatenated daily data and monthly average data
# concatenated daily example:
#   /space/hall5/sitestore/eccc/crd/ccrn/users/reo000/data/obs/NOAA_OISST/combined/oisst-avhrr-v02r01.202012_daily.nc
# monthly average example: 
#   /space/hall5/sitestore/eccc/crd/ccrn/users/reo000/data/obs/NOAA_OISST/combined/oisst-avhrr-v02r01.202012_monthly.nc

def checkFiles(filelist):
    for ifile in filelist:
        if not os.path.isfile(ifile):
            return False # a file is missing
    return True # no files are missing

class pidlist(list):
    """ class to add fxns on top of list for storing pids from subprocess
    """
    def __init__(self,data=None):
        if data is None:
            super().__init__()
        else:
            super().__init__(data)

    def wait(self,maxpid=0,verb=False):
        #pids should be a list of output from subprocess.Popen
        #maxpid is length pid list should reach before continuing
        #       generally maxproc-1 or 0
        # remember lists are mutable so original is changed; no need to return
        ind=0
        while len(self)>maxpid:
            ind=(ind+1)%len(self)
            if self[ind].poll() is not None:
                cpid=self.pop(ind)
                pidclose(cpid,verb)
        return
        
def pidclose(pid,verb=False):
    # make sure stdout and stderr are closed and display any output there
    if verb:
        for line in pid.stdout:
            print(line)
    for line in pid.stderr:
        print(line)
    if pid.returncode!=0:
        print('returncode:',pid.returncode)
    pid.stdout.close()
    pid.stderr.close()
    return

def subprocrun(cmdlist,maxproc=1,verb=False,prepfun=None):
    # verb=True prints Popen stdout
    # cmdlist should be list of commands to run in shell
    # prepfun is function to apply to each elemnt in cmdlist, eg prepb19 to load bronx-19 before running
    if type(cmdlist)==str: # if single path string passed, convert to list
        cmdlist=[cmdlist,]
    if prepfun is not None:
        cmdlist=[prepfun(el) for el in cmdlist]
    pids=pidlist()
    for icmd in cmdlist:
        if verb:
            print(icmd)
        pids.wait(maxproc-1,verb=verb)
        pids.append(subprocess.Popen(icmd, shell=True, stdout=subprocess.PIPE,  stderr=subprocess.PIPE))
    pids.wait()
    return

years = [2021,2024] #[1991, 2020]
basepath='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/data/obs/NOAA_OISST/'
savepath='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/MHW/'
cmdlistcat=[]
cmdlistavg=[]
for iy in range(years[0],years[1]+1):
    for im in range(1,13):
        if iy<2024 | (iy==2024 and im<7): # data provisional/not downloaded from July on
            flist=[basepath+f'oisst-avhrr-v02r01.{iy}{im:02}{id:02}.nc' for id in \
                   itertools.takewhile(lambda x : (dt.datetime(iy,im,1)+dt.timedelta(days=x-1)).month==im, (el for el in range(1,33)))]
            assert checkFiles(flist) # make sure files exist
            # build command to join daily files in monthly file
            foutD=f'{basepath}combined/oisst-avhrr-v02r01.{iy}{im:02}_daily.nc' 
            cmdcat=f"ncrcat {' '.join(flist)} {foutD}"
            # build command to calc monthly average file
            foutM=f'{basepath}combined/oisst-avhrr-v02r01.{iy}{im:02}_monthly.nc' 
            cmdavg=f"ncra {foutD} {foutM}"
            cmdlistcat.append(cmdcat)
            cmdlistavg.append(cmdavg)

if __name__=="__main__":
    maxproc=int(sys.argv[1])
    subprocrun(cmdlistcat,maxproc=maxproc,verb=False)
    subprocrun(cmdlistavg,maxproc=maxproc,verb=False)
    print('Done')