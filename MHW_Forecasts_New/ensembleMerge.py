import os, sys
import subprocess
import datetime as dt

# run with single argument: number of processors requested (maxproc)

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

years = [1993, 2024]
dd=1
hh=0
mdir='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/data/predictions/cansipsv3_daily/CanESM5'
fnameCanESM=lambda mdir, yyyy, mm, dd, hh, rr:f"{mdir}/cwao_CanESM5.1p1bc-v20240611_hindcast_S{yyyy:04}{mm:02}{dd:02}{hh:02}_ocean_6hr_surface_tso_r{rr:02}i00p00.nc"
fnameCanESMjoined=lambda mdir, yyyy, mm, dd, hh:f"{mdir}/joined/cwao_CanESM5.1p1bc-v20240611_hindcast_S{yyyy:04}{mm:02}{dd:02}{hh:02}_ocean_6hr_surface_tso.nc"
cmdlist=[]
for yyyy in range(years[0],years[1]+1):
    for mm in range(1,13):
        if yyyy==2024 and mm>6:
            break
        flist=[fnameCanESM(mdir,yyyy,mm,dd,hh,ii) for ii in range(1,21)]
        try:
            assert checkFiles(flist) # make sure files exist
        except:
            print(flist)
            raise
        cmd=f"ncecat {' '.join(flist)} {fnameCanESMjoined(mdir,yyyy,mm,dd,hh)}"
        cmdlist.append(cmd)

if __name__=="__main__":
    maxproc=int(sys.argv[1])
    subprocrun(cmdlist,maxproc=maxproc,verb=True)
    print('Done')
