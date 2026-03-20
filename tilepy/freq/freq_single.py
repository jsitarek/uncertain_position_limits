import numpy as np
import astropy.units as u
import gammapy.irf
from gammapy.stats import WStatCountsStatistic
from astropy.coordinates import angular_separation
from numpy.random import Generator, PCG64
import scipy
import sys

_, infile, seed, trueflux, niter=sys.argv
seed=int(seed)
trueflux=float(trueflux)
niter=int(niter)
print(f"{niter} iterations with trueflux of {trueflux} and seed {seed} from {infile}")
rng = Generator(PCG64(seed=seed))
file=np.load(infile, allow_pickle=True)
pointings = file['pointings']
angleoffset=file['angleoffset'].item()
timeperpoint=file['timeperpoint'].item()
ras=file['ras']
decs=file['decs']
ras0=file['ras0']
decs0=file['decs0']
mypixarea=file['mypixarea']
etrue=file['etrue']
etruehi=file['etruehi']
etruelo=file['etruelo']
eest=file['eest']
mask_ok=file['mask_ok']
mask_point=file['mask_point']
gw_prob=file['gw_prob']
offset_mig=file['offset_mig']
emigmy=file['emigmy']
bgd=file['bgd']
kernels=file['kernels']
kernelhalfsize=file['kernelhalfsize']
irffile=str(file['irffile'])
off=file['off']
bgdalpha=file['bgdalpha']
psf_avr=file['psf_avr']*u.Unit('deg-2') # stored without a unit
thetabins=file['thetabins']

binsdec=len(decs0)
binsra=len(ras0)
print(f"{binsdec} bins in DEC, {binsra} bins in RA, IRFs: {irffile}, {bgdalpha}")
irfs = gammapy.irf.load_irf_dict_from_file(irffile)

def draw_ra_dec():
    idx=rng.choice(a=gw_prob.shape[0]*gw_prob.shape[1], p=gw_prob.flatten()/np.sum(gw_prob))
    idx=np.unravel_index(idx,gw_prob.shape)
    dec_true=rng.uniform(decs[idx[0]], decs[idx[0]+1])
    ra_true=rng.uniform(ras[idx[1]], ras[idx[1]+1])
    return dec_true, ra_true

# Crab flux, must be the same as in the other script!!!
def srcflux(en): # en in TeV
    #return 3.23e-11 * (en/1.)**(-2.47-0.24*np.log10(en/1.)) *u.Unit('TeV-1 cm-2 s-1') # Crab Roberta
    return 2.83e-11 * (en/1.)**(-2.62) *u.Unit('TeV-1 cm-2 s-1') # HEGRA Crab

def create_excess(pointings, emigmy, trueflux, irfs, dec_true=None, ra_true=None, quiet=True):
    # randomize the position of the alert
    if dec_true == None or ra_true == None:
        dec_true, ra_true = draw_ra_dec()
        offsets=angular_separation(pointings[:,0]*u.deg, pointings[:,1]*u.deg, ra_true*u.deg, dec_true*u.deg).to_value('deg')
        if not quiet:
            print('offsets [deg]=',offsets)
        d_true=angular_separation(ras0[np.newaxis,:]*u.deg, decs0[:, np.newaxis]*u.deg, ra_true*u.deg, dec_true*u.deg)
        
        # find the bins in which the true signal resides
        #print(f" dDEC={np.abs(dec_true-decs0).min()}, dRA={np.abs(ra_true-ras0).min()}")
        idec_true=np.abs(dec_true-decs0).argmin()
        ira_true=np.abs(ra_true-ras0).argmin()
        bb=np.zeros((len(decs0), len(ras0)))
        bb[mask_ok]=np.arange(0, mask_ok.sum())
        imask_true=int(bb[idec_true, ira_true].round())
        #print(imask_true)
        #print(ras0[ira_true], decs0[idec_true])
        #print(ras1[imask_true], decs1[imask_true]) 
        
    ## calculate excess    
    maxoffset=angleoffset # deg
    excess_eest=np.zeros([len(eest), binsdec, binsra])
    for ra_p, dec_p in pointings[offsets<maxoffset]:
        d_true=angular_separation(ras0[np.newaxis,:]*u.deg, decs0[:, np.newaxis]*u.deg, ra_true*u.deg, dec_true*u.deg)
        psfbin=np.abs(d_true.to_value('deg')[mask_point][...,np.newaxis]-thetabins).argmin(axis=-1)
        #d_true=np.sqrt(d_true**2 + (binpatch*binsize*u.deg)**2)
        offset=angular_separation(ra_p*u.deg, dec_p*u.deg, ra_true*u.deg, dec_true*u.deg)
        ioff=np.argmin(np.abs(offset.to_value('deg')-offset_mig))
        #print(f"closest offset to {offset.to('deg')} is {offset_mig[ioff]}")
        emig=emigmy[:,:,ioff]
        for ien in range(len(etrue)):
            flux=trueflux*srcflux(etrue[ien])*(etruehi[ien]-etruelo[ien])*u.TeV # *0.1
            thisexposure=irfs['aeff'].evaluate(offset=offset, energy_true=etrue[ien]*u.TeV)*timeperpoint*u.s
            if not quiet:
                print(f"E={etrue[ien]:.3f}, Nph={(flux*thisexposure).to('')}")

            new_excess_etrue=(flux*thisexposure*psf_avr[ioff, ien, psfbin]*mypixarea[mask_point] *u.sr).to_value('')
            new_excess_eest=new_excess_etrue[np.newaxis, ...]*emig[ien,:, np.newaxis] # eest, etrue, decra
            excess_eest[:,mask_point]+=new_excess_eest
    return excess_eest, dec_true, ra_true, imask_true

def create_on_off (bgd, excess_eest, kernels=None, quiet=True):
    ## draw a random realization from poissonian
    off=rng.poisson(bgd*bgdalpha)/bgdalpha
    on=rng.poisson(bgd+excess_eest)
    if kernels is None:
        return on, off
    else:
        off2=np.zeros(off.shape)
        on2=np.zeros(off.shape)
        for ien in range(len(eest)):
            if not quiet:
                print(f"folding Eest={eest[ien]}")
            off2[ien,:,:]=scipy.signal.convolve2d(off[ien,:,:], kernels[ien], mode='full', boundary='fill', fillvalue=0)[kernelhalfsize[ien]:-kernelhalfsize[ien], kernelhalfsize[ien]:-kernelhalfsize[ien]]
            on2[ien,:,:]=scipy.signal.convolve2d(on[ien,:,:], kernels[ien], mode='full', boundary='fill', fillvalue=0)[kernelhalfsize[ien]:-kernelhalfsize[ien], kernelhalfsize[ien]:-kernelhalfsize[ien]]
        return on, off, on2, off2


tsmax_all = np.zeros((len(eest), niter))
ras_true = np.zeros(niter)
decs_true= np.zeros(niter)
for iiter in np.arange (niter):
    print(iiter)
    excess_eest_it, dec_true, ra_true,_ = create_excess(pointings=pointings, emigmy=emigmy, trueflux=trueflux, irfs=irfs)
    #_,_, on2_it, off2_it = create_on_off (bgd=off, excess_eest=excess_eest_it, kernels=kernels) # background from particular realization
    _,_, on2_it, off2_it = create_on_off (bgd=bgd, excess_eest=excess_eest_it, kernels=kernels) # true background
    ras_true[iiter]=ra_true
    decs_true[iiter]=dec_true
    for ien in range(len(eest)):
        shp=np.ones(on2_it[ien,:,:].shape)[mask_ok]
        wst=WStatCountsStatistic (n_on=on2_it[ien,:,:][mask_ok], n_off=off2_it[ien,:,:][mask_ok]*bgdalpha, alpha=1./bgdalpha*shp, mu_sig=0*shp) 
        tsmax=(np.where(on2_it[ien,:,:][mask_ok]>off2_it[ien,:,:][mask_ok],wst.ts,0)+2 *np.log(gw_prob[mask_ok])).max()
        tsmax_all[ien, iiter]=tsmax
    print(tsmax_all[:, iiter])

np.savez_compressed(f"out_freq_f{trueflux}_s{seed}", tsmax_all=tsmax_all, ras_true=ras_true, decs_true=decs_true)
