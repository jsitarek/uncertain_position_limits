import numpy as np
import numexpr as ne
import astropy.units as u
import gammapy.irf
import glob
from gammapy.stats import WStatCountsStatistic
from astropy.coordinates import SkyCoord
from astropy.coordinates import angular_separation
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64
import scipy
from scipy.stats import norm
import sys

_, infile, seed, trueflux, niter=sys.argv
seed=int(seed)
trueflux=float (trueflux)
niter=int(niter)

# those settings must agree with those used in the file loaded below
irffile='IRF/Prod5-North-20deg-AverageAz-4LSTs.1800s-v0.1.fits'
file=np.load(infile, allow_pickle=True) # 'gw_setup_0.07deg.npz'
print(list(file.keys()))

# those settings can be modified here
#trueflux=0.1
#seed=51
bgdalpha=5
cl=0.95

mineest=0
maxeest=10

pointings = file['pointings']
angleoffset=file['angleoffset'].item()
timeperpoint=file['timeperpoint'].item()
binsize=file['binsize'].item()
ras=file['ras']
decs=file['decs']
ras0=file['ras0']
decs0=file['decs0']
ras1=file['ras1']
decs1=file['decs1']
mypixarea=file['mypixarea']
etrue=file['etrue']
etruehi=file['etruehi']
etruelo=file['etruelo']
eest=file['eest']
eesthi=file['eesthi']
eestlo=file['eestlo']
mask_ok=file['mask_ok']
mask_point=file['mask_point']
gw_prob=file['gw_prob']
#pos_mig=file['pos_mig']
offset_mig=file['offset_mig']
emigmy=file['emigmy']
aeff_eest_nocut=file['aeff_eest_nocut']
aeff_eest_cut=file['aeff_eest_cut']
aeff_etrue=file['aeff_etrue']
bgd=file['bgd']
aeff_mig_eest=file['aeff_mig_eest']
psf_avr=file['psf_avr']*u.Unit('deg-2') # stored without a unit
thetabins=file['thetabins']

kernels=file['kernels']
kernelhalfsize=file['kernelhalfsize']

binsdec=len(decs0)
binsra=len(ras0)
print(f"{binsdec} bins in DEC, {binsra} bins in RA")

# for simulations the code uses Aeff directly from the IRFs
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

print(seed)
rng = Generator(PCG64(seed=seed))

# check which of the simulated signals are inside the ROI
def check_in_roi(ras_true, decs_true):
    rabin=np.digitize (x=ras_true, bins=ras)-1
    decbin=np.digitize (x=decs_true, bins=decs)-1
    print(rabin.max())
    print(decbin.max())
    mask_ok2=np.zeros((len(decs0)+1, len(ras0)+1), dtype=bool)
    mask_ok2[:-1,:-1]=mask_ok # adding extra overflow bin
    inroi=mask_ok2[decbin, rabin]
    print(inroi.sum(axis=-1).mean())
    return inroi


truefluxes=[]
decs_true=[]
ras_true=[]
limits_agnostic=[]
limits_bayes=[]
fluxes_bayes=[]
fluxes_bayes_ep=[]
fluxes_bayes_em=[]
limits_freq=[]

#maxeest=len(eest)


truefluxes='0.0 '+' '.join(map(str,np.logspace(-2., 1.,60)))
dirin='../juan_bgd/'
truefluxes=[a.split('_')[-2][1:] for a in glob.glob(dirin+'/out_freq_f*_s*.npz')]
truefluxes=sorted(truefluxes, key=lambda s: float(s))

tsmax_all=[]
tfs=[]
ras_true_f=[]
decs_true_f=[]
for tf in truefluxes: 
    file3=np.load(f"{dirin}/out_freq_f{tf}_s51.npz")
    tsmax_all+=[file3['tsmax_all']]
    ras_true_f+=[file3['ras_true']]
    decs_true_f+=[file3['decs_true']]
    tfs+=[float(tf)]
tsmax_all=np.array(tsmax_all)
tfs=np.array(tfs)
ras_true_f = np.array(ras_true_f)
decs_true_f = np.array (decs_true_f)
print(f"loaded frequentist files with shape: {tsmax_all.shape}")

inroi=check_in_roi(ras_true_f, decs_true_f)
inroi=np.transpose(inroi[..., np.newaxis],axes=(0,2,1))
nullmedians = np.median(tsmax_all[0,:,:], axis=-1)

for iiter in range (niter):
    excess_eest, dec_true, ra_true, imask_true = create_excess(pointings=pointings, emigmy=emigmy, trueflux=trueflux, irfs=irfs)
    truefluxes+=[trueflux]
    decs_true+=[dec_true]
    ras_true+=[ra_true]
    print(f"generate {iiter} with flux {trueflux} at position {dec_true}, {ra_true}")
    on, off, on2, off2 = create_on_off (bgd=bgd, excess_eest=excess_eest, kernels=kernels)

    ## agnostic 
    limit_wstat=np.zeros(on2.shape)
    for ien in range(mineest, maxeest):
        shp=np.ones(on2[ien,:,:].shape)[mask_ok]
        wst=WStatCountsStatistic (n_on=on2[ien,:,:][mask_ok], n_off=off2[ien,:,:][mask_ok]*bgdalpha, alpha=1./bgdalpha*shp, mu_sig=0*shp) 
        ts=wst.ts
        limit_wstat[ien,:,:][mask_ok]=wst.compute_upper_limit(norm.ppf(q=cl))
        print(f"Eest={eest[ien]}, maxts = {ts.max()}, max limit={limit_wstat[ien,:,:][mask_ok].max()}")    
    limit_agnostic=(limit_wstat[:maxeest,mask_ok]/aeff_eest_cut[:maxeest,mask_ok]).max(axis=-1)
    limits_agnostic+=[limit_agnostic]
    
    print(limit_agnostic)

    ## bayesian

    def calcBest(ON, OFF, NPOS, S):
        #delta = (S*(1+NPOS) - ON - OFF)**2 + 4 * (NPOS+1) * S * OFF
        delta = ne.evaluate('(S*(1+NPOS) - ON - OFF)**2 + 4 * (NPOS+1) * S * OFF')
        #B = (-(S*(1+NPOS) - ON - OFF) + np.sqrt(delta)) / (2*(NPOS+1))
        B= ne.evaluate('(-(S*(1+NPOS) - ON - OFF) + sqrt(delta)) / (2*(NPOS+1))')
        return B;
    def ln_p_d_true(on, off, pos0, f0, aeff_mig_eest, gw_prob):
        n0=ne.evaluate('aeff_mig_eest*f0')
        B=calcBest(on, off* bgdalpha, bgdalpha, n0) 
        poff=scipy.stats.poisson.pmf(k=np.round(ne.evaluate('off*bgdalpha')).astype('int'), mu=ne.evaluate('B*bgdalpha')) # slow!
        pon=scipy.stats.poisson.pmf(k=np.round(on).astype('int'), mu=ne.evaluate('B+n0')) # slow!
        return ne.evaluate('sum(log(pon)+log(poff),axis=2)')+gw_prob[pos0[:,:, 0]]
        #return (np.log(pon)+np.log(poff)).sum(axis=-1)+np.log(gw_prob[pos0[:,:, 0]])

    def bayes_scan(on, off, aeff_mig_eest, gw_prob, fmax, fmin=0, nf=50):
        fs=np.linspace(fmin, fmax, nf)
        pos=np.arange(on.shape[-1])
        # axis = [f0, pos0, bin excess]
        lnp=ln_p_d_true(on=on[np.newaxis, np.newaxis,...], off=off[np.newaxis, np.newaxis,...], pos0=pos[np.newaxis, ..., np.newaxis], f0=fs[..., np.newaxis, np.newaxis], aeff_mig_eest=aeff_mig_eest[np.newaxis, ...], gw_prob=gw_prob)
        return fs, lnp

    limit_bayes=np.full(eest.shape, np.inf)
    prob_norms=[]
    for ien in range (mineest,maxeest): 
        print ("bayes, eest bin: ", ien)
        fs, lnp=bayes_scan(on[ien, mask_ok], off[ien, mask_ok], aeff_mig_eest[ien, :,mask_ok[mask_point]], gw_prob[mask_ok], fmax=1.5)
        prob_norm=np.exp(lnp-lnp.max())/(np.exp(lnp-lnp.max())).sum()
        prob_norms+=[prob_norm]
    prob_norms=np.array(prob_norms)
    
    def quant_flux(fs, prob_2d):
        prob_sum=prob_2d.sum(axis=-1) # sum over position
        cdf = prob_sum.cumsum(axis=-1)

        qs=[0.5, (1-0.68)/2, 1-(1-0.68)/2, cl]
        median, valmin, valmax, ul=np.interp(qs, cdf, fs)
        return median, median-valmin, valmax-median, ul
    bayes_med=[]
    bayes_errm=[]
    bayes_errp=[]
    bayes_ul=[]
    bayes_e=[]
    for i in range (prob_norms.shape[0]):
        median, errm, errp, ul = quant_flux(fs, prob_norms[i,:,:])
        bayes_med+=[median]
        bayes_errm+=[errm]
        bayes_errp+=[errp]
        bayes_ul+=[ul]
        bayes_e+=[eest[i]]
        print(f"E={eest[i]:.3f}, f = {median:.3f}+{errp:.3f}-{errm:.3f}, < {ul:.3f}")

    limits_bayes+=[bayes_ul]
    fluxes_bayes+=[bayes_med]
    fluxes_bayes_ep+=[bayes_errp]
    fluxes_bayes_em+=[bayes_errm]

    ## frequentist method
    tsmax0=np.zeros(eest.shape)
    for ien in range(len(eest)):
        shp=np.ones(on2[ien,:,:].shape)[mask_ok]
        wst=WStatCountsStatistic (n_on=on2[ien,:,:][mask_ok], n_off=off2[ien,:,:][mask_ok]*bgdalpha, alpha=1./bgdalpha*shp, mu_sig=0*shp) 
        tsmax0[ien]=(np.where(on2[ien,:,:][mask_ok]>off2[ien,:,:][mask_ok],wst.ts,0)+2 *np.log(gw_prob[mask_ok])).max()
        print(f"Eest={eest[ien]}, maxts = {tsmax0[ien]}")
    
    print(f"substituting TS for median value in bins with Eest = {eest[tsmax0<=nullmedians]}")
    tsmax0[tsmax0<nullmedians]=nullmedians[tsmax0<nullmedians]
    tsmax_prob=((tsmax_all>tsmax0[np.newaxis, ..., np.newaxis])*inroi).sum(axis=-1)/inroi.sum(axis=-1) #tsmax_all.shape[-1]

    limit_freq=np.full(maxeest, np.inf) # prefilled with infinity
    def ffit(x, a, b, c, d, e):
        return a+b*x+c*x**2+d*x**3+e*x**4
    for ien in range(mineest,maxeest):
        if (tsmax_prob[:,ien]<cl).sum()<2 or (tsmax_prob[:,ien]>cl).sum()<2:
            continue
        ipoint=np.argmax(-np.abs(tsmax_prob[:,ien] - cl))
        i1=np.max([0,ipoint-5])
        i2=np.min([tsmax_prob.shape[0]-1, ipoint+5])
        x=tsmax_prob[i1:i2,ien]
        y=tfs[i1:i2]
        popt,_=scipy.optimize.curve_fit (ffit, x, y)
        limit_freq[ien]=ffit(cl, *popt)
    limits_freq+=[limit_freq]
    # print summary
    print(np.array(list(zip(eest[:maxeest], limit_agnostic, bayes_ul, limit_freq))))
    
truefluxes=np.array(truefluxes)
decs_true=np.array(decs_true)
ras_true=np.array(ras_true)
limits_agnostic=np.array(limits_agnostic)
limits_bayes=np.array(limits_bayes)
fluxes_bayes=np.array(fluxes_bayes)
fluxes_bayes_ep=np.array(fluxes_bayes_ep)
fluxes_bayes_em=np.array(fluxes_bayes_em)
limits_freq=np.array(limits_freq)
np.savez_compressed(f"out_limits_ebin{mineest}to{maxeest}_f{trueflux}_s{seed}",
                    limits_agnostic=limits_agnostic,
                    limits_bayes=limits_bayes,
                    fluxes_bayes=fluxes_bayes,
                    fluxes_bayes_ep=fluxes_bayes_ep,
                    fluxes_bayes_em=fluxes_bayes_em,
                    limits_freq=limits_freq,
                    ras_true=ras_true, decs_true=decs_true,
                    eest=eest[:maxeest], truefluxes=truefluxes)

