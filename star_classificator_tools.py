# -*- coding: utf-8 -*-
'''
Created on Jun 24 2020

@author: Everton Botan
@supervisor: Roberto Saito

This module classify RRLyrae based on period range, color range and double period test.
As it's unknown the color range (may be period too) on VIRCAN filters JKs 
we use a set (crossmatch) of RRLyrae from OGLE to determinate it.
'''

import os
import sys
import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.path.append('/home/botan/OneDrive/Doutorado/VVV_DATA/my_modules/')
import fit_sin_series as fitSin
import status

class StarClassificator(object):

    def __init__(self):
        self.path = f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/'
        self.tiles = sorted(os.listdir(f'{self.path}psf_ts/'))
        os.makedirs(f'{self.path}color_range',exist_ok=True)


    def _read_tables(self,tile,chip):
        self.data_table  = pd.read_csv(f'{self.path}psf_ts/{tile}/output/{chip}.ts',index_col=0,sep=',',low_memory=False)


    def _create_catalog(self,star_type,catalog):
        cat_data = []
        rrlyr_data = []
        i=1
        for tile in self.tiles:
            chips = [fn[:-3] for fn in sorted(os.listdir(f'{self.path}psf_ts/{tile}/output/')) if fn.endswith('ts')]
            j=1
            for chip in chips:
                status._print(prefix=f'Creating catalog {tile}/{chip}',iter1=i,length1=len(self.tiles),iter2=j,length2=len(chips),sufix='%')
                self._read_tables(tile,chip)
                self.data_table.loc[self.data_table.index, 'JKs_color'] = self.data_table['J'] - self.data_table['K']
                self.data_table.loc[self.data_table.index, 'tile'] = tile
                msk = (self.data_table[f'{catalog}_Type'] == star_type)
                rrlyr_data.append(self.data_table[msk])
                cat_data.append(self.data_table[['J','JERR','K','KERR','JKs_color','tile','chip','OGLE_ID','OGLE_Type','OGLE_Subtype','OGLE_P']])
                j+=1
            i+=1
        rrlyr_data = pd.concat(rrlyr_data)
        cat_data   = pd.concat(cat_data)
        return rrlyr_data, cat_data


    def get_period_ranges(self,star_type='RRLyr',catalog='OGLE', exists=True, plot=True, savefile=True):
        ''' 
        you can use this function to estimate color ranges from a set of stars matched with OGLE
        star_type must be a string that must be in the table columns
        catalog must be 'OGLE' or 'VSX (developping)'
        '''
        fpath = f'{self.path}color_range/{catalog}_{star_type}_data.csv'
        catpath = f'{self.path}color_range/color_table.csv'
        if exists:
            if os.path.exists(fpath) and os.path.exists(catpath):
                rrlyr_data = pd.read_csv(fpath,index_col=0,sep=',',low_memory=False)
                cat_data = pd.read_csv(catpath,index_col=0,sep=',',low_memory=False)
            else:
                rrlyr_data, cat_data = self._create_catalog(star_type,catalog)
                rrlyr_data.to_csv(fpath,sep=',')
                cat_data.to_csv(catpath,sep=',')
        else:
            rrlyr_data, cat_data = self._create_catalog(star_type,catalog)
            rrlyr_data.to_csv(fpath,sep=',')
            cat_data.to_csv(catpath,sep=',')
        if star_type == 'RRLyr':
            RRab_cat  = rrlyr_data[rrlyr_data[f'{catalog}_Subtype'] == 'RRab']
            RRab_pMin = np.nanpercentile(RRab_cat[f'{catalog}_P'],5)
            RRab_pMax = np.nanpercentile(RRab_cat[f'{catalog}_P'],95)
            RRc_cat   = rrlyr_data[rrlyr_data[f'{catalog}_Subtype'] == 'RRc']
            RRc_pMin = np.nanpercentile(RRc_cat[f'{catalog}_P'],5)
            RRc_pMax = np.nanpercentile(RRc_cat[f'{catalog}_P'],95)
            period_ranges = [[RRab_pMin],[RRab_pMax],[RRc_pMin],[RRc_pMax]]
            hdr = 'RRab_pMin,RRab_pMax,RRc_pMin,RRc_pMax'
            if savefile:
                np.savetxt(f'{self.path}color_range/{star_type}_period_ranges',np.array(period_ranges).T, delimiter=',',header=hdr)
            if plot:
                RRab_p = RRab_cat[f'{catalog}_P']
                RRc_p  = RRc_cat[f'{catalog}_P']
                plt.figure(figsize=[8,4])
                plt.hist(RRab_p,bins=15, label='RRab',histtype='step',color='b')
                plt.hist(RRc_p,bins=15, label='RRc',histtype='step',color='r')
                plt.axvline(RRab_p.median(), color='b', linestyle='dashed', linewidth=1,label='median: %.2f'%(RRab_p.median()))
                plt.axvline(RRc_p.median(),  color='r', linestyle='dashed', linewidth=1,label='median: %.2f'%(RRc_p.median()))
                plt.xlabel('Period [days]')
                plt.ylabel('# stars')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'{self.path}color_range/{catalog}_{star_type}_period_hist.png',dpi=300)
                #plt.show()
                plt.close
        return period_ranges


    def get_color_ranges(self,tile,star_type='RRLyr',catalog='OGLE', exists=True, plot=True,savefile=True):
        ''' 
        you can use this function to estimate color ranges from a set of stars matched with OGLE
        star_type must be a string that must be in the table columns
        catalog must be 'OGLE' or 'VSX (developping)'
        '''
        fpath = f'{self.path}color_range/{catalog}_{star_type}_data.csv'
        catpath = f'{self.path}color_range/color_table.csv'
        if exists:
            if os.path.exists(fpath) and os.path.exists(catpath):
                rrlyr_data = pd.read_csv(fpath,index_col=0,sep=',',low_memory=False)
                cat_data = pd.read_csv(catpath,index_col=0,sep=',',low_memory=False)
            else:
                rrlyr_data, cat_data = self._create_catalog(star_type,catalog)
                rrlyr_data.to_csv(fpath,sep=',')
                cat_data.to_csv(catpath,sep=',')
        else:
            rrlyr_data, cat_data = self._create_catalog(star_type,catalog)
            rrlyr_data.to_csv(fpath,sep=',')
            cat_data.to_csv(catpath,sep=',')
        if star_type == 'RRLyr':
            RRab_cat  = rrlyr_data[rrlyr_data[f'{catalog}_Subtype'] == 'RRab']
            RRc_cat   = rrlyr_data[rrlyr_data[f'{catalog}_Subtype'] == 'RRc']
            tile_cut  = cat_data["tile"] == tile
            x = cat_data[tile_cut]['JKs_color']
            y = cat_data[tile_cut]['K']
            tile_cut  = RRab_cat["tile"] == tile
            RRab_JKs  = RRab_cat[tile_cut]['JKs_color']
            RRab_Ks   = RRab_cat[tile_cut]['K']
            tile_cut  = RRc_cat["tile"] == tile
            RRc_JKs   = RRc_cat[tile_cut]['JKs_color']
            RRc_Ks    = RRc_cat[tile_cut]['K']
            RRab_cMin = np.nanpercentile(RRab_JKs,5)
            RRab_cMax = np.nanpercentile(RRab_JKs,95)
            RRc_cMin  = np.nanpercentile(RRc_JKs,5)
            RRc_cMax  = np.nanpercentile(RRc_JKs,95)
            color_ranges = [[RRab_cMin],[RRab_cMax],[RRc_cMin],[RRc_cMax]]
            hdr = 'RRab_cMin,RRab_cMax,RRc_cMin,RRc_cMax'
            if savefile:
                np.savetxt(f'{self.path}color_range/{star_type}_{tile}_color_ranges',np.array(color_ranges).T, delimiter=',',header=hdr)
            if plot:
                msk = ~x.isna()
                ymin = y[msk].min()
                ymax = y[msk].max()
                xmin = np.nanpercentile(x,.01)
                xmax = np.nanpercentile(x,99.98)
                plt.figure(figsize=[12,10])
                plt.hist2d(x[msk],y[msk],bins=1000,norm=mpl.colors.LogNorm(),cmap=mpl.cm.copper,alpha=.2,lw=1)
                cbar = plt.colorbar(aspect=40,pad=.01)
                cbar.set_label('# stars')
                plt.scatter(RRab_JKs,RRab_Ks, s=4, c='b', label=f'OGLE RRab ({len(RRab_JKs)} stars)')
                plt.vlines(x=RRab_cMin,ymin=ymin,ymax=ymax,colors='b',lw=.4)
                plt.vlines(x=RRab_cMax,ymin=ymin,ymax=ymax,colors='b',lw=.4)
                plt.scatter(RRc_JKs,RRc_Ks, s=4, c='r', label=f'OGLE RRc ({len(RRc_JKs)} stars)')
                plt.vlines(x=RRc_cMin,ymin=ymin,ymax=ymax,colors='r',lw=.4)
                plt.vlines(x=RRc_cMax,ymin=ymin,ymax=ymax,colors='r',lw=.4)
                plt.gca().invert_yaxis()
                plt.xlabel('J - Ks')
                plt.ylabel('Ks [mag]')
                plt.legend()
                plt.xlim(xmin,xmax)
                plt.ylim(20,9)
                plt.title(f'Hess diagram for tile {tile}',pad=5)
                plt.tight_layout(rect=[0, 0, 1, 0.98])
                plt.savefig(f'{self.path}color_range/{catalog}_{star_type}_{tile}_color_diagram.png',dpi=300)
                #plt.show()
                plt.close()
        return color_ranges


    def freq_cut(self,f,fMin,fMax):
        msk = ((f>fMin) & (f<fMax))
        index_val = np.concatenate(np.argwhere(msk.values==True))
        return index_val
    

    def color_cut(self,c,cMin,cMax):
        msk = ((c>cMin) & (c<cMax))
        index_val = np.concatenate(np.argwhere(msk.values==True))
        return index_val


    def _phase(self,P,t):
        phi = (t - min(t))/(P)
        phase = phi - (phi).astype(int)
        return phase


    def _phase_shift(self,t,lc,lcErr,freq):
        phase = self._phase(1./freq,t)
        xdata = np.concatenate([phase,phase+1])  
        order = np.argsort(xdata)
        ydata_err = np.concatenate([lcErr,lcErr])
        ydata = np.concatenate([lc,lc])
        xdata = xdata[order]
        ydata = ydata[order]
        ydata_err = ydata_err[order]
        return xdata,ydata,ydata_err


    def ECL_classificator(self, t, lc, lcerr, freq, phaseSpace=True, show=False):
        ''' 
        Separates nonsymmetric binaries from sample. 
        '''

        fit1     = fitSin.FitSinSeries(phaseSpace=phaseSpace,fitFreq=False)
        fitdat1  = fit1.fit_sinusoid_N_order(t,lc,lcerr,freq,order=5)

        fit2     = fitSin.FitSinSeries(phaseSpace=phaseSpace,fitFreq=False)
        fitdat2  = fit2.fit_sinusoid_N_order(t,lc,lcerr,freq/2.,order=5)

        if ((fitdat1 != 0) and (fitdat2 !=0)):
            times1         = fitdat1['magseries']['times']
            phase1         = fitdat1['magseries']['phase']
            mags1          = fitdat1['magseries']['mags']
            errs1          = fitdat1['magseries']['errs']
            fitmags1       = fitdat1['magseries']['fitmags']
            res1           = fitdat1['magseries']['residuals']
            fitparams1     = fitdat1['fitparams']
            fitparamserrs1 = fitdat1['fitparamserrs']
            rSq1           = fitdat1['fitinfo']['R2']
            ress1    = np.sum(res1**2)
        
            times2         = fitdat2['magseries']['times']
            phase2         = fitdat2['magseries']['phase']
            mags2          = fitdat2['magseries']['mags']
            errs2          = fitdat2['magseries']['errs']
            fitmags2       = fitdat2['magseries']['fitmags']
            res2           = fitdat2['magseries']['residuals']
            fitparams2     = fitdat2['fitparams']
            fitparamserrs2 = fitdat2['fitparamserrs']
            fitmag         = fitdat2['fitinfo']['fitmag']
            rSq2           = fitdat2['fitinfo']['R2']

            ress2    = np.sum(res2**2)
            #return indices of peaks
            peaks, _ = find_peaks(fitmags2, height=np.mean(fitmags2))
            if len(peaks) == 2:
                peakratio = abs(fitmags2[peaks[0]] - fitmags2[peaks[1]])
            res_std   = np.std(res2)

            if show:
                if ((fitdat1 != 0) and (fitdat2 !=0)):
                    fig, ax = plt.subplots(2,2,figsize=[12,6])
                    ax[0,0].plot(phase1,mags1      ,'.',c='C3',ms=2)
                    ax[0,0].plot(phase1+1,mags1    ,'.',c='C3',ms=2)
                    ax[0,0].plot(phase1,fitmags1   ,'-',c='C0')
                    ax[0,0].plot(phase1+1,fitmags1 ,'-',c='C0')
                    ax[0,0].invert_yaxis()
                    ax[0,0].set_xlabel(f'Phase ($Res^2$ = {ress1})')
                    ax[0,0].set_ylabel(f'Ks mag')
                    ax[0,0].text(0.05,0.9, f'Period: {round(1./freq,5)}',c='k',transform=ax[0,0].transAxes)
                    
                    ax[0,1].plot(phase1,res1,'.',c='C3',ms=2)
                    ax[0,1].invert_yaxis()
                    ax[0,1].set_ylim(-0.3,0.3)
                    ax[0,1].set_ylabel(f'Residual')
                    ax[0,1].set_xlabel(f'Phase')
                    
                    ax[1,0].plot(phase2,mags2   ,'.',c='C3',ms=2)
                    ax[1,0].plot(phase2,fitmags2,'-',c='C0')
                    ax[1,0].plot(phase2[peaks],fitmags2[peaks],'o',c='k')
                    ax[1,0].axhline(y=np.mean(fitmags2), color='k', linestyle='--')

                    ax[1,0].invert_yaxis()
                    ax[1,0].set_xlabel(f'Phase ($Res^2$ = {ress2})')
                    ax[1,0].set_ylabel(f'Ks mag')
                    ax[1,0].text(0.05,0.9, f'Period: {round(2./freq,5)} ** |pk1 - pk2| = {round(peakratio,5)} ** residual std: {round(res_std,5)}',c='k',transform=ax[1,0].transAxes)

                    ax[1,1].plot(phase2,res2,'.',c='C3',ms=2)
                    ax[1,1].invert_yaxis()
                    ax[1,1].set_ylim(-0.3,0.3)
                    ax[1,1].set_ylabel(f'Residual')
                    ax[1,1].set_xlabel(f'Phase')
                    
                    if peakratio > res_std:
                        startype = 'ECL'
                    else:
                        startype = 'UNK'

                    #if ress1/ress2 > 1.0:
                    #    startype = 'ECL'
                    #else:
                    #    startype = 'RRLyr'
                    plt.suptitle(f'Phase Space: Ress1/Ress2 = {round(ress1/ress2,5)} | Startype: {startype}')
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.savefig(f'figtest/{int(freq*10000)}.png')
                    #plt.show()
                    plt.close()
        
            if ((fitdat1 != 0) and (fitdat2 !=0)):
                if len(peaks) == 2:
                    if peakratio > res_std:
                        star_type = 'ECL'
                    else:
                        star_type = 'UNC'
                else:
                    star_type = 'UNC'
        else:
            star_type = 'UNC'
        return star_type


    def RRLyr_subtypes(self, amplitude, best_freq):
        
        pass

    def _RRLyr_subtype(self,freq,RRab_pmin,RRab_pmax):
        'amplitude vs period plot'
        if ((1./freq > RRab_pmin) and (1./freq < RRab_pmax)):
            starsubtype = 'RRab'
        else:
            starsubtype = 'RRc'
        return starsubtype


if __name__ == '__main__':
    path = f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/'
    tiles = sorted(os.listdir(f'{path}psf_ts/'))
    c = StarClassificator()

    def _phase(P,t):
        phi = (t - min(t))/(P)
        phase = phi - (phi).astype(int)
        return phase

    def read_file(fname):
        lc_dat = np.genfromtxt("../test/LS_%s.lc"%fname,delimiter=" ",names=True)      
        ls_dat = np.genfromtxt("../test/LS_%s.ls"%fname,delimiter=" ",names=True)  
        mag = lc_dat["Ks_mag"]  
        err= lc_dat["Ks_err"] 
        t = lc_dat["mjd"] 
        best_freq = ls_dat["freq"][np.argwhere(ls_dat["power"] == ls_dat["power"].max())[0]][0] 

        phase = _phase(1./best_freq,t)
        
        return t, mag, err, best_freq, phase

    stars = [fn[3:-3] for fn in os.listdir("../test/") if fn.endswith("ls")]
    for fname in stars[:]:
        t, lc, lcErr, best_freq, phase = read_file(fname)
        c.ECL_classificator(t, lc, lcErr, best_freq, phaseSpace=True,show=True)
        #c.ECL_classificator(t, lc, lcErr, best_freq, phaseSpace=False,show=True)



    #c.get_period_ranges(star_type='RRLyr',catalog='OGLE', exists=True, plot=True, savefile=True)
    #for tile in tiles:
    #    c.get_color_ranges(tile=tile, star_type='RRLyr',catalog='OGLE', exists=True, plot=True, savefile=True)
    