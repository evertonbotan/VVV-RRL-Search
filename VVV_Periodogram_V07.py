# -*- coding: utf-8 -*-
'''
Created on Jul 28 2019
Last large update on Jun 22 2020

@author: Everton Botan
@supervisor: Roberto Saito

It's (will be) a Python3.6 or higher program that perform massive search and classification of variable stars into VVV data.
'''
import os
import sys
import math
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import Table
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
from scipy import optimize, signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
import time

sys.path.append('/home/botan/OneDrive/Doutorado/VVV_DATA/my_modules/')
import clean_match_tables as cmt
import fit_sin_series as fitSin
import periodogram as pg
import variability_indicator as vi
import star_classificator_tools as sct
import status


class PeriodSearch(object):
    def __init__(self, path, tile, minP, maxP, varIndex='chi2'):
        '''
        tile: tile name as b293 it deppends of your folder architecture.
        minP: minimum period (float)
        maxP: maximum period (float)
        varIndex: uncorrelated variable index: std or chi2
        '''
        self.tile = tile
        self.varIndex = varIndex
        self.minP = minP
        self.maxP = maxP
        self.path = f'{path}/{tile}'
        os.makedirs(f'{self.path}/figures',exist_ok=True)
        os.makedirs(f'{self.path}/output',exist_ok=True)
        self.chips = [fn[:-3] for fn in sorted(os.listdir(f'{self.path}/chips/')) if fn.endswith('ts')]
        self.tiles = sorted(os.listdir('/home/botan/OneDrive/Doutorado/VVV_DATA/data/psf_ts/'))


    def organize_tables(self):
        org = cmt.CreateTable(path=self.path,
                              tile=self.tile,
                              min_sample=25,
                              raw_files=True)
        #org.plot_chips(show=False)


    def select_candidates(self):
        select = vi.Variability(path=self.path,
                                tile=self.tile,
                                method=self.varIndex,
                                maxMag=11.5,
                                stdRatio=1.5,
                                minChi2=2,
                                savePlot=True)
        select.do_selection()


    def _read_tables(self,chip):
        self.data_table  = pd.read_csv(f'{self.path}/chips/{chip}.ts',index_col=0,sep=',',low_memory=False)
        self.obs_time    = np.loadtxt(f'{self.path}/chips/{chip}.mjd')
        self.mag_cols    = [col for col in self.data_table.columns if col.split('_')[0] == 'MAG']
        self.err_cols    = [col for col in self.data_table.columns if col.split('_')[0] == 'ERR']
        self.color_cols  = [col for col in self.data_table.columns if col.split('_')[0] == 'mag']
        self.ks_mag      = self.data_table[self.mag_cols]
        self.ks_err      = self.data_table[self.err_cols]
        self.star_ids    = self.data_table.index
        if self.varIndex == 'chi2':
            self.candidates = np.genfromtxt(f'{self.path}/var_data/{chip}.chi2cand', dtype=str)
        if self.varIndex == 'std':
            self.candidates = np.genfromtxt(f'{self.path}/var_data/{chip}.stdcand', dtype=str)


    def _freq_agreement(self,f1,f2,h=4): 
        '''check if PDM and  LSG frequencies are in agreement till 4 harmonics'''
        n = 1 
        while n <= h: 
            m = 1 
            while m <= h: 
                if abs(f1/f2 - n/m) < 0.01: 
                    bol = True 
                    break 
                else: 
                    bol = False 
                m+=1 
            if bol: 
                break 
            n+=1 
        return bol 


    def do_periodogram(self, exists=True):
        j=1
        for chip in self.chips:
            if exists:
                chips_done = [fn[:-19] for fn in os.listdir(f'{self.path}/var_data/') if fn.endswith('pdm_parameters.csv')]
            else:
                chips_done = []
            if not chip in chips_done:
                self._read_tables(chip)
                lsg_pgram_params = [] 
                pdm_pgram_params = [] 
                i=1
                for star in self.candidates:
                    status._print(prefix=f'Periodogram of chip {chip}',
                                  iter1=j,
                                  length1=len(self.chips),
                                  iter2=i,
                                  length2=len(self.candidates), 
                                  sufix='%')
                    lc =  self.ks_mag.loc[star].values
                    err =  self.ks_err.loc[star].values
                    t = self.obs_time
                    pgram = pg.Periodogram(t, lc, err, self.minP, self.maxP, 
                                            normalization='psd',
                                            method='scargle',
                                            samples_per_peak=10,
                                            false=0.001,
                                            nbins=10,
                                            covers=3,
                                            mode=False)
                    lsg_freq, lsg_power, lsg_false_alarm, lsg_best_freq, lsg_fap, lsg_sig_level, lsg_all_freq = pgram.LSG()
                    #lomg Scargle is much faster than PDM, so, PDM stars only if LSG identify a true Frequency
                    if lsg_best_freq > 0:
                        pdm_freq, pdm_theta, pdm_best_freq, pdm_fap, pdm_sig_level, pdm_all_freq = pgram.CyPDM()
                        if pdm_best_freq > 0:
                            # comparison with PDM period (inside 1% and harmonics until n = 4):
                            if self._freq_agreement(f1=lsg_best_freq, f2=pdm_best_freq, h=4):
                                #coords:
                                ra    = self.data_table.loc[star]['RA']
                                dec   = self.data_table.loc[star]['DEC']
                                # J and Ks aper mag for color classification
                                j_mag = self.data_table.loc[star]['J']
                                j_err = self.data_table.loc[star]['JERR']
                                k_mag = self.data_table.loc[star]['K']
                                k_err = self.data_table.loc[star]['KERR']
                                EJK = self.data_table.loc[star]['EJK']
                                EJKERR = self.data_table.loc[star]['EJKERR']
                                # ZYJHKs psf mag for color classification
                                color_cols = [col for col in self.data_table.columns if col.split('_')[0] == 'mag' or col.split('_')[0] == 'er']
                                color_vals = [self.data_table.loc[star][col] for col in color_cols]
                                # amplitude from light curve (95 - 5 percentile)
                                amplitude = np.nanpercentile(lc,q=95) - np.nanpercentile(lc,q=5)
                                lsg_params   = [star, chip, ra, dec] + color_vals + [j_mag, j_err, 
                                                k_mag, k_err, EJK, EJKERR, lsg_best_freq, amplitude,
                                                lsg_fap, lsg_sig_level]
                                pdm_params   = [star, chip, ra, dec] + color_vals + [j_mag, j_err, 
                                                k_mag, k_err, EJK, EJKERR, pdm_best_freq, amplitude, 
                                                pdm_fap, pdm_sig_level]
                                lsg_pgram_params.append(lsg_params)
                                pdm_pgram_params.append(pdm_params)
                    i+=1
                # save periodogram data to files
                colnames = ['ID','chip','RA','DEC'] + color_cols + ['APER_J',
                            'APER_JERR', 'APER_K','APER_KERR','APER_EJK', 
                            'APER_EJKERR','best_freq','amplitude','fap','sig_level']
                lsg_pgram_params = pd.DataFrame(lsg_pgram_params, columns=colnames)
                lsg_pgram_params.set_index('ID',inplace=True)
                lsg_pgram_params.to_csv(f'{self.path}/output/{chip}_lsg_parameters.csv',sep=',')
                pdm_pgram_params = pd.DataFrame(pdm_pgram_params, columns=colnames)
                pdm_pgram_params.set_index('ID',inplace=True)
                pdm_pgram_params.to_csv(f'{self.path}/output/{chip}_pdm_parameters.csv',sep=',')
                j+=1
            else:
                j+=1



    def get_color_range(self,tile,star_type='RRLyr',catalog='OGLE'):
        c = sct.StarClassificator()
        color_range = c.get_color_ranges(tile=tile,
                                         star_type=star_type,
                                         catalog=catalog, 
                                         exists=True, 
                                         plot=True, 
                                         savefile=True)
        return color_range


    def get_period_range(self,star_type='RRLyr',catalog='OGLE'):
        p = sct.StarClassificator()
        period_range = p.get_period_ranges(star_type=star_type,
                                           catalog=catalog, 
                                           exists=True, 
                                           plot=True, 
                                           savefile=True)
        return period_range
    

    def _find_mean_sig(self,x,xpos,sig):
        ''' for use in failure modes '''
        mean = np.median(x[(x > xpos - sig) & (x < xpos + sig)])
        newsig = np.std(x[(x > mean - sig) & (x < mean + sig)])
        return mean, newsig


    def failure_modes(self, freq=1.0, sigfactor=1, plot=True):
        all_chips_params = []
        for chip in self.chips:
            lsg_params = pd.read_csv(f'{self.path}/var_data/{chip}_lsg_parameters.csv',sep=',',index_col='ID')
            all_chips_params.append(lsg_params)
        all_chips_params = pd.concat(all_chips_params)
        gsig = 0.1
        x = all_chips_params['best_freq']
        y = all_chips_params['amplitude']
        bad_ids = []
        alias = []
        xbar = []
        height = []
        xpos = 1
        while xpos < x.max():
            mean,sig = self._find_mean_sig(x,xpos,gsig)
            #print(xpos, mean, sig)
            alias.append((mean,sig))
            s = all_chips_params.index[((x > mean - sig*sigfactor) & (x < mean + sig*sigfactor))]
            #print('BadIds:', len(s), 'for freq: ', mean)
            xbar.append(xpos)
            height.append(len(s))
            bad_ids.append(s)
            xpos+=1
        bad_ids = np.concatenate(bad_ids)
        good_ids = [n for n in all_chips_params.index if n not in bad_ids]
        all_chips_params.loc[good_ids].to_csv(f'{self.path}/var_data/{self.tile}_lsg_aliasless_parameters.csv',sep=',')
        all_chips_params.loc[bad_ids].to_csv(f'{self.path}/var_data/{self.tile}_lsg_alias_parameters.csv',sep=',')
        #print('Selected: ',len(good_ids),' Aliases: ',len(bad_ids),' Total: ',len(y))
        if plot:
            textstr = '\n'.join((f'Total: {len(y)}',
                                f'Aliases: {len(bad_ids)}'))
            props = dict(boxstyle='square', facecolor='w', alpha=0.3)

            fig, ax = plt.subplots(figsize=[6,3],tight_layout=True)
            ax.scatter(x.loc[bad_ids],y.loc[bad_ids],marker='.',s=1,c='r',alpha=.4)
            ax.scatter(x.loc[good_ids],y.loc[good_ids],marker='.',s=1,c='k',alpha=.4)
            ax.text(0.05, 0.95, textstr,transform=ax.transAxes, fontsize=11,verticalalignment='top', bbox=props)
            ax.set_xlabel('Frequency [$days^{-1}$]')
            ax.set_ylabel('Amplitude [Ks mag]')
            ax.set_ylim(0,1.5)
            ax.set_xlim(-0.2,10.2)
            plt.tight_layout()
            plt.savefig(f'{self.path}/var_data/ampxfreq.png',dpi=300)
            #plt.show()
            plt.close()

            plt.figure(figsize=[7,4])
            plt.bar(xbar,height,width=0.9)
            plt.xlabel('Frequency [$day^{-1}$]')
            plt.ylabel('Counts')
            plt.title('Number of stars by frequency aliases')
            plt.tight_layout()
            plt.savefig(f'{self.path}/var_data/barplot_aliases.png',dpi=300)
            #plt.show()
            plt.close()
        return good_ids, bad_ids




    def _phase(self,P,t):
        phi = (t - min(t))/(P)
        phase = phi - (phi).astype(int)
        return phase


    def _center_curve(self,fitmags,phase,mags,errs,anchor=0.25):
        '''
        It Shifts the phased light curve to align the min mag value (valley) to anchor for better visualization.
        '''
        min_arg = fitmags.argmax()
        phi = phase[min_arg]
        phase_shift = phase - phi
        phase_shift[phase_shift < 0 ] += 1
        phase_shift = phase_shift + anchor
        phase_shift[phase_shift > 1 ] -= 1
        return phase_shift


    def periodogram_plot(self,star,fitargs,pmin,pmax,fig_path,plotfit=True,show=False):
        t,lc,lcErr,freq,order,phaseSpace,fitFreq,iterative,fitastrobase = fitargs
        # fit with P
        fitdat1 = self._curve_fit_switch(t,lc,lcErr,freq,order,phaseSpace,fitFreq,iterative,fitastrobase)
        fitdat2 = self._curve_fit_switch(t,lc,lcErr,freq/2.,order,phaseSpace,fitFreq,iterative,fitastrobase)
        if ((fitdat1 != 0) and (fitdat2 != 0)):
            times1   = fitdat1['magseries']['times']
            phase1   = fitdat1['magseries']['phase']
            mags1    = fitdat1['magseries']['mags']
            errs1    = fitdat1['magseries']['errs']
            fitmags1 = fitdat1['magseries']['fitmags']
            # fit with 2P
            
            times2   = fitdat2['magseries']['times']
            phase2   = fitdat2['magseries']['phase']
            mags2    = fitdat2['magseries']['mags']
            errs2    = fitdat2['magseries']['errs']
            fitmags2 = fitdat2['magseries']['fitmags']
            # Power spectrum
            pgram = pg.Periodogram(x=t, y=lc, yerr=lcErr, 
                                    minP=0.1,
                                    maxP=1000,
                                    normalization='psd',
                                    method='scargle',
                                    samples_per_peak=10,
                                    false=0.001,
                                    nbins=10,
                                    covers=3,
                                    mode=False)
            frequency, power, false_alarm, best_freq, fap, sig_level, all_freq = pgram.LSG()
            ms = 3
            fig, ax = plt.subplots(4,1,figsize=[6,9],gridspec_kw={'wspace':0, 'hspace':0.4})
            ax[0].set_title(f'ID: {star}')
            ax[0].errorbar(x=t,y=lc,yerr=lcErr,
                            capsize=2,elinewidth=0.8,fmt='.',
                            mec='k',mfc='k',ms=ms,ecolor='r')
            ax[0].invert_yaxis()
            ax[0].set_xlabel('MJD [days]')
            ax[0].set_ylabel('Ks [mag]')

            ax[1].hlines(sig_level, xmin=frequency.max(), xmax=frequency.min(), ls='dashed', lw=.8, color='r')
            ax[1].text(s='FAP level 0.1%', x=(0.75/pmin), y=sig_level + sig_level*.02, color='r')
            ax[1].plot(frequency,power,'k-',lw=.5)
            ax[1].set_xlim(1./pmax, 1./pmin)
            ax[1].set_xlabel('frequency [$day^{-1}$]')
            ax[1].set_ylabel('LSG power')

            #plot phased lc with P:
            period = 1./freq
            if fitdat1 != 0:
                # centrzlize the phased light curve:
                phase_shift = self._center_curve(fitmags1,phase1,mags1,errs1,anchor=0.5)
                ax[2].plot(phase_shift  ,mags1,'.k',ms=ms)
                ax[2].plot(phase_shift+1,mags1,'.k',ms=ms)
                if plotfit:
                    ax[2].plot(phase_shift  ,fitmags1,'r.',ms=1)
                    ax[2].plot(phase_shift+1,fitmags1,'r.',ms=1)
            else:
                ax[2].plot(phase1,mags1  ,'.k',ms=ms)
                ax[2].plot(phase1+1,mags1,'.k',ms=ms)
            ax[2].invert_yaxis()
            ax[2].set_xlabel(f'Phase [Period: {round(period,5)} days]')
            ax[2].set_ylabel('Ks [mag]')

            #plot phased lc with 2P:
            period = 2./best_freq
            if fitdat2 != 0:
                # centrzlize the phased light curve:
                phase_shift = self._center_curve(fitmags2,phase2,mags2,errs2,anchor=0.25)
                ax[3].plot(phase_shift,mags2,'.k',ms=ms)
                if plotfit:
                    ax[3].plot(phase_shift,fitmags2,'r.',ms=1)
            else:
                ax[3].plot(phase2,mags2,'.k',ms=ms)
            ax[3].invert_yaxis()
            ax[3].set_xlabel(f'Phase [Period: {round(period,5)} days]')
            ax[3].set_ylabel('Ks [mag]')

            os.makedirs(fig_path,exist_ok=True)
            plt.savefig(f'{fig_path}/LSG_{star}.png',dpi=100,pad_inches=0.02)
            if show:
                plt.show()
            plt.close()


        
    def _read_periodogram_outputs(self,chip):
        self.lsg_params = pd.read_csv(f'{self.path}/var_data/{chip}_lsg_parameters.csv',sep=',',index_col='ID')
        self.pdm_params = pd.read_csv(f'{self.path}/var_data/{chip}_pdm_parameters.csv',sep=',',index_col='ID')


    def _RRLyr_subtype(self,freq,RRab_pmin,RRab_pmax):
        'amplitude vs period plot'
        if ((1./freq > RRab_pmin) and (1./freq < RRab_pmax)):
            starsubtype = 'RRab'
        else:
            starsubtype = 'RRc'
        return starsubtype


    def _stars_already_plotted(self):
        stars_done = ([_[4:-4] for _ in sorted(os.listdir(f'{self.path}/lc_plots/short_period/pos_visual_inspection/RRLyr')) if _[4:-4] != ''] +
                      [_[4:-4] for _ in sorted(os.listdir(f'{self.path}/lc_plots/short_period/pos_visual_inspection/ECL')) if _[4:-4] != ''] +
                      [_[4:-4] for _ in sorted(os.listdir(f'{self.path}/lc_plots/short_period/pos_visual_inspection/IDK')) if _[4:-4] != ''])
        return stars_done


    def _curve_fit_switch(self,t,lc,lcErr,freq,order,phaseSpace,fitFreq,iterative,fitastrobase):
        fit = fitSin.FitSinSeries(phaseSpace=phaseSpace,fitFreq=fitFreq)
        if fitastrobase:
            fitdat = fit.fit_sinusoid_astrobase(t,lc,lcErr,freq,order=order)
        else:
            if iterative:
                fitdat  = fit.fit_sinusoid_iterative(t,lc,lcErr,freq,order=order)
            else:
                fitdat  = fit.fit_sinusoid_N_order(t,lc,lcErr,freq,order=order)
        return fitdat




    def do_periodogram_plots(self,pmin,pmax,fpath,phaseSpace,fitFreq,order,iterative,fitastrobase):
        '''
        It does the first plot and automated classification for ECL by
        fitting two sinusoid series to each light curve and comparing the depth of
        the two valleys to standar deviation of the resifuals.
        If the depths difference is great than residual deviation it
        probabelly is a assimetrical ECL.
        '''
        print(f'Removing seasonal aliases from tile {self.tile}...')
        aliasfree_id, _ = self.failure_modes(freq=1.0, sigfactor=1, plot=True)
        ecl_fitparams = []
        unc_P_fitparams = []
        unc_2P_fitparams = []
        output_data = []
        i = 1
        for chip in self.chips:
            self._read_tables(chip)
            self._read_periodogram_outputs(chip)
            candidates = [star for star in self.lsg_params.index if star in aliasfree_id]
            #stars_done = self._stars_already_plotted()
            #candidates = [_ for _ in candidates if _ not in stars_done]
            j=1
            for star in candidates:
                status._print(prefix=f'Plotting chip {chip}',
                              iter1=i,
                              length1=len(self.chips),
                              iter2=j,
                              length2=len(candidates),
                              sufix='%')

                t = self.obs_time
                lc = self.ks_mag.loc[star].values
                lcErr = self.ks_err.loc[star].values
                ra        = self.data_table.RA.loc[star]
                dec       = self.data_table.DEC.loc[star]
                best_freq = self.lsg_params['best_freq'].loc[star]
                amplitude = self.lsg_params['amplitude'].loc[star]
                # ZYJHKs psf mag for color classification
                color_cols = [col for col in self.data_table.columns if col.split('_')[0] == 'mag' or col.split('_')[0] == 'er']
                color_vals = [self.data_table.loc[star][col] for col in color_cols]
                                
                #period select
                if ((1./best_freq > pmin) and (1./best_freq < pmax)):
                    # test for ECLs
                    c = sct.StarClassificator()
                    startype = c.ECL_classificator(t, lc, lcErr, best_freq, phaseSpace=True)
                    # ECL star type
                    if startype == 'ECL':
                        fitdat1  = self._curve_fit_switch(t,lc,lcErr,best_freq/2.,order,phaseSpace,fitFreq,iterative,fitastrobase)
                        fitparams = fitdat1['fitparams']
                        R2 = fitdat1['fitinfo']['R2']
                        chi2 = fitdat1['fitinfo']['Chi2']
                        res = fitdat1['magseries']['residuals']
                        errs = fitdat1['magseries']['errs']
                        meanRESS = np.sum(res**2)/(len(res)-1)
                        red_chi2 = np.sum(res**2 / errs**2)/(len(res)-1)

                        fit_parameters  = [star,ra,dec,best_freq,amplitude,R2,chi2,meanRESS,red_chi2] + color_vals
                        fit_params_cols = ['ID','RA','DEC','Freq','Amplitude','R2','Chi2','residual_sum','red_chi2'] + color_cols
                        n=0 
                        k=1
                        while n < len(fitparams[0]):
                            fitpar = fitdat1['fitparams']
                            fitparerrs = fitdat1['fitparamserrs']
                            fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                            fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                            n+=1
                            k+=1
                        ecl_fitparams.append(pd.Series(fit_parameters,index=fit_params_cols))
                        
                        figpath = f'{fpath}/{startype}'
                        fitargs = [t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase]
                        self.periodogram_plot(star,fitargs,pmin,pmax,figpath,plotfit=True,show=False)
                        if R2 > 0.6:
                            figpath = f'{fpath}/{startype}_bonafide'
                            self.periodogram_plot(star,fitargs,pmin,pmax,figpath,plotfit=True,show=False)

                    # Unclassified for visual inspection:
                    else:
                        ''' do two plots, one with LSG P and oter with 2P'''
                        # with P
                        fitdat1  = self._curve_fit_switch(t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase)
                        fitparams = fitdat1['fitparams']
                        R2 = fitdat1['fitinfo']['R2']
                        chi2 = fitdat1['fitinfo']['Chi2']
                        res = fitdat1['magseries']['residuals']
                        errs = fitdat1['magseries']['errs']
                        meanRESS = np.sum(res**2)/(len(res)-1)
                        red_chi2 = np.sum(res**2 / errs**2)/(len(res)-1)

                        fit_parameters  = [star,ra,dec,best_freq,amplitude,R2,chi2,meanRESS,red_chi2] + color_vals
                        fit_params_cols = ['ID','RA','DEC','Freq','Amplitude','R2','Chi2','residual_sum','red_chi2'] + color_cols
                        n=0 
                        k=1
                        while n < len(fitparams[0]):
                            fitpar = fitdat1['fitparams']
                            fitparerrs = fitdat1['fitparamserrs']
                            fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                            fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                            n+=1
                            k+=1
                        unc_P_fitparams.append(pd.Series(fit_parameters,index=fit_params_cols))

                        figpath = f'{fpath}/{startype}'
                        fitargs = [t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase]
                        self.periodogram_plot(star,fitargs,pmin,pmax,figpath,plotfit=True,show=False)
                        if R2 > 0.6:
                            figpath = f'{fpath}/{startype}_bonafide'
                            self.periodogram_plot(star,fitargs,pmin,pmax,figpath,plotfit=True,show=False)
                        

                        # with 2P
                        fitdat1  = self._curve_fit_switch(t,lc,lcErr,best_freq/2.,order,phaseSpace,fitFreq,iterative,fitastrobase)
                        fitparams = fitdat1['fitparams']
                        R2 = fitdat1['fitinfo']['R2']
                        chi2 = fitdat1['fitinfo']['Chi2']
                        res = fitdat1['magseries']['residuals']
                        errs = fitdat1['magseries']['errs']
                        meanRESS = np.sum(res**2)/(len(res)-1)
                        red_chi2 = np.sum(res**2 / errs**2)/(len(res)-1)

                        fit_parameters  = [star,ra,dec,best_freq,amplitude,R2,chi2,meanRESS,red_chi2] + color_vals
                        fit_params_cols = ['ID','RA','DEC','Freq','Amplitude','R2','Chi2','residual_sum','red_chi2'] + color_cols
                        n=0 
                        k=1
                        while n < len(fitparams[0]):
                            fitpar = fitdat1['fitparams']
                            fitparerrs = fitdat1['fitparamserrs']
                            fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                            fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                            n+=1
                            k+=1
                        unc_2P_fitparams.append(pd.Series(fit_parameters,index=fit_params_cols))
                j+=1
            i+=1
        ecl_fitparams = pd.DataFrame(ecl_fitparams)
        ecl_fitparams.set_index('ID',inplace=True)
        ecl_fitparams.to_csv(f'{fpath}/{self.tile}_ecl_fit_parameters.csv',sep=',')
        
        unc_P_fitparams = pd.DataFrame(unc_P_fitparams)
        unc_P_fitparams.set_index('ID',inplace=True)
        unc_P_fitparams.to_csv(f'{fpath}/{self.tile}_unc_P_parameters.csv',sep=',')
        
        unc_2P_fitparams = pd.DataFrame(unc_2P_fitparams)
        unc_2P_fitparams.set_index('ID',inplace=True)
        unc_2P_fitparams.to_csv(f'{fpath}/{self.tile}_unc_2P_parameters.csv',sep=',')











    def do_periodogram_replots(self,pmin,pmax,fpath,phaseSpace,fitFreq,order,iterative,fitastrobase):
        '''
        It does the first plot and automated classification for ECL by
        fitting two sinusoid series to each light curve and comparing the depth of
        the two valleys to standar deviation of the resifuals.
        If the depths difference is great than residual deviation it
        probabelly is a assimetrical ECL.
        '''
        print(f'Removing seasonal aliases from tile {self.tile}...')
        _, aliasfree_id = self.failure_modes(freq=1.0, sigfactor=1, plot=True)
        ecl_fitparams = []
        unc_P_fitparams = []
        unc_2P_fitparams = []
        output_data = []
        i = 1
        for chip in self.chips:
            self._read_tables(chip)
            self._read_periodogram_outputs(chip)
            candidates = [star for star in self.lsg_params.index if star in aliasfree_id]
            
            stars_done = self._stars_already_plotted()
            candidates = [_ for _ in candidates if _ not in stars_done]
            j=1
            for star in candidates:
                status._print(prefix=f'Plotting chip {chip}',
                              iter1=i,
                              length1=len(self.chips),
                              iter2=j,
                              length2=len(candidates),
                              sufix='%')

                t = self.obs_time
                lc = self.ks_mag.loc[star].values
                lcErr = self.ks_err.loc[star].values
                ra        = self.data_table.RA.loc[star]
                dec       = self.data_table.DEC.loc[star]
                best_freq = self.lsg_params['best_freq'].loc[star]
                amplitude = self.lsg_params['amplitude'].loc[star]
                # ZYJHKs psf mag for color classification
                color_cols = [col for col in self.data_table.columns if col.split('_')[0] == 'mag' or col.split('_')[0] == 'er']
                color_vals = [self.data_table.loc[star][col] for col in color_cols]
                
                if ((amplitude > 0.2) and (amplitude < 0.5)):
                    #period select
                    if ((1./best_freq > 0.4) and (1./best_freq < 0.6)):
                        # test for ECLs
                        c = sct.StarClassificator()
                        startype = c.ECL_classificator(t, lc, lcErr, best_freq, phaseSpace=True)
                        # ECL star type
                        if startype == 'ECL':
                            fitdat1  = self._curve_fit_switch(t,lc,lcErr,best_freq/2.,order,phaseSpace,fitFreq,iterative,fitastrobase)
                            if fitdat1 !=0:
                                fitparams = fitdat1['fitparams']
                                R2 = fitdat1['fitinfo']['R2']
                                chi2 = fitdat1['fitinfo']['Chi2']
                                res = fitdat1['magseries']['residuals']
                                errs = fitdat1['magseries']['errs']
                                meanRESS = np.sum(res**2)/(len(res)-1)
                                red_chi2 = np.sum(res**2 / errs**2)/(len(res)-1)

                                fit_parameters  = [star,ra,dec,best_freq,amplitude,R2,chi2,meanRESS,red_chi2] + color_vals
                                fit_params_cols = ['ID','RA','DEC','Freq','Amplitude','R2','Chi2','residual_sum','red_chi2'] + color_cols
                                n=0 
                                k=1
                                while n < len(fitparams[0]):
                                    fitpar = fitdat1['fitparams']
                                    fitparerrs = fitdat1['fitparamserrs']
                                    fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                                    fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                                    n+=1
                                    k+=1
                                ecl_fitparams.append(pd.Series(fit_parameters,index=fit_params_cols))
                                
                                figpath = f'{fpath}/{startype}_remaining'
                                fitargs = [t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase]
                                self.periodogram_plot(star,fitargs,pmin,pmax,figpath,plotfit=True,show=False)
                                if R2 > 0.6:
                                    figpath = f'{fpath}/{startype}_bonafide_remaining'
                                    self.periodogram_plot(star,fitargs,pmin,pmax,figpath,plotfit=True,show=False)

                        # Unclassified for visual inspection:
                        else:
                            ''' do two plots, one with LSG P and oter with 2P'''
                            # with P
                            fitdat1  = self._curve_fit_switch(t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase)
                            if fitdat1 !=0:
                                fitparams = fitdat1['fitparams']
                                R2 = fitdat1['fitinfo']['R2']
                                chi2 = fitdat1['fitinfo']['Chi2']
                                res = fitdat1['magseries']['residuals']
                                errs = fitdat1['magseries']['errs']
                                meanRESS = np.sum(res**2)/(len(res)-1)
                                red_chi2 = np.sum(res**2 / errs**2)/(len(res)-1)

                                fit_parameters  = [star,ra,dec,best_freq,amplitude,R2,chi2,meanRESS,red_chi2] + color_vals
                                fit_params_cols = ['ID','RA','DEC','Freq','Amplitude','R2','Chi2','residual_sum','red_chi2'] + color_cols
                                n=0 
                                k=1
                                while n < len(fitparams[0]):
                                    fitpar = fitdat1['fitparams']
                                    fitparerrs = fitdat1['fitparamserrs']
                                    fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                                    fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                                    n+=1
                                    k+=1
                                unc_P_fitparams.append(pd.Series(fit_parameters,index=fit_params_cols))

                                figpath = f'{fpath}/{startype}_remaining'
                                fitargs = [t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase]
                                self.periodogram_plot(star,fitargs,pmin,pmax,figpath,plotfit=True,show=False)
                                if R2 > 0.6:
                                    figpath = f'{fpath}/{startype}_bonafide_remaining'
                                    self.periodogram_plot(star,fitargs,pmin,pmax,figpath,plotfit=True,show=False)
                            

                            # with 2P
                            fitdat1  = self._curve_fit_switch(t,lc,lcErr,best_freq/2.,order,phaseSpace,fitFreq,iterative,fitastrobase)
                            if fitdat1 !=0:
                                fitparams = fitdat1['fitparams']
                                R2 = fitdat1['fitinfo']['R2']
                                chi2 = fitdat1['fitinfo']['Chi2']
                                res = fitdat1['magseries']['residuals']
                                errs = fitdat1['magseries']['errs']
                                meanRESS = np.sum(res**2)/(len(res)-1)
                                red_chi2 = np.sum(res**2 / errs**2)/(len(res)-1)

                                fit_parameters  = [star,ra,dec,best_freq,amplitude,R2,chi2,meanRESS,red_chi2] + color_vals
                                fit_params_cols = ['ID','RA','DEC','Freq','Amplitude','R2','Chi2','residual_sum','red_chi2'] + color_cols
                                n=0 
                                k=1
                                while n < len(fitparams[0]):
                                    fitpar = fitdat1['fitparams']
                                    fitparerrs = fitdat1['fitparamserrs']
                                    fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                                    fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                                    n+=1
                                    k+=1
                                unc_2P_fitparams.append(pd.Series(fit_parameters,index=fit_params_cols))
                j+=1
            i+=1
        ecl_fitparams = pd.DataFrame(ecl_fitparams)
        ecl_fitparams.set_index('ID',inplace=True)
        ecl_fitparams.to_csv(f'{fpath}/{self.tile}_ecl_fit_parameters_.csv',sep=',')
        
        unc_P_fitparams = pd.DataFrame(unc_P_fitparams)
        unc_P_fitparams.set_index('ID',inplace=True)
        unc_P_fitparams.to_csv(f'{fpath}/{self.tile}_unc_P_parameters_.csv',sep=',')
        
        unc_2P_fitparams = pd.DataFrame(unc_2P_fitparams)
        unc_2P_fitparams.set_index('ID',inplace=True)
        unc_2P_fitparams.to_csv(f'{fpath}/{self.tile}_unc_2P_parameters_.csv',sep=',')





    def do_replot(self,pmin,pmax,phaseSpace,fitFreq,order,iterative,fitastrobase):
        ''' 
        It replots periodogram plots after first visual inspection.
        You must perform visual inspection on previous plots, searching 
        for RRLyr in UNC/P folder. You may also repeat visual inspection in
        folder UNC/2P to check and search for new ECL.
        '''
        i = 1
        dirpath = f'{self.path}/lc_plots/short_period/pos_visual_inspection'
        subfolders = [_ for _ in sorted(os.listdir(dirpath)) if os.path.isdir(os.path.join(dirpath, _)) ]  #['RRLyr','ECL','IDK']
        ECL_params = []
        RRLyr_params = []
        UNC_params = []
        for chip in self.chips:
            self._read_tables(chip)
            self._read_periodogram_outputs(chip)
            for folder in subfolders:
                star_list = [_[4:-4] for _ in sorted(os.listdir(f'{self.path}/lc_plots/short_period/pos_visual_inspection/{folder}')) if _.endswith('png') and _[4:-24] == chip]
                stars2plot = star_list#[_ for _ in self.star_ids if _ in star_list and _ in self.lsg_params.index]
                j = 1
                for star in stars2plot:
                    status._print(  prefix=f'Plotting chip {chip}',
                                    iter1=i,
                                    length1=len(self.chips),
                                    iter2=j,
                                    length2=len(stars2plot),
                                    sufix='%')

                    t = self.obs_time
                    lc = self.ks_mag.loc[star].values
                    lcErr = self.ks_err.loc[star].values
                    ra        = self.data_table.RA.loc[star]
                    dec       = self.data_table.DEC.loc[star]
                    best_freq = self.lsg_params['best_freq'].loc[star]
                    amplitude = self.lsg_params['amplitude'].loc[star]
                    # ZYJHKs psf mag for color classification
                    color_cols = [col for col in self.data_table.columns if col.split('_')[0] == 'mag' or col.split('_')[0] == 'er']
                    color_vals = [self.data_table.loc[star][col] for col in color_cols]


                    if folder[:3] == 'ECL':
                        figpath = f'{self.path}/lc_plots/short_period/pos_visual_inspection/{folder}'

                        fitdat1  = self._curve_fit_switch(t,lc,lcErr,best_freq/2.,order,phaseSpace,fitFreq,iterative,fitastrobase)
                        fitparams = fitdat1['fitparams']
                        R2 = fitdat1['fitinfo']['R2']
                        chi2 = fitdat1['fitinfo']['Chi2']
                        res = fitdat1['magseries']['residuals']
                        errs = fitdat1['magseries']['errs']
                        meanRESS = np.sum(res**2)/(len(res)-1)
                        red_chi2 = np.sum(res**2 / errs**2)/(len(res)-1)

                        fit_parameters  = [star,ra,dec,best_freq,amplitude,R2,chi2,meanRESS,red_chi2] + color_vals
                        fit_params_cols = ['ID','RA','DEC','Freq','Amplitude','R2','Chi2','residual_sum','red_chi2'] + color_cols
                        
                        n=0 
                        k=1
                        while n < len(fitparams[0]):
                            fitpar = fitdat1['fitparams']
                            fitparerrs = fitdat1['fitparamserrs']
                            fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                            fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                            n+=1
                            k+=1
                        if folder == 'ECL':
                            ECL_params.append(pd.Series(fit_parameters,index=fit_params_cols))
                        fitargs = [t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase]
                        self.periodogram_plot(star,fitargs,pmin,pmax,figpath,plotfit=True,show=False)
                        
                    elif folder == 'RRLyr':
                        figpath = f'{self.path}/lc_plots/short_period/pos_visual_inspection/{folder}'

                        fitdat1  = self._curve_fit_switch(t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase)
                        fitparams = fitdat1['fitparams']
                        R2 = fitdat1['fitinfo']['R2']
                        chi2 = fitdat1['fitinfo']['Chi2']
                        res = fitdat1['magseries']['residuals']
                        errs = fitdat1['magseries']['errs']
                        meanRESS = np.sum(res**2)/(len(res)-1)
                        red_chi2 = np.sum(res**2 / errs**2)/(len(res)-1)

                        fit_parameters  = [star,ra,dec,best_freq,amplitude,R2,chi2,meanRESS,red_chi2] + color_vals
                        fit_params_cols = ['ID','RA','DEC','Freq','Amplitude','R2','Chi2','residual_sum','red_chi2'] + color_cols
                        n=0 
                        k=1
                        while n < len(fitparams[0]):
                            fitpar = fitdat1['fitparams']
                            fitparerrs = fitdat1['fitparamserrs']
                            fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                            fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                            n+=1
                            k+=1
                        RRLyr_params.append(pd.Series(fit_parameters,index=fit_params_cols))
                        fitargs = [t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase]
                        self.periodogram_plot(star,fitargs,pmin,pmax,figpath,plotfit=True,show=False)

                    else: # folder == 'IDK':
                        figpath = f'{self.path}/lc_plots/short_period/pos_visual_inspection/{folder}'

                        fitdat1  = self._curve_fit_switch(t,lc,lcErr,best_freq/2.,order,phaseSpace,fitFreq,iterative,fitastrobase)
                        fitparams = fitdat1['fitparams']
                        R2 = fitdat1['fitinfo']['R2']
                        chi2 = fitdat1['fitinfo']['Chi2']
                        res = fitdat1['magseries']['residuals']
                        errs = fitdat1['magseries']['errs']
                        meanRESS = np.sum(res**2)/(len(res)-1)
                        red_chi2 = np.sum(res**2 / errs**2)/(len(res)-1)

                        fit_parameters  = [star,ra,dec,best_freq,amplitude,R2,chi2,meanRESS,red_chi2] + color_vals
                        fit_params_cols = ['ID','RA','DEC','Freq','Amplitude','R2','Chi2','residual_sum','red_chi2'] + color_cols
                        n=0 
                        k=1
                        while n < len(fitparams[0]):
                            fitpar = fitdat1['fitparams']
                            fitparerrs = fitdat1['fitparamserrs']
                            fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                            fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                            n+=1
                            k+=1
                        if folder == 'IDK':
                            UNC_params.append(pd.Series(fit_parameters,index=fit_params_cols))
                        fitargs = [t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase]
                        self.periodogram_plot(star,fitargs,pmin,pmax,figpath,plotfit=True,show=False)
                    j+=1
            i+=1

        ECL_params = pd.DataFrame(ECL_params)
        ECL_params.set_index('ID',inplace=True)
        ECL_params.to_csv(f'{self.path}/lc_plots/short_period/pos_visual_inspection/{self.tile}_ecl_parameters.csv',sep=',')
        
        RRLyr_params = pd.DataFrame(RRLyr_params)
        RRLyr_params.set_index('ID',inplace=True)
        RRLyr_params.to_csv(f'{self.path}/lc_plots/short_period/pos_visual_inspection/{self.tile}_rrlyr_parameters.csv',sep=',')
        
        UNC_params = pd.DataFrame(UNC_params)
        UNC_params.set_index('ID',inplace=True)
        UNC_params.to_csv(f'{self.path}/lc_plots/short_period/pos_visual_inspection/{self.tile}_unc_parameters.csv',sep=',')


    def do_final_periodogram_plots(self,pmin,pmax,phaseSpace,fitFreq,order,iterative,fitastrobase):
        ''' 
        It replots periodogram plots after first visual inspection.
        You must perform visual inspection on previous plots, searching 
        for RRLyr in UNC/P folder. You may also repeat visual inspection in
        folder UNC/2P to check and search for new ECL.
        '''
        i = 1
        folders = ['RRLyr','ECL','IDK']
        ECL_params = []
        RRLyr_params = []
        UNC_params = []
        for chip in self.chips:
            self._read_tables(chip)
            self._read_periodogram_outputs(chip)
            for folder in folders:
                
                star_list = [_[4:-4] for _ in sorted(os.listdir(f'{self.path}/lc_plots/short_period/pos_visual_inspection/{folder}')) if _[4:-4] != '']
                stars2plot = [_ for _ in self.star_ids if _ in star_list and _ in self.lsg_params.index]
                j = 1
                for star in stars2plot:
                    status._print(  prefix=f'Plotting chip {chip}',
                                    iter1=i,
                                    length1=len(self.chips),
                                    iter2=j,
                                    length2=len(stars2plot),
                                    sufix='%')

                    t = self.obs_time
                    lc = self.ks_mag.loc[star].values
                    lcErr = self.ks_err.loc[star].values
                    ra        = self.data_table.RA.loc[star]
                    dec       = self.data_table.DEC.loc[star]
                    best_freq = self.lsg_params['best_freq'].loc[star]
                    amplitude = self.lsg_params['amplitude'].loc[star]
                    # ZYJHKs psf mag for color classification
                    color_cols = [col for col in self.data_table.columns if col.split('_')[0] == 'mag' or col.split('_')[0] == 'er']
                    color_vals = [self.data_table.loc[star][col] for col in color_cols]


                    if folder == 'ECL':
                        figpath = f'{self.path}/lc_plots/short_period/pos_visual_inspection/{folder}'

                        fitdat1  = self._curve_fit_switch(t,lc,lcErr,best_freq/2.,order,phaseSpace,fitFreq,iterative,fitastrobase)
                        fitparams = fitdat1['fitparams']

                        R2 = fitdat1['fitinfo']['R2']
                        chi2 = fitdat1['fitinfo']['Chi2']
                        res = fitdat1['magseries']['residuals']
                        errs = fitdat1['magseries']['errs']
                        RESS = np.sum(res**2)
                        red_chi2 = np.sum(res**2 / errs**2)/(len(res)-1)

                        res_std = np.std(res)
                        fit_parameters  = [star,ra,dec,best_freq,amplitude,R2,chi2,RESS,res_std,red_chi2] + color_vals
                        fit_params_cols = ['ID','RA','DEC','Freq','Amplitude','R2','Chi2','residual_sum','residual_std','red_chi2'] + color_cols
                        n=0 
                        k=1
                        while n < len(fitparams[0]):
                            fitpar = fitdat1['fitparams']
                            fitparerrs = fitdat1['fitparamserrs']
                            fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                            fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                            n+=1
                            k+=1
                        ECL_params.append(pd.Series(fit_parameters,index=fit_params_cols))
                        fitargs = [t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase]
                        self.periodogram_plot(star,fitargs,figpath,plotfit=True,show=False)

                    if folder == 'RRLyr':
                        figpath = f'{self.path}/lc_plots/short_period/pos_visual_inspection/{folder}'

                        fitdat1  = self._curve_fit_switch(t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase)
                        fitparams = fitdat1['fitparams']

                        R2 = fitdat1['fitinfo']['R2']
                        chi2 = fitdat1['fitinfo']['Chi2']
                        res = fitdat1['magseries']['residuals']
                        errs = fitdat1['magseries']['errs']
                        RESS = np.sum(res**2)
                        red_chi2 = np.sum(res**2 / errs**2)/(len(res)-1)

                        res_std = np.std(res)
                        fit_parameters  = [star,ra,dec,best_freq,amplitude,R2,chi2,RESS,res_std,red_chi2] + color_vals
                        fit_params_cols = ['ID','RA','DEC','Freq','Amplitude','R2','Chi2','residual_sum','residual_std','red_chi2'] + color_cols
                        n=0 
                        k=1
                        while n < len(fitparams[0]):
                            fitpar = fitdat1['fitparams']
                            fitparerrs = fitdat1['fitparamserrs']
                            fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                            fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                            n+=1
                            k+=1
                        RRLyr_params.append(pd.Series(fit_parameters,index=fit_params_cols))
                        fitargs = [t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase]
                        self.periodogram_plot(star,fitargs,figpath,plotfit=True,show=False)

                    if folder == 'IDK':
                        figpath = f'{self.path}/lc_plots/short_period/pos_visual_inspection/{folder}'

                        fitdat1  = self._curve_fit_switch(t,lc,lcErr,best_freq/2.,order,phaseSpace,fitFreq,iterative,fitastrobase)
                        fitparams = fitdat1['fitparams']


                        R2 = fitdat1['fitinfo']['R2']
                        chi2 = fitdat1['fitinfo']['Chi2']
                        res = fitdat1['magseries']['residuals']
                        errs = fitdat1['magseries']['errs']
                        RESS = np.sum(res**2)
                        red_chi2 = np.sum(res**2 / errs**2)/(len(res)-1)

                        res_std = np.std(res)
                        fit_parameters  = [star,ra,dec,best_freq,amplitude,R2,chi2,RESS,res_std,red_chi2] + color_vals
                        fit_params_cols = ['ID','RA','DEC','Freq','Amplitude','R2','Chi2','residual_sum','residual_std','red_chi2'] + color_cols
                        n=0 
                        k=1
                        while n < len(fitparams[0]):
                            fitpar = fitdat1['fitparams']
                            fitparerrs = fitdat1['fitparamserrs']
                            fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                            fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                            n+=1
                            k+=1
                        UNC_params.append(pd.Series(fit_parameters,index=fit_params_cols))
                        fitargs = [t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase]
                        self.periodogram_plot(star,fitargs,figpath,plotfit=True,show=False)
                    j+=1
            i+=1

        ECL_params = pd.DataFrame(ECL_params)
        ECL_params.set_index('ID',inplace=True)
        ECL_params.to_csv(f'{self.path}/lc_plots/short_period/pos_visual_inspection/{self.tile}_ecl_parameters.csv',sep=',')
        
        RRLyr_params = pd.DataFrame(RRLyr_params)
        RRLyr_params.set_index('ID',inplace=True)
        RRLyr_params.to_csv(f'{self.path}/lc_plots/short_period/pos_visual_inspection/{self.tile}_rrlyr_parameters.csv',sep=',')
        
        UNC_params = pd.DataFrame(UNC_params)
        UNC_params.set_index('ID',inplace=True)
        UNC_params.to_csv(f'{self.path}/lc_plots/short_period/pos_visual_inspection/{self.tile}_unc_parameters.csv',sep=',')




    def plot_remaining(self,pmin,pmax,phaseSpace,fitFreq,order,iterative,fitastrobase):
        '''
        It does the first plot and automated classification for ECL by
        fitting two sinusoid series to each light curve and comparing the depth of
        the two valleys to standar deviation of the resifuals.
        If the depths difference is great than residual deviation it
        probabelly is a assimetrical ECL.
        '''
        print(f'Removing seasonal aliases from tile {self.tile}...')
        #aliasfree_id, _ = self.failure_modes(freq=1.0, sigfactor=1, plot=True)
        _, aliasfree_id = self.failure_modes(freq=1.0, sigfactor=1, plot=True)
        stars_done = self._stars_already_plotted()
        ecl_fitparams = []
        unc_P_fitparams = []
        unc_2P_fitparams = []
        output_data = []
        i = 1
        for chip in self.chips:
            self._read_tables(chip)
            self._read_periodogram_outputs(chip)
            candidates = [star for star in self.lsg_params.index if star in aliasfree_id]
            candidates = [_ for _ in candidates if _ not in stars_done]
            j=1
            for star in candidates:
                status._print(prefix=f'Plotting chip {chip}',
                              iter1=i,
                              length1=len(self.chips),
                              iter2=j,
                              length2=len(candidates),
                              sufix='%')

                t = self.obs_time
                lc = self.ks_mag.loc[star].values
                lcErr = self.ks_err.loc[star].values
                ra        = self.data_table.RA.loc[star]
                dec       = self.data_table.DEC.loc[star]
                best_freq = self.lsg_params['best_freq'].loc[star]
                amplitude = self.lsg_params['amplitude'].loc[star]
                # ZYJHKs psf mag for color classification
                color_cols = [col for col in self.data_table.columns if col.split('_')[0] == 'mag' or col.split('_')[0] == 'er']
                color_vals = [self.data_table.loc[star][col] for col in color_cols]
                                
                #amplitude selection
                if ((amplitude > 0.2) and (amplitude < 0.5)):
                    #period select
                    if ((1./best_freq > 0.4) and (1./best_freq < 0.5)):
                        # test for ECLs
                        c = sct.StarClassificator()
                        startype = c.ECL_classificator(t, lc, lcErr, best_freq, phaseSpace=True)

                        ''' renaming old names'''
                        # ECL star type
                        if startype == 'ECL':
                            doubleP = True
                            figpath = f'{self.path}/lc_plots/short_period/pre_visual_inspection/remaining2/{startype}'

                            fitdat1  = self._curve_fit_switch(t,lc,lcErr,best_freq/2.,order,phaseSpace,fitFreq,iterative,fitastrobase)
                            
                            fitparams = fitdat1['fitparams']

                            fit_parameters  = [star,ra,dec,best_freq,amplitude] + color_vals
                            fit_params_cols = ['ID','RA','DEC','Freq','Amplitude'] + color_cols
                            n=0 
                            k=1
                            while n < len(fitparams[0]):
                                fitpar = fitdat1['fitparams']
                                fitparerrs = fitdat1['fitparamserrs']
                                fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                                fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                                n+=1
                                k+=1
                            ecl_fitparams.append(pd.Series(fit_parameters,index=fit_params_cols))
                            fitargs = [t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase]
                            self.periodogram_plot(star,fitargs,pmin,pmax,figpath,plotfit=True,show=False)


                        # Unclassified for visual inspection:
                        else:
                            ''' do two plots, one with LSG P and oter with 2P'''
                            # with P
                            doubleP = False
                            figpath = f'{self.path}/lc_plots/short_period/pre_visual_inspection/remaining2/{startype}/P'
                            fitdat1  = self._curve_fit_switch(t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase)
                            fitparams = fitdat1['fitparams']
                            fitargs = [t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase]
                            self.periodogram_plot(star,fitargs,pmin,pmax,figpath,plotfit=True,show=False)
                            
                            fit_parameters  = [star,ra,dec,best_freq,amplitude] + color_vals
                            fit_params_cols = ['ID','RA','DEC','Freq','Amplitude'] + color_cols
                            n=0 
                            k=1
                            while n < len(fitparams[0]):
                                fitpar = fitdat1['fitparams']
                                fitparerrs = fitdat1['fitparamserrs']
                                fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                                fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                                n+=1
                                k+=1
                            unc_P_fitparams.append(pd.Series(fit_parameters,index=fit_params_cols))

                            # with 2P
                            doubleP = True
                            figpath = f'{self.path}/lc_plots/short_period/pre_visual_inspection/remaining2/{startype}/2P'
                            fitdat1  = self._curve_fit_switch(t,lc,lcErr,best_freq/2.,order,phaseSpace,fitFreq,iterative,fitastrobase)
                            fitparams = fitdat1['fitparams']
                            fitargs = [t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase]
                            self.periodogram_plot(star,fitargs,pmin,pmax,figpath,plotfit=True,show=False)

                            fit_parameters  = [star,ra,dec,best_freq/2.,amplitude] + color_vals
                            fit_params_cols = ['ID','RA','DEC','Freq','Amplitude'] + color_cols
                            n=0 
                            k=1
                            while n < len(fitparams[0]):
                                fitpar = fitdat1['fitparams']
                                fitparerrs = fitdat1['fitparamserrs']
                                fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                                fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                                n+=1
                                k+=1
                            unc_2P_fitparams.append(pd.Series(fit_parameters,index=fit_params_cols))
                j+=1
            i+=1
        ecl_fitparams = pd.DataFrame(ecl_fitparams)
        ecl_fitparams.set_index('ID',inplace=True)
        ecl_fitparams.to_csv(f'{self.path}/lc_plots/short_period/pre_visual_inspection/remaining2/{self.tile}_ecl_fit_parameters.csv',sep=',')
        
        unc_P_fitparams = pd.DataFrame(unc_P_fitparams)
        unc_P_fitparams.set_index('ID',inplace=True)
        unc_P_fitparams.to_csv(f'{self.path}/lc_plots/short_period/pre_visual_inspection/remaining2/{self.tile}_unc_P_parameters.csv',sep=',')
        
        unc_2P_fitparams = pd.DataFrame(unc_2P_fitparams)
        unc_2P_fitparams.set_index('ID',inplace=True)
        unc_2P_fitparams.to_csv(f'{self.path}/lc_plots/short_period/pre_visual_inspection/remaining2/{self.tile}_unc_2P_parameters.csv',sep=',')


    def do_bonafide_plots(self,pmin,pmax,phaseSpace,fitFreq,order,iterative,fitastrobase):
        ''' 
        It replots periodogram plots after first visual inspection.
        You must perform visual inspection on previous plots, searching 
        for RRLyr in UNC/P folder. You may also repeat visual inspection in
        folder UNC/2P to check and search for new ECL.
        '''
        i = 1
        folders = ['RRLyr','ECL_bonafide','UNC_bonafide']
        ECL_params = []
        RRLyr_params = []
        UNC_params = []
        for chip in self.chips:
            self._read_tables(chip)
            self._read_periodogram_outputs(chip)
            for folder in folders:
                
                star_list = [_[4:-4] for _ in sorted(os.listdir(f'{self.path}/lc_plots/short_period/pos_visual_inspection/{folder}')) if _[4:-4] != '']
                stars2plot = [_ for _ in self.star_ids if _ in star_list and _ in self.lsg_params.index]
                j = 1
                for star in stars2plot:
                    status._print(  prefix=f'Plotting chip {chip}',
                                    iter1=i,
                                    length1=len(self.chips),
                                    iter2=j,
                                    length2=len(stars2plot),
                                    sufix='%')

                    t = self.obs_time
                    lc = self.ks_mag.loc[star].values
                    lcErr = self.ks_err.loc[star].values
                    ra        = self.data_table.RA.loc[star]
                    dec       = self.data_table.DEC.loc[star]
                    best_freq = self.lsg_params['best_freq'].loc[star]
                    amplitude = self.lsg_params['amplitude'].loc[star]
                    # ZYJHKs psf mag for color classification
                    color_cols = [col for col in self.data_table.columns if col.split('_')[0] == 'mag' or col.split('_')[0] == 'er']
                    color_vals = [self.data_table.loc[star][col] for col in color_cols]


                    if folder == 'ECL_bonafide':
                        figpath = f'{self.path}/lc_plots/short_period/pos_visual_inspection/{folder}2'

                        fitdat1  = self._curve_fit_switch(t,lc,lcErr,best_freq/2.,order,phaseSpace,fitFreq,iterative,fitastrobase)
                        fitparams = fitdat1['fitparams']

                        R2 = fitdat1['fitinfo']['R2']
                        chi2 = fitdat1['fitinfo']['Chi2']
                        res = fitdat1['magseries']['residuals']
                        errs = fitdat1['magseries']['errs']
                        RESS = np.sum(res**2)
                        red_chi2 = np.sum(res**2 / errs**2)/(len(res)-1)

                        res_std = np.std(res)
                        fit_parameters  = [star,ra,dec,best_freq,amplitude,R2,chi2,RESS,res_std,red_chi2] + color_vals
                        fit_params_cols = ['ID','RA','DEC','Freq','Amplitude','R2','Chi2','residual_sum','residual_std','red_chi2'] + color_cols
                        n=0 
                        k=1
                        while n < len(fitparams[0]):
                            fitpar = fitdat1['fitparams']
                            fitparerrs = fitdat1['fitparamserrs']
                            fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                            fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                            n+=1
                            k+=1
                        ECL_params.append(pd.Series(fit_parameters,index=fit_params_cols))
                        fitargs = [t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase]
                        self.periodogram_plot(star,fitargs,pmin,pmax,figpath,plotfit=True,show=False)
                        

                    if folder == 'RRLyr':
                        figpath = f'{self.path}/lc_plots/short_period/pos_visual_inspection/{folder}2'

                        fitdat1  = self._curve_fit_switch(t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase)
                        fitparams = fitdat1['fitparams']

                        R2 = fitdat1['fitinfo']['R2']
                        chi2 = fitdat1['fitinfo']['Chi2']
                        res = fitdat1['magseries']['residuals']
                        errs = fitdat1['magseries']['errs']
                        RESS = np.sum(res**2)
                        red_chi2 = np.sum(res**2 / errs**2)/(len(res)-1)

                        res_std = np.std(res)
                        fit_parameters  = [star,ra,dec,best_freq,amplitude,R2,chi2,RESS,res_std,red_chi2] + color_vals
                        fit_params_cols = ['ID','RA','DEC','Freq','Amplitude','R2','Chi2','residual_sum','residual_std','red_chi2'] + color_cols
                        n=0 
                        k=1
                        while n < len(fitparams[0]):
                            fitpar = fitdat1['fitparams']
                            fitparerrs = fitdat1['fitparamserrs']
                            fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                            fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                            n+=1
                            k+=1
                        RRLyr_params.append(pd.Series(fit_parameters,index=fit_params_cols))
                        fitargs = [t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase]
                        self.periodogram_plot(star,fitargs,pmin,pmax,figpath,plotfit=True,show=False)

                    if folder == 'UNC_bonafide':
                        figpath = f'{self.path}/lc_plots/short_period/pos_visual_inspection/{folder}2'

                        fitdat1  = self._curve_fit_switch(t,lc,lcErr,best_freq/2.,order,phaseSpace,fitFreq,iterative,fitastrobase)
                        fitparams = fitdat1['fitparams']


                        R2 = fitdat1['fitinfo']['R2']
                        chi2 = fitdat1['fitinfo']['Chi2']
                        res = fitdat1['magseries']['residuals']
                        errs = fitdat1['magseries']['errs']
                        RESS = np.sum(res**2)
                        red_chi2 = np.sum(res**2 / errs**2)/(len(res)-1)

                        res_std = np.std(res)
                        fit_parameters  = [star,ra,dec,best_freq,amplitude,R2,chi2,RESS,res_std,red_chi2] + color_vals
                        fit_params_cols = ['ID','RA','DEC','Freq','Amplitude','R2','Chi2','residual_sum','residual_std','red_chi2'] + color_cols
                        n=0 
                        k=1
                        while n < len(fitparams[0]):
                            fitpar = fitdat1['fitparams']
                            fitparerrs = fitdat1['fitparamserrs']
                            fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                            fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                            n+=1
                            k+=1
                        UNC_params.append(pd.Series(fit_parameters,index=fit_params_cols))
                        fitargs = [t,lc,lcErr,best_freq,order,phaseSpace,fitFreq,iterative,fitastrobase]
                        self.periodogram_plot(star,fitargs,pmin,pmax,figpath,plotfit=True,show=False)
                    j+=1
            i+=1

        ECL_params = pd.DataFrame(ECL_params)
        ECL_params.set_index('ID',inplace=True)
        ECL_params.to_csv(f'{self.path}/lc_plots/short_period/pos_visual_inspection/{self.tile}_ecl_bona_parameters.csv',sep=',')
        
        RRLyr_params = pd.DataFrame(RRLyr_params)
        RRLyr_params.set_index('ID',inplace=True)
        RRLyr_params.to_csv(f'{self.path}/lc_plots/short_period/pos_visual_inspection/{self.tile}_rrlyr_bona_parameters.csv',sep=',')
        
        UNC_params = pd.DataFrame(UNC_params)
        UNC_params.set_index('ID',inplace=True)
        UNC_params.to_csv(f'{self.path}/lc_plots/short_period/pos_visual_inspection/{self.tile}_unc_bona_parameters.csv',sep=',')


    def merge_parameter_files(self):
        RRLyr_params = pd.read_csv(f'{self.path}/lc_plots/short_period/pos_visual_inspection/{self.tile}_rrlyr_bona_parameters.csv',sep=',',index_col='ID')
        UNC_params_remaining = pd.read_csv(f'{self.path}/lc_plots/short_period/pre_visual_inspection/{self.tile}_unc_P_parameters_.csv',sep=',',index_col='ID')
        rrl_ids = [_[4:-4] for _ in sorted(os.listdir(f'{self.path}/lc_plots/short_period/pos_visual_inspection/RRLyr')) if _[4:-4] != '']
        for star in UNC_params_remaining.index:
            if star in rrl_ids:
                RRLyr_params.loc[star] = UNC_params_remaining.loc[star]

        RRLyr_params.to_csv(f'{self.path}/lc_plots/short_period/pos_visual_inspection/{self.tile}_rrlyr_bona_parameters.csv',sep=',')


if __name__ == '__main__':
    #reload my libs for testing
    import importlib
    importlib.reload(sys.modules['clean_match_tables'])
    importlib.reload(sys.modules['fit_sin_series'])
    importlib.reload(sys.modules['periodogram'])
    importlib.reload(sys.modules['variability_indicator'])
    importlib.reload(sys.modules['star_classificator_tools'])
     
    path = '/home/botan/OneDrive/Doutorado/VVV_DATA/data/psf_ts'
    tiles = sorted(os.listdir('data/psf_ts/'))
    for tile in tiles:#[5:6]:
        #tile = 'b293'
        # PERIOD SEARCHs
        p_search = PeriodSearch(path=path, tile=tile, minP=0.1, maxP=1000 , varIndex='chi2')

        #p_search.organize_tables()
        #p_search.select_candidates()
        #p_search.do_periodogram(exists=True)
        
        #p_search.do_periodogram_fix()
        #p_search.failure_modes(freq=1.0, sigfactor=1, plot=True)


        #figurepath = f'{path}/{tile}/lc_plots/short_period/pre_visual_inspection'
    
        #figurepath = f'{path}/{tile}/lc_plots/long_period/pre_visual_inspection'
        #pmin = 0.1
        #pmax = 1.0
        #p_search.do_periodogram_replots(pmin=pmin,pmax=pmax,fpath=figurepath,phaseSpace=True,fitFreq=True,order=5,iterative=False,fitastrobase=False)
    
        #p_search.do_replot(pmin=0.1,pmax=1.0,phaseSpace=True,fitFreq=True,order=5,iterative=False,fitastrobase=False)
        
        #p_search.plot_remaining(pmin=0.1,pmax=1000,phaseSpace=True,fitFreq=True,order=5,iterative=False,fitastrobase=False)
    
        #p_search.do_final_periodogram_plots(pmin=0.1,pmax=1.0,phaseSpace=True,fitFreq=True,order=5,iterative=False,fitastrobase=False)
        #p_search.do_bonafide_plots(pmin=0.1,pmax=1.0,phaseSpace=True,fitFreq=True,order=5,iterative=False,fitastrobase=False)

        p_search.merge_parameter_files()