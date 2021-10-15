# -*- coding: utf-8 -*-
"""
Created on Jun 16 2020

@author: Everton Botan
@supervisor: Roberto Saito

imput:
    tile:       tile name ex.: "b294"
    method:     method to perform uncorrelated (correlated in future) selection:
                    "std" to use standar deviation
                    "chi2" to use reduced chi squared
    maxMag:     limiting magnitude for brighest stars
    sample_min: min number of epoch 
output:
    selected index - run over a single chip - for importing on other program/routine.
    files with selected indexes - run over the whole tile, named by chips - for run it itself.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import optimize, signal
import matplotlib as mpl
import matplotlib.pyplot as plt
from termcolor import colored

class Variability(object):

    def __init__(self, path, tile, method, maxMag, stdRatio=1.5, minChi2=2, savePlot=True):
        self.path = f'{path}'
        self.tile = tile
        self.chips = [fn[:-3] for fn in sorted(os.listdir(f'{self.path}/chips/')) if fn.endswith('ts')]
        self.method = method
        self.maxMag = maxMag
        self.stdRatio = stdRatio
        self.minChi2 = minChi2
        self.savePlot = savePlot
        os.makedirs(f'{self.path}/var_data',exist_ok=True)

    def _read_tables(self,chip):
        self.data_table  = pd.read_csv(f'{self.path}/chips/{chip}.ts',index_col=0,sep=',',low_memory=False)
        mag_cols         = [col for col in self.data_table.columns if col.split('_')[0] == 'MAG']
        err_cols         = [col for col in self.data_table.columns if col.split('_')[0] == 'ERR']
        self.mag_table   = self.data_table[mag_cols]
        self.err_table   = self.data_table[err_cols]
    

    def _wheigt_mean(self,x_val,x_err):
        # a wheigtened mean (average)
        w = 1.0/x_err**2
        mean = (w.values*x_val).sum(axis=1) / w.sum(axis=1)
        return mean


    def _std(self,x_val,x_err):
        n = np.shape(x_val)[1]
        sigma = np.sqrt( ((x_val.subtract(x_val.mean(axis=1),axis='index'))**2).sum(axis=1) / (n-1) )
        return sigma


    def _wheigt_std(self,x_val,x_err):
        # a wheigtened standard deviation
        # computed from individual magnitudes and photometric uncertainties
        w = 1.0/x_err**2
        n = np.shape(x_val)[1]
        x_avg = self._wheigt_mean(x_val,x_err)
        sigma =np.sqrt( n * ( (w.values*(x_val.subtract(x_avg,axis='index'))**2).sum(axis=1) ) / ( (n-1) * w.sum(axis=1) ) )
        return sigma


    def _chi2(self,x_val,x_err):
        #n = np.shape(x_val)[1] # wrong -    not all LC has x_val length
        n = (~np.isnan(x_val)).sum(axis=1)
        chi2 = ( ( ( x_val.subtract( x_val.mean(axis=1),axis='index' ) )**2 /x_err.values**2 ).sum(axis=1) )/(n-1)
        return chi2


    def _strateva(self,x,*args):
        ''' 
        Strateva, I., Ivezić, Ž., Knapp, G. R., et al. AJ, 122, 1861 (2001)
        f = a + b*10**(0.4*x) + c*10**(0.8*x)
        Ferreira Lopes C. E., Cross N. J. G. A&A 604, A121 (2017) [eq 18]
        f = a + b*10**(0.4*x) + c*10**(0.5*x) + d*10**(-1.4*x)
        '''
        if len(args) == 4:
            c = [0,0.4,0.5,-1.4]
            f = 0
            for i,n in enumerate(args):
                f += n*10**(c[i]*x)
        elif len(args) == 3:
            c = [0,0.4,0.8]
            f = 0
            for i,n in enumerate(args):
                f += n*10**(c[i]*x)
        else:
            raise RuntimeError('Wrong number of constants for Strateva function')
        return f
    

    def _fit_strateva(self,x,y,err,abs_sigma):
        flag = ~(pd.isnull(x) | pd.isnull(y))
        init_guess = [1e-3,1e-5,1e-8]
        fit, cov = curve_fit(self._strateva,x[flag],y[flag],
                        sigma=err[flag],
                        p0=init_guess,
                        absolute_sigma=abs_sigma)
        return fit



    def select_stars(self,chip):
        self._read_tables(chip)
        mag_mean = self.mag_table.mean(axis=1)
        err_mean = self.err_table.mean(axis=1)
        #filter to brightest stars:
        flag = mag_mean > self.maxMag
        mag_mean  = mag_mean[flag]
        err_mean  = err_mean[flag]
        mag_table = self.mag_table[flag]
        err_table = self.err_table[flag]
        candidates = []

        if self.method == 'std': 
            #select stars by comparing standard deviation expected randon noise (std from regular stars):
            std = self._wheigt_std(mag_table,err_table)
            #fit model to standard deviation
            std_a, std_b, std_c = self._fit_strateva(mag_mean, std, err_mean, True)
            std_model_x = np.linspace(mag_mean.min(),mag_mean.max(),1000)
            std_model_y = self._strateva(std_model_x, std_a, std_b, std_c)
            #fit model to errors
            err_a, err_b, err_c = self._fit_strateva(mag_mean, err_mean, err_mean, False)
            err_model_x = np.linspace(mag_mean.min(),mag_mean.max(),1000)
            err_model_y = self._strateva(err_model_x, err_a, err_b, err_c)
            for star in mag_table.index:
                star_mag = mag_mean.loc[star]
                star_std = std.loc[star]
                star_fit_std = self._strateva(star_mag, std_a, std_b, std_c)
                std_index = star_std / star_fit_std
                if std_index > self.stdRatio:
                    candidates.append(star)

        if self.method == 'chi2':
            #uses the reduced chi^2 to identify variable stars
            chi2 = self._chi2(mag_table,err_table)
            for star in chi2.index[chi2 > self.minChi2]:
                candidates.append(star)
            if self.savePlot:
                self._plot_histogram(chip,chi2)
                self._plot_red_chi2(mag_mean,chip,chi2,chi2cut=self.minChi2)
        
        filepath = f'{self.path}/var_data/{chip}.{self.method}cand'
        with open(filepath, 'w') as file_handler:
            for item in candidates:
                file_handler.write(f'{item}\n')
        if self.savePlot:
            self._plot_dispersion(chip,candidates,mag_mean,err_mean,mag_table,err_table)
            
        return candidates

    def _plot_dispersion(self,chip,stars,mag_mean,err_mean,mag_table,err_table):
        from matplotlib import rc
        rc('text', usetex=True)
        plt.rcParams.update({'font.size': 12})

        std = self._wheigt_std(mag_table,err_table)
        flag = ~(mag_mean.isna() | std.isna())
        std = std[flag]
        mag_mean = mag_mean[flag]
        err_mean = err_mean[flag]
        
        
        std_a, std_b, std_c = self._fit_strateva(mag_mean, std, err_mean, True)
        std_model_x = np.linspace(mag_mean.min(),mag_mean.max(),1000)
        std_model_y = self._strateva(std_model_x, std_a, std_b, std_c)
        #fit model to errors
        err_a, err_b, err_c = self._fit_strateva(mag_mean, err_mean, err_mean, False)
        err_model_x = np.linspace(mag_mean.min(),mag_mean.max(),1000)
        err_model_y = self._strateva(err_model_x, err_a, err_b, err_c)
        
        plt.figure(figsize=[6,2.5], tight_layout=True)

        #plt.hist2d(mag_mean,std,bins=(300,150),norm=mpl.colors.LogNorm(1,40,False),cmap=mpl.cm.rainbow,lw=0)
        #plt.colorbar()
        plt.scatter(mag_mean,std, marker='o',s=1, lw=0,c='k', alpha=.12)
        plt.scatter(mag_mean.loc[mag_mean.index.intersection(stars)],
                    std.loc[mag_mean.index.intersection(stars)],
                    marker='o',s=1, lw=0,c='r',label=f'Candidates ({self.method})',alpha=.5)
        if self.method == 'std':
            plt.plot(std_model_x,std_model_y*1.5,label=r'$\frac{\sigma}{Strateva} > 1.5$')
        #plt.plot(err_model_x,err_model_y,"brown",label='Photometric err fit')
        #plt.legend()
        plt.xlabel(r'$K_s$ [mag]')
        plt.ylabel(r'$\sigma K_s$') 
        plt.ylim(-0.0,0.28)
        #plt.tight_layout()
        plt.savefig(f'{self.path}/var_data/{self.method}_{chip}.png',dpi=300,bbox_inches = 'tight',pad_inches=0.05)
        plt.close()

    
    def _plot_red_chi2(self,mag_mean,chip,chi2,chi2cut):
        from matplotlib import rc
        rc('text', usetex=True)
        plt.rcParams.update({'font.size': 12})

        plt.figure(figsize=[6,2.5], tight_layout=True)

        #plt.hist2d(mag_mean,std,bins=(300,150),norm=mpl.colors.LogNorm(1,40,False),cmap=mpl.cm.rainbow,lw=0)
        #plt.colorbar()
        plt.scatter(mag_mean,chi2, marker='o',s=2, lw=0,c='k', alpha=.13)
        plt.hlines(y=chi2cut,xmin=mag_mean.min(),xmax=mag_mean.max(),colors='red',lw=.7)
        plt.xlabel(r'$K_s$ [mag]')
        plt.ylabel(r'$\chi^2$') 
        plt.ylim(-0.0,8)
        #plt.tight_layout()
        plt.savefig(f'{self.path}/var_data/red_chi2_{chip}.png',dpi=300,bbox_inches = 'tight',pad_inches=0.05)
        plt.close()


    def _plot_histogram(self,chip,chi2):
        plt.figure(figsize=[10,6])
        plt.hist(chi2,bins=100,histtype='step')
        plt.axvline(2,c='r',ls='--',label=r'$\chi^2 = 2$' )
        plt.xlim(0,70)
        plt.xlabel(r'$\chi^2$')
        plt.ylabel('N')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.path}/var_data/hist_{chip}.png',dpi=300)
        plt.close()


    def do_selection(self):
        i=1
        for chip in self.chips:
            self._status(prefix=f'Selecting candidates using {self.method} on chip {chip}',
                         iter1=i,
                         length1=len(self.chips),
                         iter2=i,
                         length2=len(self.chips),
                         sufix='%')
            self.select_stars(chip)
            i+=1

    def _status(self, prefix, iter1, length1, iter2, length2, sufix):
        print(colored('\r%s: [%i/%i] %.1f %s'%(prefix,
                                                iter1,
                                                length1,
                                                iter2/length2*100,
                                                sufix), 
                                                'yellow'),
                                                end='\r')

if __name__ == '__main__':
    
    tiles = sorted(os.listdir("data/psf_ts/"))
    for tile in tiles:
        path = f'/home/botan/Dropbox/Doutorado/VVV_DATA/data/psf_ts/{tile}'
        candidates = Variability(path=path,
                                 tile=tile,
                                 method="std",
                                 maxMag=11.5,
                                 stdRatio=1.5,
                                 minChi2=2,
                                 savePlot=True)
        candidates.do_selection()