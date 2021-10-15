# -*- coding: utf-8 -*-
"""
Created on Jul 3 2020

@author: Everton Botan
@supervisor: Roberto Saito, Antônio Kanaan

imputs:
    x:                time.
    y:                signal.
    yerr:             signal error.
    minP:             minimal period for frequency grid.
    maxP:             maximum period for frequency grid.
    normalization:    "standard","model","log","psd". "psd" is designed to be comparable
                      to standard Fourier power spectrum density.
    method:           "baluev","bootstrap","scargle". False alarm probability method. 
                      "scargle" is the aplication of Scargle (1982) false alarm probability. 
                      When used the power spectrum is not normalized.
    samples_per_peak: number of frequency points per spectral line.
    false:            false alarm probability level.
    nbins:            number of bins for PDM.
    covers:           number of cobers for PDM.
    mode:             0 (or  False): for calling functions when importing, 
                      1: perform Lomb-Scargle, 
                      2: perform CyPDM, 
                      3: perform PDM, 
                      4: perform LSG with Spectral Window analisys (DEVELOPPING).
Outputs:
    frequency:        frequency grid.
    power/theta:      Lomb-Scargle power (LSG) or theta (PDM) probability spectrum.
    false_alarm:      false alarm probabílity for each frequency from frequency grid.
    sig:              observed probability for 0.1% false alarm level.
    best_freq:        best frequency among all frequencies selected above fap_level.
                      It may use (if mode = 2) prewhitening strategy to test for alias. 
                      See Kepler, S. O. Baltic Astronomy, vol.2, 515-529, 1993.
    all_freq:         return all frequencies above fap_level.
"""

import sys
import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.special import gammaln
from scipy.stats import beta
from PyAstronomy.pyTiming import pyPDM

sys.path.append('home/botan/Dropbox/Doutorado/VVV_DATA/my_modules/')
import fit_sin_series as fitSin
import cypdm

class Periodogram(object):

    def __init__(self, x, y, yerr, minP, maxP, 
                    normalization="standard",
                    method="baluev",
                    samples_per_peak=10,
                    false=0.001,
                    nbins=10,
                    covers=3,
                    mode=False):

        msk = ~(np.isnan(yerr) | np.isnan(y))
        self.t = x[msk]
        self.lc = y[msk]
        self.lcErr = yerr[msk]
        self.maxfreq = 1./minP
        self.minfreq = 1./maxP
        self.normalization = normalization
        self.method = method
        self.samples_per_peak = samples_per_peak
        self.false = false
        self.nbins = nbins
        self.covers = covers
        #if mode:
        #    self.mode = mode
        #    self.periodogram_params = self._do_periodogram()

    def _lsg_false_alarm(self,power,avg_power,N):
        false = 1 - ( 1 - np.exp( -power/avg_power ) )**N
        return false


    def _lsg_fap_level(self,false,avg_power,N):
        p_obs = np.log( N/false ) * avg_power
        return p_obs


    def _all_frequencies(self, x, y, y_level, dx=0.1):
        """ return all frequencies above fap_level """
        peaks, _ = find_peaks(y, height=y_level)
        i = 0 
        peaks_ = [] 
        while i < len(peaks): 
            msk = ((x[peaks] <= x[peaks][i] + dx) & (x[peaks] >= x[peaks][i])) 
            max_peak = peaks[msk][np.argmax(y[peaks[msk]])] 
            if max_peak not in peaks_: 
                peaks_.append(max_peak) 
            i+=msk.sum()
        return peaks_


    def LSG(self):
        ls = LombScargle(self.t,self.lc,self.lcErr,normalization=self.normalization)
        #W = self.t[-1] - self.t[0]
        #df = 1.0/(2*self.samples_per_peak*W)
        #frequency = np.arange(self.minfreq,self.maxfreq,df)
        #power = ls.power(frequency)
        frequency, power = ls.autopower(minimum_frequency=self.minfreq,
                                        maximum_frequency=self.maxfreq,
                                        samples_per_peak=self.samples_per_peak)
        if self.method == "scargle":
            self.normalization = "psd"
            avg_power = power.mean()
            N = len(frequency)//self.samples_per_peak
            false_alarm = self._lsg_false_alarm(power,avg_power,N)
            sig = self._lsg_fap_level(self.false,avg_power,N)
        else:
            false_alarm = ls.false_alarm_probability(power,method=self.method)
            sig = ls.false_alarm_level(self.false,method=self.method)
        
        peaks = self._all_frequencies(frequency,power,sig) # all peaks above significance.
        all_freq = np.array([frequency[peaks][np.argsort(power[peaks])[::-1]], power[peaks][np.argsort(power[peaks])[::-1]]]) # sorted reverse. Bestfreq is the first value.
        if all_freq[0].shape[0] > 0:
            best_freq = all_freq[0][0]
            fap = false_alarm[peaks][np.argsort(power[peaks])[::-1]][0]
        else:
            best_freq = -99.99
            fap = -99.99
        return frequency, power, false_alarm, best_freq, fap, sig, all_freq


    def _beta_incomplete(self, x, a, b):
        ''' based on section 6.4 Numerical Recipies in C '''
        if ((x < 0.0) | (x > 1.0)):
            raise RuntimeError('Bad theta in routine betai')
        if ((x == 0.0) | (x == 1.0)):
            bt = 0.0
        else:
            #factors in front of the continued fraction
            bt = np.exp(gammaln(a + b) - gammaln(a) - gammaln(b) + a*np.log10(x) + b*np.log10(1.0 - x))
        if (x < (a+1.0)/(a+b+2.0)):  #uses continued fraction directly
            out = bt * beta.pdf(x,a,b)/a
        else:
            out = 1.0 - bt * beta.pdf(1.0-x,b,a)/b #uses continued fraction after making the symetry transformation 
        return out


    def _pdm_false_alarm(self, theta, nEpoch, bins, nf):
        '''see PDM manual from Stellingwerf webpage: http://www.stellingwerf.com/rfs-bin/index.cgi?action=PageView&id=34'''
        a = (nEpoch-bins)/2.0
        b = (bins-1.0)/2.0
        x = (nEpoch-bins)*theta/(nEpoch-1.0)
        fap = 1 - (1 - self._beta_incomplete(x, a, b))**(nf/self.samples_per_peak)
        return fap


    def _pdm_fap_level(self, false, theta, nEpoch, bins, nf):
        theta = np.arange(0.001,1.0,0.001)
        for i,x in enumerate(theta):
            fap_level = self._pdm_false_alarm(x, nEpoch, bins, nf)
            if fap_level > 0.001:
                theta_obs = theta[i-1]
                break
        return theta_obs


    def PDM(self):
        W = self.t[-1] - self.t[0]
        df = 1.0/(self.samples_per_peak * W)
        scan = pyPDM.Scanner(minVal=self.minfreq, maxVal=self.maxfreq, dVal=df, mode="frequency")
        pdm = pyPDM.PyPDM(self.t, self.lc)
        frequency, theta = pdm.pdmEquiBinCover(self.nbins, self.covers, scan)
        best_freq = frequency[theta == theta.min()][0]
        best_theta = theta[theta == theta.min()][0]
        nf = len(frequency)
        nEpoch = len(self.lc)
        fap = self._pdm_false_alarm(best_theta, nEpoch, self.nbins, nf)
        sig = self._pdm_fap_level(self.false, theta, nEpoch, self.nbins, nf)
        peaks = self._all_frequencies(frequency,1-theta,1-sig) # all peaks above significance.
        all_freq = np.array([frequency[peaks][np.argsort(theta[peaks])], theta[peaks][np.argsort(theta[peaks])]]) # sorted reverse. Bestfreq is the first value.
        if all_freq[0].shape[0] > 0:
            best_freq = all_freq[0][0]
            best_theta = all_freq[1][0]
            fap = self._pdm_false_alarm(best_theta, nEpoch, self.nbins, nf)
        else:
            best_freq = -99.99
            fap = -99.99
        return frequency, theta, best_freq, fap, sig, all_freq


    def CyPDM(self):
        ''' 
        Faster version, written in C, of original PDM from PyAstronomy
        https://github.com/LiyrAstroph/CyPDM       
        '''
        W = self.t[-1] - self.t[0]
        df = 1.0/(self.samples_per_peak * W)
        pdm = cypdm.CyPDM(self.t, self.lc, self.nbins, self.covers)
        frequency = np.arange(self.minfreq, self.maxfreq, df)
        period = 1.0/frequency
        theta = pdm.getPDM_EquiBin(period)
        nf = len(frequency)
        nEpoch = len(self.lc)
        sig = self._pdm_fap_level(self.false, theta, nEpoch, self.nbins, nf)
        peaks = self._all_frequencies(frequency,1-theta,1-sig) # all peaks above significance.
        all_freq = np.array([frequency[peaks][np.argsort(theta[peaks])], theta[peaks][np.argsort(theta[peaks])]]) # sorted reverse. Bestfreq is the first value.
        if all_freq[0].shape[0] > 0:
            best_freq = all_freq[0][0]
            best_theta = all_freq[1][0]
            fap = self._pdm_false_alarm(best_theta, nEpoch, self.nbins, nf)
        else:
            best_freq = -99.99
            fap = -99.99
        return frequency, theta, best_freq, fap, sig, all_freq



    #def _spectral_window(self, x, freq, amplitude=1., phase=0.):
    #    y = 10 + amplitude * np.cos( 2*np.pi*(x*freq + phase) )
    #    yerr = np.abs(y)*0.1
    #    frequency, power, false_alarm, best_freq, sig, all_freq = self.LSG(x,y,yerr)
    #    return frequency, power, sig
    #
    #
    #def _prewhitening(self, x, y, yerr, freq):
    #    fit = fitSin.FitSinSeries(x = x,
    #                              y = y,
    #                              yerr = yerr,
    #                              freq = freq)
    #    params = fit.fourier_params
    #    y_fit = fit.sin_series(x,params)
    #    y_white = y - y_fit
    #    return y_white
    #
    #
    #def _cross_correlation(self, y1, y2):
    #    y1 = y1/y1.max()
    #    y2 = y2/y2.max()
    #    cor_ref = np.correlate(y1,y1)
    #    cor = np.correlate(y1, y2)
    #    return cor, cor_ref
    #
    #
    #def spectral_window_analisys(self):
    #    frequency, power, false_alarm, best_freq, sig, all_freq = self.LSG()
    #    if len(all_freq) != 0:
    #        pdm_params = self.PDM(self.t, self.lc, self.lcErr)
    #        if abs(all_freq[0]/pdm_params[2] - 1) < 0.001:
    #            best_freq = all_freq[0]
    #            spec_freq, spec_power, _ = self._spectral_window(self.t, best_freq)
    #            try:
    #                i=0
    #                while i < len(all_freq):
    #                    freq = all_freq[i]
    #                    y_white = self._prewhitening(self.t, self.lc, self.lcErr, freq)
    #                    y_white_err = np.abs(y_white)*0.1
    #                    new_frequency, new_power, new_false_alarm, new_sig, new_all_freq = self.LSG(self.t, y_white, y_white_err)
    #                    new_best_freq = new_frequency[new_power == new_power.max()][0]
    #                    new_spec_freq, new_spec_power, new_ = self._spectral_window(self.t, new_best_freq)
    #                    cor, cor_ref = self._cross_correlation(spec_power, new_spec_power)
    #                    if cor < cor_ref/2: # improve in future
    #                        best_freq = all_freq[i]
    #                        test = [new_power, new_sig, spec_power, new_spec_power]
    #                        break
    #                    else:
    #                        i += 1
    #            except:
    #                test = []
    #                #print("======> ERROR HAPPENED!")
    #        else:
    #            #print(" ++++ deu ruim ++++ ")
    #            best_freq = -99.99
    #            pdm_params = -99.99
    #            test = []
    #    else:
    #        best_freq = -99.99
    #        pdm_params = -99.99
    #        test = []
    #    return  frequency, power, false_alarm, sig, all_freq, best_freq, test, pdm_params
    #
    #
    #def _do_periodogram(self):
    #    if self.mode == 1:
    #        periodogram_params = self.LSG()
    #    if self.mode == 2:
    #        periodogram_params = self.CyPDM()
    #    if self.mode == 3:
    #        periodogram_params = self.PDM()
    #    #if self.mode == 4:
    #    #    periodogram_params = self.spectral_window_analisys()
    #    return periodogram_params

if __name__ == '__main__':
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import time

    data_table = pd.read_csv("../test/b293_1_z_8_k.ts", index_col="ID") 
    mag_cols = [col for col in data_table.columns if col.split("_")[0] == "MAG"]
    err_cols = [col for col in data_table.columns if col.split("_")[0] == "ERR"]
    selected_stars = pd.read_csv("../test/b293_1_z_8_k.chi2cand", index_col=0).index        
    mag = data_table[mag_cols]
    err = data_table[err_cols]
    x = np.genfromtxt("../test/b293_1_z_8_k.mjd")


    #dat = pd.read_csv("test/data_pdm_test.csv")
    #y = dat["Val-"].values + dat["Val-"].values * 0.01
    #yerr = dat["Val-"].values * 0.001
    #x = dat["Time"].values

    for star in selected_stars[:1]:
        star = "b293_271.09762_-28.93505"
        print("LS: ", star)
        y = mag.loc[star].values
        yerr = err.loc[star].values

        t_ini = time.time()
        per = Periodogram(x=x,
                          y=y,
                          yerr=yerr, 
                          minP=0.1,
                          maxP=1000, 
                          normalization="psd",
                          method="scargle",
                          samples_per_peak=10,
                          mode=False)

        frequency, power, false_alarm, best_freq, fap, sig, all_freq = per.LSG()

        if len(all_freq[0]) > 0:
            per2 = Periodogram(x=x,
                            y=y,
                            yerr=yerr, 
                            minP=0.1,
                            maxP=1000, 
                            normalization="psd",
                            method="scargle",
                            samples_per_peak=10,
                            mode=False)

            frequency2, theta2, best_freq2, fap2, sig2, all_freq2 = per2.CyPDM()
            t_end = time.time()
            print(f"Elapsed time: {t_end-t_ini}")

            if abs(best_freq/best_freq2 - 1) < 0.001:
                print("All Freq: ", all_freq)
                print("PDM FAP: ", sig2)
                print(len(frequency), len(frequency2))
                print("    PLOTED: ", star)
                fig, axs = plt.subplots(3,2, figsize = [16,8])
                fig.suptitle(f'Periodogram Test: {star}')

                #PDM
                axs[0,0].errorbar(x,y,yerr,fmt=".",ms=5,c="k",capsize=2)
                axs[0,0].invert_yaxis()
                axs[0,0].set_xlabel("day/mjd")
                axs[0,0].set_ylabel("Ks/mag")

                axs[1,0].plot(frequency2, theta2, "k", lw=.8)
                axs[1,0].hlines(sig2, xmin=frequency2.max(), xmax=frequency2.min(), ls="dashed", lw=.8, color="r")
                axs[1,0].text(s="FAP level", x=(frequency2.max())*.9, y=sig2 - 0.1, color="r")
                axs[1,0].set_xlabel("frequency [$day^{-1}$]")
                axs[1,0].set_ylabel("PDM theta")
                #axs[1,0].set_xscale("log")
                axs[1,0].set_ylim(-0.05,1.05)

                period = 1./best_freq2
                print("P = ", period)
                shift = (x - min(x))/(2*period)
                phase = shift - (shift).astype(int)

                axs[2,0].plot(phase,   y, ".k", markersize=5)
                axs[2,0].plot(phase+1, y, ".k", markersize=5)
                axs[2,0].invert_yaxis()
                axs[2,0].set_xlabel(f"Phase [Period: {round(period,5)} days, fap: {fap2*100} %]")
                axs[2,0].set_ylabel("Ks/mag")


                #LSG
                axs[0,1].errorbar(x,y,yerr,fmt=".",ms=5,c="k",capsize=2)
                axs[0,1].invert_yaxis()
                axs[0,1].set_xlabel("day/mjd")
                axs[0,1].set_ylabel("Ks/mag")

                axs[1,1].plot(frequency, power, "k",lw=.8)
                axs[1,1].hlines(sig, xmin=frequency.max(), xmax=frequency.min(), ls="dashed", lw=.8, color="r")
                axs[1,1].text(s="FAP level", x=(frequency.max())*.9, y=sig + sig*.01, color="r")
                axs[1,1].set_xlabel("frequency [$day^{-1}$]")
                axs[1,1].set_ylabel("LSG power")
                #axs[1,1].set_xscale("log")

                period = 1./best_freq
                shift = (x - min(x))/(2*period)
                phase = shift - (shift).astype(int)
                
                axs[2,1].plot(phase,   y, ".k", markersize=5)
                axs[2,1].plot(phase+1, y, ".k", markersize=5)
                axs[2,1].invert_yaxis()
                axs[2,1].set_xlabel(f"Phase [Period: {round(period,5)} days]")
                axs[2,1].set_ylabel("Ks/mag")

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                #plt.savefig(f"../test/{star}a.png",dpi=300)
                plt.show()
                plt.close()
        
        