# -*- coding: utf-8 -*-
'''
Created on Jun 24 2020

@author: Everton Botan
@supervisor: Roberto Saito

This module have tools to work with classifyed RR Lyrae.

'''

import os
import sys
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt


# my modules:
sys.path.append('/home/botan/OneDrive/Doutorado/VVV_DATA/my_modules/')
import match
import fit_sin_series as fitSin
import periodogram as pg
import status


class RRLyrTools(object):

    def __init__(self):
        self.main_path = f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/psf_ts'
        self.tiles = sorted(os.listdir(self.main_path))
        #self.chips = [fn[:-3] for fn in sorted(os.listdir(f'{self.main_path}/{tile}/chips')) if fn.endswith('ts')]

    def read_data(self):
        self.rrl_data = pd.read_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/rrl_dat.csv', index_col='ID')
        self.extintion_data = pd.read_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/all_variables_extintion.csv',index_col='ID')
    
    def _do_match(self,c1,ra_u1,c2,ra_u2,max_sep):
        '''invoque match object and give back pandas indexes for both tables
        c1 is the catalog
        c2 is your stars to match
        '''
        c1 = c1[['RA','DEC']]
        c2 = c2[['RA','DEC']]
        bm = match.Match()
        idx1, idx2, sep2d = bm.match(c1 = c1.values, ra_u1 = ra_u1, 
                                     c2 = c2.values, ra_u2 = ra_u2,
                                     max_sep = max_sep)
        id1 = c1.index[idx1]
        id2 = c2.index[idx2]
        return id1, id2, sep2d

    ''' These equations for absolute magnitudes are from Alonso-García (2015): 
            https://iopscience.iop.org/article/10.1088/0004-6256/149/3/99 '''
    def M_ks(self,period,metallicity):
        abs_ks = - 0.6365 - 2.347*np.log10(period) + 0.1747*np.log10(metallicity)
        return abs_ks

    def M_H(self,period,metallicity):
        abs_h = - 0.5539 - 2.302*np.log10(period) + 0.1781*np.log10(metallicity)
        return abs_h

    def M_J(self,period,metallicity):
        abs_j = - 0.2361 - 1.830*np.log10(period) + 0.1886*np.log10(metallicity)
        return abs_j
    
    def M_Y(self,period,metallicity):
        abs_y = 0.0090 - 1.467*np.log10(period) + 0.1966*np.log10(metallicity)
        return abs_y
    
    def M_Z(self,period,metallicity):
        abs_z = 0.1570 - 1.247*np.log10(period) + 0.2014*np.log10(metallicity)
        return abs_z

    def M_ks2(self,period,FeH):
        # Muraveva (2015) https://iopscience.iop.org/article/10.1088/0004-637X/807/2/127
        abs_ks = - 2.53*np.log10(period) -0.95 + 0.07* FeH
        return abs_ks

    def color_excess(self, mag1, mag2, abs_mag1, abs_mag2):
        colExc = (mag1 - mag2) - (abs_mag1 - abs_mag2)
        return colExc

    def reddening_correction(self,mag,k,colExc):
        # k is passband and extintion law deppendent. 
        # You may use Cardelli k = 1.692 for J; 1.054 for H; and 0.689 for Ks
        # colExc may be calculated by color_excess function or provided by BEAM calculator.
        mag_0 =  mag - k*colExc
        return mag_0

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

    def _normalize(self, num, lower=0.0, upper=360.0, b=False):
        #https://gist.github.com/phn/1111712/35e8883de01916f64f7f97da9434622000ac0390
        """Normalize number to range [lower, upper) or [lower, upper].
        Parameters
        ----------
        num : float
            The number to be normalized.
        lower : float
            Lower limit of range. Default is 0.0.
        upper : float
            Upper limit of range. Default is 360.0.
        b : bool
            Type of normalization. See notes.
        Returns
        -------
        n : float
            A number in the range [lower, upper) or [lower, upper].
        Raises
        ------
        ValueError
        If lower >= upper.
        Notes
        -----
        If the keyword `b == False`, the default, then the normalization
        is done in the following way. Consider the numbers to be arranged
        in a circle, with the lower and upper marks sitting on top of each
        other. Moving past one limit, takes the number into the beginning
        of the other end. For example, if range is [0 - 360), then 361
        becomes 1. Negative numbers move from higher to lower
        numbers. So, -1 normalized to [0 - 360) becomes 359.
        If the keyword `b == True` then the given number is considered to
        "bounce" between the two limits. So, -91 normalized to [-90, 90],
        becomes -89, instead of 89. In this case the range is [lower,
        upper]. This code is based on the function `fmt_delta` of `TPM`.
        Range must be symmetric about 0 or lower == 0.
        Examples
        --------
        >>> normalize(-270,-180,180)
        90
        >>> import math
        >>> math.degrees(normalize(-2*math.pi,-math.pi,math.pi))
        0.0
        >>> normalize(181,-180,180)
        -179
        >>> normalize(-180,0,360)
        180
        >>> normalize(36,0,24)
        12
        >>> normalize(368.5,-180,180)
        8.5
        >>> normalize(-100, -90, 90, b=True)
        -80.0
        >>> normalize(100, -90, 90, b=True)
        80.0
        >>> normalize(181, -90, 90, b=True)
        -1.0
        >>> normalize(270, -90, 90, b=True)
        -90.0
        """
        
        from math import floor, ceil
        # abs(num + upper) and abs(num - lower) are needed, instead of
        # abs(num), since the lower and upper limits need not be 0. We need
        # to add half size of the range, so that the final result is lower +
        # <value> or upper - <value>, respectively.
        res = num
        if not b:
            if lower >= upper:
                raise ValueError("Invalid lower and upper limits: (%s, %s)" %
                                (lower, upper))

            res = num
            if num > upper or num == lower:
                num = lower + abs(num + upper) % (abs(lower) + abs(upper))
            if num < lower or num == upper:
                num = upper - abs(num - lower) % (abs(lower) + abs(upper))

            res = lower if res == upper else num
        else:
            total_length = abs(lower) + abs(upper)
            if num < -total_length:
                num += ceil(num / (-2 * total_length)) * 2 * total_length
            if num > total_length:
                num -= floor(num / (2 * total_length)) * 2 * total_length
            if num > upper:
                num = total_length - num
            if num < lower:
                num = -total_length - num

            res = num * 1.0  # Make all numbers float, to be consistent
        return res

    def r2r(self,r):
        """Normalize angle in radians to [0, 2π)."""
        norm = self._normalize(r, 0, 2 * math.pi)
        #norm = self._phase_wrap(r,0,2*np.pi)
        return norm


    def _phase_wrap(self,num,lower,upper): 
        if lower >= upper: 
            raise ValueError(f'Invalid lower and upper limits: {lower}, {upper}') 
        if num < lower: 
            num = upper - abs(num - lower) % abs(upper - lower) 
        if num > upper: 
            num = lower + abs(num - upper) % abs(upper - lower) 
        res = lower if num == upper else num 
        return res 


    def _period_phase_metalicity(self,calibrate=False):
        # this is a subrotine to calibrate a period-phase-[Fe/H] relationship using OGLE.
        if calibrate:
            self.read_data()
            ogle_data = pd.read_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/ogle_iv/ident.dat', index_col=0)
            ogle_ids, vvv_ids, sep = self._do_match(c1=ogle_data,ra_u1='deg',c2=self.rrl_data,ra_u2='deg',max_sep=1)
            rrab_dat = pd.read_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/ogle_iv/RRab.dat', index_col=0)
            rrc_dat = pd.read_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/ogle_iv/RRc.dat', index_col=0)
            ogle_Z = []
            for n in range(len(ogle_ids)):
                ogle_id = ogle_ids[n]
                vvv_id = vvv_ids[n]
                rrType = ogle_data.loc[ogle_id].type
                ra = ogle_data.loc[ogle_id].RA
                dec = ogle_data.loc[ogle_id].DEC

                if rrType == 'RRab':
                    # Jurcsik & kovács (1996): https://ui.adsabs.harvard.edu/abs/1996A%26A...312..111J/abstract
                    # phi31 from I to V - linear relation: phi31_V = -0.512 + 1.059 * phi31_I: https://ui.adsabs.harvard.edu/abs/1998AcA....48..341M/abstract
                    fpath = f'/home/botan/OneDrive/Doutorado/VVV_DATA/ogle_iv/ogle_iv_rrlyr_V/{ogle_id}.dat'
                    if os.path.exists(fpath):
                        lc_dat = np.loadtxt(fpath)
                        t = lc_dat.T[0]
                        lc = lc_dat.T[1]
                        lcErr = lc_dat.T[2]
                        freq = 1./rrab_dat.loc[ogle_id].P

                        if len(np.array(lc)) > 25:
                            fit = self._curve_fit_switch(t,lc,lcErr,freq,
                                                        order=8,
                                                        phaseSpace=False,
                                                        fitFreq=False,
                                                        iterative=False,
                                                        fitastrobase=False)
                            if fit != 0:
                                fitpar = fit['fitparams']
                                fitparerrs = fit['fitparamserrs']
                                fit_parameters  = [vvv_id,ra,dec,ogle_id,freq]
                                fit_params_cols = ['ID','RA','DEC','OGLE_ID','OGLE_Freq']
                                n=0 
                                k=1
                                while n < len(fitpar[0]):
                                    fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                                    fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                                    n+=1
                                    k+=1


                                phi31_V = self.r2r(fitpar[1][2] - 3 * fitpar[1][0])
                                #phi31_V = 0.115 + 0.682 * self.r2r(phi31_I)
                                p = rrab_dat.loc[ogle_id].P
                                FeH = - 5.038 - 5.394 * p + 1.345 * phi31_V
                                if FeH < -4:
                                    FeH = - 5.038 - 5.394 * p + 1.345 * (phi31_V + np.pi)
                                #FeH = - 0.08126 - 5.394 * p + 1.345 * phi31_V
                                #phi31 = -0.512 + 1.059* (rrab_dat.loc[ogle_id].phi_31)
                                #p = rrab_dat.loc[ogle_id].P
                                #FeH = 4.86*(np.log10(p))**2 + 0.0183*(phi31)**2 - 0.820*(np.log10(p)) * phi31 - 4.260
                                
                                print('[FE/H]:',FeH)

                                mags  = fit['magseries']['mags']
                                phase = fit['magseries']['phase']
                                fitmags = fit['magseries']['fitmags']
                                
                                plt.figure()
                                plt.plot(phase,mags,'k.')
                                plt.plot(phase,fitmags,'r-')
                                plt.savefig(f'figtest/{ogle_id}.png')
                                plt.close()

                                rrab_dat.loc[ogle_id,'VVV_ID'] = vvv_id
                                rrab_dat.loc[ogle_id,'type'] = 'RRab'
                                rrab_dat.loc[ogle_id,'[Fe/H]'] = FeH

                                #ogle_Z.append(pd.Series(fit_parameters,index=fit_params_cols))

                if rrType == 'RRc':
                    # Morgan et.al. (2007, 2013): https://ui.adsabs.harvard.edu/abs/2014IAUS..301..461M/abstract
                    fpath = f'/home/botan/OneDrive/Doutorado/VVV_DATA/ogle_iv/ogle_iv_rrlyr_I/{ogle_id}.dat'
                    if os.path.exists(fpath):

                        lc_dat = np.loadtxt(fpath)
                        t = lc_dat.T[0]
                        lc = lc_dat.T[1]
                        lcErr = lc_dat.T[2]
                        freq = 1./rrc_dat.loc[ogle_id].P
                        if len(np.array(lc)) > 25:
                            fit = self._curve_fit_switch(t,lc,lcErr,freq,
                                                        order=8,
                                                        phaseSpace=False,
                                                        fitFreq=False,
                                                        iterative=False,
                                                        fitastrobase=False)
                            if fit != 0:
                                fitpar = fit['fitparams']
                                fitparerrs = fit['fitparamserrs']
                                fit_parameters  = [vvv_id,ra,dec,ogle_id,freq]
                                fit_params_cols = ['ID','RA','DEC','OGLE_ID','OGLE_Freq']
                                n=0 
                                k=1
                                while n < len(fitpar[0]):
                                    fit_parameters += [fitpar[0][n],fitparerrs[0][n],fitpar[1][n],fitparerrs[1][n]]
                                    fit_params_cols += [f'A{k}',f'AERR{k}',f'PHI{k}',f'PHIERR{k}']
                                    n+=1
                                    k+=1
                                phi31_V = self.r2r(fitpar[1][2] - 3 * fitpar[1][0])
                                #phi31_V = -0.512 + 1.059 * self.r2r(phi31_I)
                                p = rrc_dat.loc[ogle_id].P
                                FeH = 0.0348 * phi31_V**2 + 0.196 * phi31_V -8.507 * p + 0.367
                                #FeH = 4.86*(np.log10(p))**2 + 0.0183*(phi31_V)**2 - 0.820*(np.log10(p)) * phi31_V - 4.260
                                rrc_dat.loc[ogle_id,'VVV_ID'] = vvv_id
                                rrc_dat.loc[ogle_id,'type'] = 'RRc'
                                rrc_dat.loc[ogle_id,'[Fe/H]'] = FeH
                #else:
                #    raise RuntimeError(f'There is no law for rrType: {rrType}. Stopping!')
            rrab_dat.to_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/ogle_iv/RRab.dat')
            rrc_dat.to_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/ogle_iv/RRc.dat')
            ogle_Z = pd.concat([rrab_dat.loc[[_ for _ in ogle_ids if _ in rrab_dat.index]],rrc_dat.loc[[_ for _ in ogle_ids if _ in rrc_dat.index]]],axis=0)
            ogle_Z.to_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/OGLE_metalicity.dat')
        else:
            ogle_Z = pd.read_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/OGLE_metalicity.dat', index_col='VVV_ID')
            






    def rrab_metalicity(self,p, mode, rrType):
        # Mode 0: 
        # Mode 1: Feast (2010): https://academic.oup.com/mnrasl/article/408/1/L76/1008478
        if mode == 0:
            FeH = -7.82*np.log10(p) - 3.43 
        if mode == 1:
            FeH = -5.62*np.log10(p) - 2.81
        if mode == 2:
            FeH = self._period_phase_metalicity()
        else:
            raise RuntimeError(f'There is no mode: {mode}. Stopping!')
        return FeH


    def rrl_distance(self,Ks_0,M_Ks):
        dist = 10**(1 + (Ks_0 - M_Ks)/5) #in pc
        return dist

    def calculate_distance(self):
        self.rrl_dist = []
        self.read_data()
        rrl_ids = self.rrl_data.index
        rrl_p = self.rrl_data.period
        rrl_AKs = 0.689 * self.extintion_data.loc[rrl_ids].E_JK #http://mill.astro.puc.cl/BEAM/coffinfo.php
        Z = 0.0025
        FeH = -1.23
        FeHlist = []
        # Z = 0.0025, value for the globular cluster 2MASS-GC 02. Alonso-García (2015, p. 10): https://iopscience.iop.org/article/10.1088/0004-6256/149/3/99
        # We need to get this from somewere else. BEAM? 
        for star in rrl_ids:
            p      = rrl_p.loc[star]
            Ks_mag = self.rrl_data.loc[star].mag_Ks
            AKs    = rrl_AKs.loc[star]
            #FeH = self.rrab_metalicity(p)
            #FeHlist.append(FeH)
            #M_Ks   = self.M_ks(p,Z)
            M_Ks   = self.M_ks2(p,FeH)
            Ks_0   = self.reddening_correction(Ks_mag,AKs)

            dist   = self.rrl_distance(Ks_0,M_Ks)
            self.rrl_dist.append(dist)
            print(star, dist)
        print(np.mean(FeHlist))
        plt.hist(self.rrl_dist,bins=40)
        plt.axvline(x=8000,c='r')
        plt.savefig("mygraph.png")
        plt.close()

if __name__ == '__main__':
    rr = RRLyrTools()
    #rr.calculate_distance()
    rr._period_phase_metalicity(calibrate=True)