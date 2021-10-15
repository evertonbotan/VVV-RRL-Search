# -*- coding: utf-8 -*-
"""
Created on Nov 04 2020

@author: Everton Botan
@supervisor: Roberto Saito

Calculate distance for VVV RR Lyrae.
"""
import os
import sys
import copy
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.signal import find_peaks
import scipy

# Use LaTex fonts
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams.update({'font.size': 12})

# set comma to dot - Brazilian decimal notation
import locale
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
locale.setlocale(locale.LC_NUMERIC, 'pt_BR.UTF-8')
import matplotlib as mpl
mpl.rcParams['axes.formatter.use_locale'] = True

# my library
sys.path.append('/home/botan/OneDrive/Doutorado/VVV_DATA/my_modules/')
import math_functions
import red_clump_tools as rct

class RRLTools(object):
    
    def __init__(self,gc_distance):
        self.gc_distance = gc_distance
        self.path = '/home/botan/OneDrive/Doutorado/VVV_DATA'


    def LCmagKs(self):
        # calculate mean Ks magnitude
        tiles = sorted(os.listdir(f'{self.path}/data/psf_ts/'))
        for tile in tiles:
            print(f'Working on tile {tile}. Wait...')
            chips = [_[:-3] for _ in os.listdir(f'{self.path}/data/psf_ts/{tile}/chips/') if _.endswith('.ts')]
            for chip in chips:
                chipData = pd.read_csv(f'{self.path}/data/psf_ts/{tile}/chips/{chip}.ts',index_col='ID')
                magCols  = [_ for _ in chipData.columns if _[:3] == 'MAG']
                errCols  = [_ for _ in chipData.columns if _[:3] == 'ERR']
                err_msk  = (chipData[errCols] > 0.2).values
                nEpoch   = (~chipData[errCols].mask(err_msk).isnull()).sum(axis=1)
                ks_mag   = chipData[magCols].mask(err_msk).sum(axis=1) / nEpoch
                ks_err   = np.sqrt((chipData[errCols].mask(err_msk)**2).sum(axis=1)) / nEpoch
                for star in self.all_dat.index:
                    if star[:-20] == chip:
                        self.all_dat.loc[star,'lc_mean']     = ks_mag.loc[star]
                        self.all_dat.loc[star,'lc_mean_err'] = ks_err.loc[star]
            print('    --> Done')
        self.all_dat.to_csv(f'{self.path}/data/all_variables_match_vsx_ogle_gaia_viva2.csv',sep=',')

    def psf_color(self):
        tiles = sorted(os.listdir(f'{self.path}/data/psf_ts/'))
        for tile in tiles:
            print(f'Working on tile {tile}. Wait...')
            chips = [_.split('.')[0] for _ in os.listdir(f'{self.path}/data/psf_ts/{tile}/chips/') if _.endswith('.ts')]
            for chip in chips:
                chipData = pd.read_csv(f'{self.path}/data/psf_ts/{tile}/chips/{chip}.ts',index_col='ID')
                magCols  = [_ for _ in chipData.columns if _.split("_")[0] == 'mag']
                errCols  = [_ for _ in chipData.columns if _.split("_")[0] == 'er']
                for star in self.all_dat.index:
                    if star[:-20] == chip:
                        self.all_dat.loc[star,magCols] = chipData.loc[star,magCols]
                        self.all_dat.loc[star,errCols] = chipData.loc[star,errCols]
            print('    --> Done')
        cols = ['RA', 'DEC', 'mag_Ks', 'mag_J','mag_Z', 'mag_Y','lc_mean', 'lc_mean_err',
       'mag_H', 'er_Z', 'er_Y', 'er_J', 'er_H', 'er_Ks', 'period', 'amplitude', 'dup_id1',
       'dup_id2', 'dup_id3', 'OGLE_ID', 'OGLE_Type', 'OGLE_Subtype', 'OGLE_I',
       'OGLE_V', 'OGLE_P', 'OGLE_A', 'VSX_ID', 'VSX_Name', 'VSX_Type', 'VSX_P',
       'GAIA_ID', 'GAIA_AngDist', 'GAIA_Parallax', 'GAIA_Parallax_ERROR',
       'GAIA_PMRA', 'GAIA_PMRA_ERROR', 'GAIA_PMDEC', 'GAIA_PMDEC_ERROR',
       'gal_l', 'gal_b',  'rest', 'b_rest_x',
       'B_rest_xa', 'rlen', 'ResFlag', 'vivaID', 'vivaP']
        #self.all_dat.to_csv(f'{self.path}/data/all_variables_match_vsx_ogle_gaia_viva2.csv',sep=',')
        return self.all_dat[cols]

    def read_data(self):
        tiles = sorted(os.listdir(f'{self.path}/data/psf_ts/'))
        duplicates = pd.read_csv(f'{self.path}/data/chip_overlap_ids.csv',index_col=0)
        
        rrl_fitparams = []
        for tile  in tiles:
            path = f'{self.path}/data/psf_ts/{tile}/lc_plots/short_period/pos_visual_inspection'
            rrl_fitparams.append(pd.read_csv(f'{path}/{tile}_rrlyr_bona_parameters.csv',sep=',',index_col='ID'))
        rrl_fitparams = pd.concat(rrl_fitparams)
        magCols = [_ for _ in rrl_fitparams.columns if _[:3] == 'mag']
        errCols = [_ for _ in rrl_fitparams.columns if _[:3] == 'er_']
        id2drop = []
        for _ in rrl_fitparams.index: 
            if _ in duplicates.index: 
                for col in duplicates.columns: 
                    star2drop = duplicates.loc[_,col] 
                    if star2drop not in id2drop: 
                        id2drop.append(star2drop)
        self.rrl_ids = [_ for _ in rrl_fitparams.index if _ not in id2drop]
        self.rrl_fitparams = rrl_fitparams.loc[self.rrl_ids].fillna(-99)
        self.BEAM_extintion = pd.read_csv(f'{self.path}/data/all_variables_extintion_.csv',index_col='ID')
        self.extintion3D = pd.read_csv(f'{self.path}/3D_Extintion_Map/table1jk.csv')
        self.all_dat = pd.read_csv(f'{self.path}/data/all_variables_match_vsx_ogle_gaia_viva2_.csv',index_col='ID')
        

    def Fe_abundance(self,period,mode=1):
        # mode = 1: Sarajedini (2006) https://ui.adsabs.harvard.edu/abs/2006AJ....132.1361S/abstract
        # mode = 1: Feast (2010) https://ui.adsabs.harvard.edu/abs/2010MNRAS.408L..76F/abstract
        if mode==1:
            FeH = -7.82*np.log10(period) - 3.43
            sigma = 0.45
        if mode==2:
            FeH = -5.62*np.log10(period) - 2.81
            sigma = 0.42
        return FeH,sigma

    def metallicity_from_Fe_abundance(self,abundance,sigma,alpha=0.3):
        z = 10**(abundance + np.log10(0.638*10**alpha + 0.362) - 1.765)
        z_sigma = 10**(abundance + np.log10(0.638*10**alpha + 0.362) - 1.765) * np.log(10) * sigma
        return z,z_sigma

    def M_Ks(self,period,metallicity,offset=0):
        abs_ks = - 0.6365 - 2.347*np.log10(period) + 0.1747*np.log10(metallicity) + offset
        # this theoretical absolute magnitude has errors bellow survey photometric precision.
        # Thus it has been ignored.
        return abs_ks

    def M_Ks2(self,period,FeH,offset=0):
        # This is PL relationship (equation 5) from Muraveva et. al. (2015) AJ 807:127
        abs_ks = -1.27 - 2.73*np.log10(period) + 0.03*FeH + offset
        return abs_ks

    def M_H(self,period,metallicity,offset=0):
        abs_h = - 0.5539 - 2.302*np.log10(period) + 0.1781*np.log10(metallicity) + offset
        return abs_h

    def M_J(self,period,metallicity,offset=0):
        abs_j = - 0.2361 - 1.830*np.log10(period) + 0.1886*np.log10(metallicity) + offset
        return abs_j
    
    def M_Y(self,period,metallicity,offset=0):
        abs_y = 0.0090 - 1.467*np.log10(period) + 0.1966*np.log10(metallicity) + offset
        return abs_y
    
    def M_Z(self,period,metallicity,offset=0):
        abs_z = 0.1570 - 1.247*np.log10(period) + 0.2014*np.log10(metallicity) + offset
        return abs_z

    def extintion(self, magA, magAerr, magB, magBerr, abs_magA, abs_magB):
        #color excess
        EJKs = (magA - magB) - (abs_magA - abs_magB)
        sigma     = abs(np.sqrt(magAerr**2 + magBerr**2))
        return EJKs,sigma

    def reddenig(self,extintion,extintionSigma,redIndex):
        red   = redIndex*extintion
        sigma = abs(redIndex*extintionSigma)
        return red,sigma

    def red_free_mag(self,mag,err,red_mag,red_err):
        mag_0 = mag - red_mag
        sigma = abs(np.sqrt(err**2 + red_err**2))
        return mag_0,sigma

    def dist_module(self,Ks_0,ErrKs,AbsKs):
        dist  = 10**(1 + (Ks_0 - AbsKs)/5) #in parsec
        sigma = abs(2**(1 + (Ks_0 - AbsKs)/5) * 5**((Ks_0 - AbsKs)/5) * np.log(10) * ErrKs)
        return dist,sigma

    def calc_distances_color(self,lcmean,lcmean_err,MagKs,ErrKs,MagJ,ErrJ,MagH,ErrH,
                      beam_E_JK,beam_E_JK_err,period,
                      abundanceMode=1,magoffset=0):
        # MagKs is the Ks mean magnitude from light curve
        # MagJ and MagH are the magnitudes from color campain
        FeH, FeH_sigma = self.Fe_abundance(period, mode=abundanceMode)
        Z, Z_sigma     = self.metallicity_from_Fe_abundance(FeH,FeH_sigma,alpha=0.3)
        AbsKs          = self.M_Ks(period,Z,offset=magoffset)
        #AbsKs          = self.M_Ks2(period,FeH,offset=magoffset)
        AbsJ           = self.M_J(period,Z,offset=magoffset)
        AbsH           = self.M_H(period,Z,offset=magoffset)
        E_HKs          = np.nan
        E_HKS_sigma    = np.nan
        E_JKs          = np.nan
        E_JKS_sigma    = np.nan
        if MagKs > 0:
            if MagJ > 0:
                # if J magnitude is availeable we calculate extintion
                # from the difference between observed and intrinsec magnitude
                redIndex           = 0.689 # Cardelli
                #redIndex           = 0.398#0.464# 0.689 Alonso-García (2017) https://iopscience.iop.org/article/10.3847/2041-8213/aa92c3
                #redIndex           = 0.575 # mine, obteined by RC in tile b309
                E_JKs, E_JKS_sigma = self.extintion(MagJ,ErrJ,MagKs,ErrKs,AbsJ,AbsKs)
                AKs, AKs_sigma     = self.reddenig(E_JKs,E_JKS_sigma,redIndex=redIndex)
                extintion_flag     = 1
            else:
                if MagH > 0:
                    # if J is missing but not H we calculate extintion
                    # from the difference between observed and intrinsec magnitude
                    redIndex           = 1.888 # Cardeli
                    #redIndex           = 1.30#1.888 Alonso-García (2017) https://iopscience.iop.org/article/10.3847/2041-8213/aa92c3
                    #redIndex           = 1.04 # mine, obteined by RC in tile b309
                    E_HKs, E_HKS_sigma = self.extintion(MagH,ErrH,MagKs,ErrKs,AbsH,AbsKs)
                    AKs, AKs_sigma     = self.reddenig(E_HKs,E_HKS_sigma,redIndex=redIndex)
                    extintion_flag     = 2
                else:
                    # if J and H magnitude is missing we used BEAM.
                    redIndex       = 0.689 # Cardeli
                    #redIndex       = 0.575 # mine, obteined by RC in tile b309
                    E_JKs          = beam_E_JK
                    E_JKS_sigma    = beam_E_JK_err
                    AKs,AKs_sigma  = self.reddenig(E_JKs,E_JKS_sigma,redIndex=redIndex)
                    extintion_flag = 3
        else:
            # if J and H magnitude is missing we used BEAM.
            redIndex       = 0.689 # Cardeli
            #redIndex       = 0.575 # mine, obteined by RC in tile b309
            E_JKs          = beam_E_JK
            E_JKS_sigma    = beam_E_JK_err
            AKs,AKs_sigma  = self.reddenig(E_JKs,E_JKS_sigma,redIndex=redIndex)
            extintion_flag = 3
        
        Ks_0, Ks_0_err   = self.red_free_mag(lcmean,lcmean_err,AKs,AKs_sigma)
        dist, dist_err   = self.dist_module(Ks_0,Ks_0_err,AbsKs)
        return [dist, dist_err, FeH, FeH_sigma, Z, Z_sigma, AbsKs, AbsJ, AbsH, E_HKs, E_HKS_sigma, E_JKs, E_JKS_sigma, AKs, AKs_sigma, extintion_flag]


    def calc_distances_BEAM(self,lcmean,lcmean_err,MagKs,ErrKs,MagJ,ErrJ,MagH,ErrH,
                            beam_E_JK,beam_E_JK_err,period,
                            abundanceMode=1,magoffset=0):
        # MagKs is the Ks mean magnitude from light curve
        # MagJ and MagH are the magnitudes from color campain
        FeH, FeH_sigma = self.Fe_abundance(period, mode=abundanceMode)
        Z, Z_sigma     = self.metallicity_from_Fe_abundance(FeH,FeH_sigma,alpha=0.3)
        AbsKs          = self.M_Ks(period,Z,offset=magoffset)
        #AbsKs          = self.M_Ks2(period,FeH,offset=magoffset)
        AbsJ           = self.M_J(period,Z,offset=magoffset)
        AbsH           = self.M_H(period,Z,offset=magoffset)
        E_HKs          = np.nan
        E_HKS_sigma    = np.nan
        E_JKs          = np.nan
        E_JKS_sigma    = np.nan

        redIndex       = 0.689 # Cardeli
        #redIndex       = 0.575 # mine, obteined by RC in tile b309
        E_JKs          = beam_E_JK
        E_JKS_sigma    = beam_E_JK_err
        AKs,AKs_sigma  = self.reddenig(E_JKs,E_JKS_sigma,redIndex=redIndex)
        extintion_flag = 3
        Ks_0, Ks_0_err   = self.red_free_mag(lcmean,lcmean_err,AKs,AKs_sigma)
        dist, dist_err   = self.dist_module(Ks_0,Ks_0_err,AbsKs)
        return [dist, dist_err, FeH, FeH_sigma, Z, Z_sigma, AbsKs, AbsJ, AbsH, E_HKs, E_HKS_sigma, E_JKs, E_JKS_sigma, AKs, AKs_sigma, extintion_flag]


    def cartezian_projections(self,d,gal_l,gal_b):
        dx = d*np.cos(math.radians(gal_b))*np.cos(math.radians(gal_l))
        rx = dx - self.gc_distance
        ry = d*np.cos(math.radians(gal_b))*np.sin(math.radians(gal_l))
        rz = d*np.sin(math.radians(gal_b))
        return rx,ry,rz


    def get_distance(self,magoffset,campain='variability',method='color'):
        # setting campain to variability uses lc mean for Ks
        # setting campain to color uses Ks from color campain
        # seting method to color, it uses color and BEAM to get extintion
        # seting method to BEAM, it uses only BEAM to get extintion
        dist_table = pd.DataFrame()
        for star in self.rrl_ids:
            #if self.rrl_fitparams.loc[star].mag_Ks != -99:
            ra            = self.all_dat.loc[star,'RA']
            dec           = self.all_dat.loc[star,'DEC']
            gal_l         = self.all_dat.loc[star,'gal_l']
            gal_b         = self.all_dat.loc[star,'gal_b']
            period        = 1./self.rrl_fitparams.loc[star].Freq
            if campain == 'variability':
                lcmean     = self.all_dat.loc[star].lc_mean
                lcmean_err = self.all_dat.loc[star].lc_mean_err
            elif campain == 'color':
                lcmean     = self.all_dat.loc[star].mag_Ks
                lcmean_err = self.all_dat.loc[star].er_Ks
            else:
                raise ValueError(f'{campain} is not a valid setting.')
            MagKs         = self.all_dat.loc[star].mag_Ks
            ErrKs         = self.all_dat.loc[star].er_Ks
            MagH          = self.all_dat.loc[star].mag_H
            ErrH          = self.all_dat.loc[star].er_H
            MagJ          = self.all_dat.loc[star].mag_J
            ErrJ          = self.all_dat.loc[star].er_J
            beam_E_JK     = self.BEAM_extintion.loc[star].E_JK
            beam_E_JK_err = self.BEAM_extintion.loc[star].SigmaE_JK
            if method=='color':
                params = self.calc_distances_color(lcmean,lcmean_err,MagKs,ErrKs,MagJ,ErrJ,MagH,ErrH,
                                            beam_E_JK,beam_E_JK_err,period,
                                            abundanceMode=1,magoffset=magoffset)
            if method=='BEAM':                     
                params = self.calc_distances_BEAM(lcmean,lcmean_err,MagKs,ErrKs,MagJ,ErrJ,MagH,ErrH,
                                            beam_E_JK,beam_E_JK_err,period,
                                            abundanceMode=1,magoffset=magoffset)
            x,y,z = self.cartezian_projections(params[0],gal_l,gal_b)
            cols = ['RA','DEC','gal_l','gal_b','x','y','z',
                    'VVVtype','distance','distanceSigma',
                    '[Fe/H]','[Fe/H]_err','Z','Z_err',
                    'M_Ks','M_J','M_H','E(H-Ks)','E(H-KS)_err',
                    'E(J-Ks)','E(J-KS)_err',
                    'AKs','AKs_err','ExtintionFlag']
            dist_table.loc[star,cols] = [ra,dec,gal_l,gal_b,x,y,z]+['RRL']+params
        return dist_table
    



#======== RED CLUMP PEAKS ========#
    def get_RC_peaks(self,Rv):
        rc_tools = rct.RedClump(Rv)
        rc_peaks = rc_tools.find_RC_peaks(plot=False,show=False)
        return rc_peaks

    def plot_RC_CMD(sels,Rv):
        rc_tools = rct.RedClump(Rv)
        rc_cmd = rc_tools.red_clump_inclination(method='2gaussian',plotHist=False)



if __name__ == "__main__":
    import importlib
    importlib.reload(sys.modules['math_functions'])
    path = '/home/botan/OneDrive/Doutorado/VVV_DATA'
    GC_dist = 8178 # +- 13 pc https://www.aanda.org/articles/aa/full_html/2019/05/aa35656-19/aa35656-19.html
    d = RRLTools(gc_distance=GC_dist)
    d.read_data()
    rrl_ids = d.rrl_ids
    rrl_dat = d.all_dat.loc[rrl_ids]
    distances = d.get_distance(magoffset=0,campain='variability',method='color')
    distances_BEAM = d.get_distance(magoffset=0,campain='variability',method='BEAM')
    ogle_bulge_rrl = pd.read_csv('/home/botan/OneDrive/Doutorado/VVV_DATA/ogle_iv_bulge/dat/RRab.csv',index_col=0)
    ogle_dat = pd.read_csv('/home/botan/OneDrive/Doutorado/VVV_DATA/data/ogle_dat.csv',index_col=0)
    ogle_rrl = ogle_dat[ogle_dat.Subtype == 'RRab']

    rrl_names = {}
    for n,star in enumerate(rrl_ids):
        rrl_names[star]= '%s-RRL-%04d'%(star.split('_')[0],n)


    # Red Clump distance, color and magnitude peaks.
    Rv = 0.689
    rc_peaks_path = f'{path}/data/red_clump_peaks_{Rv}.csv'
    if not os.path.exists(rc_peaks_path):
        Rc_peaks = d.get_RC_peaks(Rv=Rv)
        Rc_peaks.to_csv(rc_peaks_path)
    else:
        Rc_peaks = pd.read_csv(rc_peaks_path, index_col=0)
    

    # Red Clump Peaks cartesian projection
    for tile in Rc_peaks.index:
        gal_l = Rc_peaks.loc[tile,'tile_central_l']
        gal_b = Rc_peaks.loc[tile,'tile_central_b']
        RC_dist = Rc_peaks.loc[tile,'RC_peak1_dist']
        RC_dist_err = Rc_peaks.loc[tile,'RC_peak1_dist_sigma']
        rx,ry,rz = d.cartezian_projections(RC_dist,gal_l,gal_b)
        rxErr,ryErr,rzErr = d.cartezian_projections(RC_dist_err,gal_l,gal_b)

    









    # globular clusters from: https://www.physics.mcmaster.ca/~harris/mwgc.dat
    # aglomerados 'NAME':(l,b,d(kpc),r_c(arcmin),r_h(arcmin),[Fe/H])
    glob_clusters = {'NGC 6540':(3.28510,-3.31300,5.3,0.03,-999,-1.35),
                    'NGC 6544' :(5.83815,-2.20354,3.0,0.05,1.21,-1.40),
                    'NGC 6553' :(5.25024,-3.02296,6.0,0.53,1.03,-0.18),
                    'Djorg 2'  :(2.76357,-2.50848,6.3,0.33,1.05,-0.65),
                    'Terzan 9' :(3.60314,-1.98878,7.1,0.03,0.78,-1.05),
                    'Terzan 10':(4.42075,-1.86289,5.8,0.90,1.55,-1.00)}
    
    glob_cluster_df = pd.DataFrame(glob_clusters,index=['l','b','d','r_c','r_d','[Fe/H]']).T
    glob_cluster_df.to_csv('glob_cluster.csv')

    EB_data = pd.read_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/ecl_dat.csv', index_col='ID')
    BEAM_extintion = pd.read_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/all_variables_extintion_.csv', index_col='ID')
    
    # match with globular clusters
    def sep_2d(l_cat, b_cat, l_targ, b_targ):
        sep = np.sqrt((l_targ-l_cat)**2 + (b_targ-b_cat)**2) 
        return sep
    
    def match(rfactor=3):
        match_ids = pd.DataFrame()
        for cluster in glob_clusters.keys():
            l_cat  = glob_clusters[cluster][0]
            b_cat  = glob_clusters[cluster][1]
            r_h = glob_clusters[cluster][4]*u.arcmin
            if r_h < 0:
                r_h = glob_clusters[cluster][3]*u.arcmin
            id_list = {}
            for star in rrl_ids:
                l_targ = rrl_dat.loc[star,'gal_l']
                b_targ = rrl_dat.loc[star,'gal_b']
                dist2d = sep_2d(l_cat, b_cat, l_targ, b_targ)*u.deg
                if dist2d < r_h.to('deg') * rfactor:
                    id_list[star]= [dist2d.value / r_h.to('deg').value, 'RRL']

            for star in EB_data.index:
                l_targ = EB_data.loc[star,'gal_l']
                b_targ = EB_data.loc[star,'gal_b']
                dist2d = sep_2d(l_cat, b_cat, l_targ, b_targ)*u.deg
                if dist2d < r_h.to('deg') * rfactor:
                    id_list[star]= [dist2d.value / r_h.to('deg').value, 'EB']

            for n,id in enumerate(list(id_list.keys())):
                match_ids.loc[id,['cluster','sep_factor','n','type']] = [cluster,round(id_list[id][0],1),f'{int(n+1)}',id_list[id][1]]
        return match_ids

    # make a table for rrl inside globular cluster:
    # flag 0 : known RRL
    # flag 1 : New RRL
    matches = match()
    for _ in matches.index:
        if matches.loc[_,'type'] == 'RRL':
            matches.loc[_,'OID'] = rrl_names[_]
            matches.loc[_,'OGLE_ID'] = rrl_dat.loc[_,'OGLE_ID']
            matches.loc[_,['d','d_err','[Fe/H]','[Fe/H]_err','E(J-Ks)','E(J-KS)_err','E(H-Ks)','E(H-KS)_err','ExtintionFlag']] = distances.loc[_,['distance','distanceSigma','[Fe/H]', '[Fe/H]_err','E(J-Ks)','E(J-KS)_err','E(H-Ks)','E(H-KS)_err','ExtintionFlag']].values
            matches.loc[_,['RA','DEC','gal_l','gal_b','period','amplitude','mag_Ks','er_Ks','mag_J','er_J','mag_H','er_H']] = rrl_dat.loc[_,['RA','DEC','gal_l', 'gal_b','period', 'amplitude','lc_mean','lc_mean_err','mag_J','er_J','mag_H','er_H']].values
            matches.loc[_,['J-Ks']] = rrl_dat.loc[_,'mag_J'] - rrl_dat.loc[_,'mag_Ks']
            matches.loc[_,['H-Ks']] = rrl_dat.loc[_,'mag_H'] - rrl_dat.loc[_,'mag_Ks']

        if matches.loc[_,'type'] == 'EB':
            matches.loc[_,['OID','OGLE_ID','RA','DEC','gal_l','gal_b','mag_Ks','er_Ks']] = EB_data.loc[_,['OID','OGLE_ID','RA','DEC','gal_l','gal_b','lc_mean','lc_mean_err']].values
            matches.loc[_,['E(J-Ks)','E(J-KS)_err']] = BEAM_extintion.loc[_,['E_JK','SigmaE_JK']].values
            matches.loc[_,['period']] = d.all_dat.loc[_,'period']*2
            matches.loc[_,['mag_J','er_J','mag_H','er_H']] = d.all_dat.loc[_,['mag_J','er_J','mag_H','er_H']]
            matches.loc[_,['J-Ks']] = d.all_dat.loc[_,'mag_J'] - d.all_dat.loc[_,'mag_Ks']
            matches.loc[_,['H-Ks']] = d.all_dat.loc[_,'mag_H'] - d.all_dat.loc[_,'mag_Ks']
            matches.loc[_,'ExtintionFlag'] = 3
    matches.to_csv('rrl_match_glob_clusters.csv',sep='\t')

                       
    cluster_distances = {}  # distance, sigma_stat, sigma_syst
    # Culster distances based on RRL distances:
    for cluster in ['NGC 6544', 'Djorg 2', 'Terzan 9', 'Terzan 10']:
        ids_cluster = matches.index[matches.cluster == cluster]
        msk = matches.loc[ids_cluster,'type'] == 'RRL'
        if cluster == 'NGC 6544':
            ids_rrl = [_ for _ in ids_cluster[msk] if _ != ids_cluster[msk][3]]
        elif cluster == 'Terzan 10':
            ids_rrl = [_ for _ in ids_cluster[msk] if _ != 'b308_3_z_14_k_270.76699_-26.09268']
        else:
            ids_rrl = [_ for _ in ids_cluster[msk]]
        if cluster == 'Terzan 9':
            cluster_distances[cluster] = (  matches.loc[ids_rrl,'d'].mean(), 
                                        matches.loc[ids_rrl,'d'].std(), 
                                        np.sqrt(np.power(matches.loc[ids_rrl,'d_err'],2).sum())/(len(ids_rrl)),
                                        matches.loc[ids_rrl,'period'].mean())
        else:
            cluster_distances[cluster] = (  matches.loc[ids_rrl,'d'].mean(), 
                                        matches.loc[ids_rrl,'d'].std(), 
                                        np.sqrt(np.power(matches.loc[ids_rrl,'d_err'],2).sum())/(len(ids_rrl)-1),
                                        matches.loc[ids_rrl,'period'].mean())


    # Globular Cluster CMD

    
    #                       ax,    ,xlim,   
    plot_par = {'NGC 6544' :([0,0],[0.40,1.8]),
                'Djorg 2'  :([0,1],[0.25,1.4]),
                'Terzan 9' :([1,0],[0.37,1.9]),
                'Terzan 10':([1,1],[0.25,2.0])}

    extintion = distances['E(J-Ks)'].mean()
    reddening = extintion * 0.689
    xlabel='(J-Ks)'
    ckey = 'mag_J'
    arrow_xpos = 0.4

    EB_data = pd.read_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/ecl_dat.csv', index_col='ID')


    font_size = 11
    plt.rcParams.update({'font.size': font_size})
    fig, axes = plt.subplots(2, 2, figsize=(7,7))
    fig.subplots_adjust(wspace=0,hspace=0)

    for cluster in [_ for _ in list(glob_clusters.keys()) if _ != 'NGC 6553']:
        r_h = glob_clusters[cluster][4]*u.arcmin.to('deg')
        if r_h < 0:
            r_h = glob_clusters[cluster][3]*u.arcmin.to('deg')
        rfactor = 3

        tiles = []
        for _ in matches[matches.cluster == cluster].index:
            if _[:4] not in tiles:
                tiles.append(_[:4])

        for tile in tiles:
            tileData = []
            chips = [_[:-3] for _ in os.listdir(f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/psf_ts/{tile}/chips/') if _.endswith('.ts')]
            for chip in chips:
                chipData = pd.read_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/psf_ts/{tile}/chips/{chip}.ts',index_col='ID')
                tileData.append(chipData)
    
            tileData = pd.concat(tileData)
            tileData = tileData.drop_duplicates()
            color = tileData.mag_J - tileData.mag_Ks
            icrs = SkyCoord(ra=tileData.RA, dec=tileData.DEC,unit=(u.deg, u.deg))
            gal  = icrs.galactic

            l_cat  = glob_clusters[cluster][0]
            b_cat  = glob_clusters[cluster][1]
            l_targ = gal.l.deg
            b_targ = gal.b.deg

            dist2d = sep_2d(l_cat, b_cat, l_targ, b_targ)
            cluster_ids = tileData.index[dist2d < rfactor*r_h]
            
            mag_cols = [_ for _ in tileData.columns if _[:3] == 'MAG']
            err_cols = [_ for _ in tileData.columns if _[:3] == 'ERR']
            
            Ks_mag = tileData.loc[cluster_ids,mag_cols].mean(axis=1)
            Ks_err = np.sqrt((tileData.loc[cluster_ids,err_cols]**2).sum(axis=1))/((~tileData.loc[cluster_ids,err_cols].isna()).sum(axis=1) -1 )
            
            # mag and color inside cluster 3xr_h
            c_color  = color.loc[cluster_ids]
            mask   = ~c_color.isnull()
            c_color  = c_color[mask]
            c_Ks_mag = Ks_mag[mask]
            c_Ks_err = Ks_err[mask]

            # mag and color from 2d matched RRL
            c_rrl_ids = [_ for _ in matches.index if _[:4] == tile and matches.loc[_,"cluster"] == cluster and matches.loc[_,"type"] == 'RRL']
            c_rrl_mag = rrl_dat.loc[c_rrl_ids,'lc_mean']
            c_rrl_color = rrl_dat.loc[c_rrl_ids,'mag_J'] - rrl_dat.loc[c_rrl_ids,'mag_Ks']

            # mag and color from 2d matched EBs
            EB_ids = [_ for _ in matches.index if _[:4] == tile and matches.loc[_,"cluster"] == cluster and matches.loc[_,"type"] == 'EB']
            EB_mag = EB_data.loc[EB_ids,'lc_mean']
            EB_color = EB_data.loc[EB_ids,'mag_J'] - EB_data.loc[EB_ids,'mag_Ks']

            ax1 = plot_par[cluster][0][0]
            ax2 = plot_par[cluster][0][1]
            #fig, axes = plt.subplots(1, 1, figsize=(7,7))
            axes[ax1,ax2].scatter(  c_color,
                                    c_Ks_mag,
                                    marker='.',
                                    c='dimgray',
                                    s=10,
                                    alpha=.1,)
            axes[ax1,ax2].scatter(  c_rrl_color,
                                    c_rrl_mag,
                                    marker='^',
                                    c='red',
                                    label='RRL',
                                    s=30)
            for _ in c_rrl_ids:
                axes[ax1,ax2].text( s=f'{matches.n[_]}',
                                    x=c_rrl_color.loc[_],
                                    y=c_rrl_mag.loc[_] - 0.6,
                                    c='red',
                                    ha='center',
                                    va='top',
                                    weight="bold")
            
            axes[ax1,ax2].scatter(  EB_color,
                                    EB_mag,
                                    marker='s',
                                    c='blue',
                                    label='BE',
                                    s=30)
            for _ in EB_ids:
                axes[ax1,ax2].text( s=f'{matches.n[_]}',
                                    x=EB_color.loc[_],
                                    y=EB_mag.loc[_] + 0.6,
                                    c='blue',
                                    ha='center',
                                    va='bottom',
                                    weight="bold")
            if ax1==0 and ax2==1:
                axes[ax1,ax2].legend()
            
            axes[ax1,ax2].set_xlim(0.25,1.99)
            axes[ax1,ax2].set_ylim(11.1,17.9)
            axes[ax1,ax2].set_xlabel(r'$\mathrm{%s}$'%xlabel)
            axes[ax1,ax2].set_ylabel(r'$\mathrm{K_s\ [mag]}$')
            axes[ax1,ax2].invert_yaxis()
            # reddening vector
            axes[ax1,ax2].annotate("", xy=(arrow_xpos+extintion, 11.4+reddening),
                                     xytext=(arrow_xpos, 11.4),
                                     arrowprops=dict(arrowstyle="->", color='r'))
            axes[ax1,ax2].text(0.5,0.04, f'{tile} | {cluster}',c='k',ha='center',transform=axes[ax1,ax2].transAxes)
    for ax in fig.get_axes():
        ax.label_outer()
    plt.savefig(f'CMD_GC_{ckey}.png',dpi=200,bbox_inches='tight',pad_inches=0.05)
    plt.show()
       








    # b309 Red Clump Dust Lane
    plt.rcParams.update({'font.size': 13})

    tiles = sorted(os.listdir(f'{path}/data/psf_ts/'))
    all_dat = pd.read_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/all_variables_match_vsx_ogle_gaia.csv',index_col='ID')
    all_var_extintion = pd.read_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/all_variables_extintion.csv',index_col='ID')
    AKs = 0.689 * all_var_extintion.E_JK #http://mill.astro.puc.cl/BEAM/coffinfo.php

    tileData = []
    tile = 'b309'
    chips = [_[:-3] for _ in os.listdir(f'{path}/data/psf_ts/{tile}/chips/') if _.endswith('.ts')]
    for chip in chips:
        chipData = pd.read_csv(f'{path}/data/psf_ts/{tile}/chips/{chip}.ts',index_col='ID')
        tileData.append(chipData[['RA','DEC','mag_J','mag_Ks']])
    
    tileData = pd.concat(tileData)
    tileData = tileData.drop_duplicates()
    icrs = SkyCoord(ra=tileData.RA, dec=tileData.DEC,unit=(u.deg, u.deg))
    gal  = icrs.galactic
    l = gal.l.deg
    b = gal.b.deg
    tileData.loc[tileData.index, 'gal_l'] = l
    tileData.loc[tileData.index, 'gal_b'] = b

    color = tileData.mag_J - tileData.mag_Ks
    msk   = ~color.isnull()
    mag   = tileData.mag_Ks
    mag   = mag[msk]
    color = color[msk]


    #RC selection
    rc_msk1 = ((color > 0.8) & (color < 1.35) & (mag < 14.5))
    rc_msk2 = ((color > 1.35) & (mag < 14.5))


    bins=(400,300) 
    cmap = copy.copy(mpl.cm.get_cmap("jet"))# plt.cm.jet
    cmap.set_bad('w', 1.)
    cmap_multicolor = copy.copy(mpl.cm.get_cmap("jet")) # plt.cm.jet
    cmap_multicolor.set_bad('w', 1.)
    N, xedges, yedges = np.histogram2d(color,mag,bins=bins)
    fig, axes = plt.subplots(2,3, figsize=(9,6), gridspec_kw={'width_ratios': [10,10,2],'hspace':0.2,'wspace':0.1})    
    img = axes[0,0].imshow(np.log10(N.T),
                        origin='lower',
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                        aspect='auto',
                        interpolation='nearest',
                        cmap=cmap)
    axes[0,0].invert_yaxis()
    axes[0,0].set_xlabel(r'$\mathrm{(J-K_s)}$')
    axes[0,0].set_ylabel(r'$\mathrm{K_s \ [mag]}$')
    axes[0,0].xaxis.set_label_position('top') 
    

    left, bottom, width, height = (0.8, mag.min(), color.max()-0.8, 14.5-mag.min())
    rect = plt.Rectangle((left, bottom), width, height,
                     facecolor="black", alpha=0.3)
    axes[0,0].add_patch(rect)
    axes[0,0].vlines(x=1.35,ymin=mag.min(), ymax=14.5, color='k',linestyles='dashed',lw=1)
    axes[0,0].set_xlim(0.01,2.8)
    axes[0,0].tick_params(top=True, bottom=False, left=True, right=False, labelleft=True, labelright=False, labelbottom=False, labeltop=True)

    
    
    indexes = [_ for _ in all_dat.index if _ in AKs.index and _[:4] == 'b309']
    color_morm = AKs.loc[indexes]/AKs.loc[indexes].max()
    axes[0,1].scatter(all_dat.loc[indexes].gal_l, all_dat.loc[indexes].gal_b, 
                c=1-color_morm, marker='s', s=50, lw = 0, 
                cmap='inferno', alpha=.75)
    axes[0,1].invert_xaxis()
    axes[0,1].set_xlim(tileData['gal_l'].max(),tileData['gal_l'].min())
    axes[0,1].set_ylim(tileData['gal_b'].min(),tileData['gal_b'].max())
    axes[0,1].tick_params(top=True, bottom=False, left=False, right=True, labelleft=False, labelright=True, labelbottom=False, labeltop=True)
    axes[0,1].set_xlabel(r'$l\ \mathrm{[graus]}$')
    axes[0,1].set_ylabel(r'$b\ \mathrm{[graus]}$')
    axes[0,1].xaxis.set_label_position('top')
    axes[0,1].yaxis.set_label_position('right') 

    #colorbar
    a = np.array([[AKs.loc[indexes].max(),AKs.loc[indexes].min()]])
    cmap = plt.get_cmap('inferno').reversed()                                              
    cax = plt.axes([0.91, 0.56, 0.01, 0.3])
    img = axes[0,2].imshow(a, cmap=cmap)
    axes[0,2].set_visible(False)
    cbar = plt.colorbar(img, orientation="vertical", cax=cax)
    cbar.ax.set_ylabel('$A_{Ks}$', rotation=90)


    bins=(150,150)
    cmap = copy.copy(mpl.cm.get_cmap('inferno'))# plt.cm.jet
    cmap.set_bad('w', 1.)
    cmap_multicolor = copy.copy(mpl.cm.get_cmap('inferno')) # plt.cm.jet
    cmap_multicolor.set_bad('w', 1.)
    N, xedges, yedges = np.histogram2d(tileData['gal_l'].loc[color[rc_msk1].index],tileData['gal_b'].loc[color[rc_msk1].index],bins=bins)
    axes[1,0].imshow(np.log10(N.T),
                        origin='lower',
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                        aspect='auto',
                        interpolation='nearest',
                        cmap=cmap)
    axes[1,0].invert_xaxis()
    axes[1,0].set_xlim(tileData['gal_l'].max(),tileData['gal_l'].min())
    axes[1,0].set_ylim(tileData['gal_b'].min(),tileData['gal_b'].max())
    axes[1,0].text(0.5,1.05, r'$\mathrm{0,8 < (J-K_s) < 1,35}$',c='k',ha='center',transform=axes[1,0].transAxes)
    axes[1,0].tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelright=False, labelbottom=True, labeltop=False)
    axes[1,0].set_xlabel(r'$l\ \mathrm{[graus]}$')
    axes[1,0].set_ylabel(r'$b\ \mathrm{[graus]}$')

    N, xedges, yedges = np.histogram2d(tileData['gal_l'].loc[color[rc_msk2].index],tileData['gal_b'].loc[color[rc_msk2].index],bins=bins)
    img = axes[1,1].imshow(np.log10(N.T),
                        origin='lower',
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                        aspect='auto',
                        interpolation='nearest',
                        cmap=cmap)
    axes[1,1].invert_xaxis()
    axes[1,1].set_xlim(tileData['gal_l'].max(),tileData['gal_l'].min())
    axes[1,1].set_ylim(tileData['gal_b'].min(),tileData['gal_b'].max())
    axes[1,1].text(0.5,1.05, r'$\mathrm{(J-K_s) > 1,35}$',c='k',ha='center',transform=axes[1,1].transAxes)
    axes[1,1].tick_params(top=False, bottom=True, left=False, right=True, labelleft=False, labelright=True, labelbottom=True, labeltop=False)
    axes[1,1].set_xlabel(r'$l\ \mathrm{[graus]}$')
    axes[1,1].set_ylabel(r'$b\ \mathrm{[graus]}$')
    axes[1,1].yaxis.set_label_position('right') 


    cbar_ax = plt.axes([0.91, 0.14, 0.01, 0.3])
    cb = fig.colorbar(img, 
                    ticks=[0, 1, 2, 3],
                    format=r'$10^{%i}$',
                    shrink=0.6 ,
                    cax=cbar_ax)
    cb.set_label(r'n\'{u}mero por pixel',rotation=90)
    axes[0,2].set_visible(False)
    axes[1,2].set_visible(False)
    plt.savefig('b309_RC_dust_lane.png',dpi=300,bbox_inches = 'tight',pad_inches=0.05)
    plt.show()










    # RRL Bailey diagram and histogram of periods
    
    # Oosterhoff I (Navarrete 2015, ZOROTOVIC, 2010)
    def OoI_curve(period):
        A_V = -2.627 - 22.046*np.log10(period) - 30.876*(np.log10(period))**2 # ZOROTOVIC, 2010
        A_OoI =  0.32*np.power(A_V,2./3) # conversion to Ks mag Navarrete 2015
        return A_OoI

    def OoII_curve(period):
        A_J = 0.064 - 2.481*np.log10(period) +10.345*(np.log10(period))**3 #Navarrete 2015
        A_OoII = np.power(A_J/2.6,2./3)  # conversion to Ks mag Navarrete 2015
        return A_OoII

    def OoI_ogle(period):
        #A_V = -2.627 - 22.046*np.log10(period) - 30.876*(np.log10(period))**2 # ZOROTOVIC, 2010
        #A_I = A_V/1.6
        A_I = -1.64 - 13.78*np.log10(period) - 19.30*(np.log10(period))**2 # Kunder 2013 ; https://iopscience.iop.org/article/10.1088/0004-6256/145/2/33
        return A_I
    
    def OoII_ogle(period):
        #_V = -2.627 - 22.046*(np.log10(period)-0.03) - 30.876*(np.log10(period)-0.03)**2 # ZOROTOVIC, 2010
        #_I = A_V/1.6
        A_I = -0.89 - 11.46*(np.log10(period)) - 19.30*(np.log10(period))**2 # Kunder 2013 ; https://iopscience.iop.org/article/10.1088/0004-6256/145/2/33
        return A_I
    


    import matplotlib.gridspec as gridspec
    from matplotlib import cm
    from matplotlib.colors import Normalize 
    from scipy.interpolate import interpn

    p1 = np.linspace(0.4,0.7,100)
    p2 = np.linspace(0.55,0.9,100)
    periods = 1./d.rrl_fitparams.Freq
    amplitudes = d.rrl_fitparams.Amplitude
    fig = plt.figure(figsize=[7,7],tight_layout=True)
    gs = gridspec.GridSpec(2, 2, height_ratios=[2,1.5])

    ax1 = fig.add_subplot(gs[0,0])

   
    cmap = copy.copy(mpl.cm.get_cmap("viridis"))# plt.cm.jet
    cmap.set_bad('w', 1.)
    cmap_multicolor = copy.copy(mpl.cm.get_cmap("viridis")) # plt.cm.jet
    cmap_multicolor.set_bad('w', 1.)
    
    bins=(30,30)
    N, xedges, yedges = np.histogram2d(periods,amplitudes,bins=bins)
    Z = interpn( ( 0.5*(xedges[1:] + xedges[:-1]) , 0.5*(yedges[1:]+yedges[:-1]) ) , N , np.vstack([periods,amplitudes]).T , method = "splinef2d", bounds_error = False)
    #To be sure to plot all data
    Z[np.where(np.isnan(Z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    idx = Z.argsort()
    x, y, z = periods[idx], amplitudes[idx], Z[idx]

    ax1.scatter(x,
                y,
                marker='.',
                s=20,
                c=z,
                cmap=cmap,
                ec='none',
                alpha=1)

    ax1.plot(p1,OoI_curve(p1),'r-')
    ax1.plot(p2,OoII_curve(p2),'r--')
    ax1.text(s=r'$\mathrm{Nossas\ RRL}$',  x=0.5, y=1.02,transform=ax1.transAxes,ha='center')

    ax1.set_ylabel('$\mathrm{Amplitude\ K_s\ [mag]}$')
    ax1.set_xlabel('$\mathrm{P\ [dias]}$')
    ax1.set_xlim(0.21,0.99)
    ax1.set_ylim(0.0,1)

    bins=(30,30)
    N, xedges, yedges = np.histogram2d(ogle_bulge_rrl.P,ogle_bulge_rrl.A_I,bins=bins)
    Z = interpn( ( 0.5*(xedges[1:] + xedges[:-1]) , 0.5*(yedges[1:]+yedges[:-1]) ) , N , np.vstack([ogle_bulge_rrl.P,ogle_bulge_rrl.A_I]).T , method = "splinef2d", bounds_error = False)
    #To be sure to plot all data
    Z[np.where(np.isnan(Z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    idx = Z.argsort()
    x, y, z = ogle_bulge_rrl.P[idx], ogle_bulge_rrl.A_I[idx], Z[idx]

    ax2 = fig.add_subplot(gs[0,1])
    ax2.scatter(x,
                y,
                marker='.',
                s=10,
                c=z,
                cmap=cmap,
                ec='none',
                alpha=1)
    ax2.plot(p1,OoI_ogle(p1),'r-')
    ax2.plot(p2,OoII_ogle(p2),'r--')

    ax2.text(s=r'$\mathrm{RRab\ do\ OGLE}$',  x=0.5, y=1.02,transform=ax2.transAxes,ha='center')
    ax2.set_ylabel('$\mathrm{Amplitude\ I\ [mag]}$')
    ax2.set_xlabel('$\mathrm{P\ [dias]}$')
    ax2.set_xlim(0.21,0.99)
    ax2.set_ylim(0.0,1)


    ax3 = fig.add_subplot(gs[1,:])
    weights = np.ones_like(periods)/float(len(periods))
    ax3.hist(periods,
                bins=30,
                #density=True,
                weights=weights,
                histtype='barstacked',
                lw=.5,
                color='k',
                edgecolor='w',
                alpha=0.6,
                label='Nossas RRL')
    weights = np.ones_like(ogle_bulge_rrl.P)/float(len(ogle_bulge_rrl.P))
    ax3.hist(ogle_bulge_rrl.P,
                bins=30,
                weights=weights,
                #density=True,
                lw=1.5,
                histtype='step',
                color='r',
                edgecolor='r',
                alpha=1,
                label='RRab do OGLE')
    ax3.legend()
    ax3.set_xlabel('$\mathrm{P\ [dias]}$')
    ax3.set_ylabel('$\#\ \mathrm{normalizado}$')
    ax3.set_xlim(0.21,.99)
    plt.savefig('bailey_diagram.png',dpi=300,pad_inches=0.05)
    plt.show()


    # prints
    print('Percentual de RRL com AKs < 0,15:', (len(amplitudes[amplitudes < .15])/len(amplitudes) )*100)
    print('Percentual de RRL do OGLE com AI < 0,3:', (len(ogle_bulge_rrl.A_I[ogle_bulge_rrl.A_I < .3])/len(ogle_bulge_rrl.A_I) )*100)






    ''' old version '''
    fig,ax = plt.subplots(2,1,figsize=[7,7],gridspec_kw={'height_ratios': [2,1.5]})
    fig.subplots_adjust(wspace=0,hspace=0)
    ax[0].scatter(periods,amplitudes,
                marker='.',
                color='k',
                ec='none',
                alpha=0.5,)

    ax[0].plot(p,OoI_curve(p),'r-')
    ax[0].plot(p,OoII_curve(p),'r--')

    ax[0].set_ylabel('$\mathrm{Amplitude\ [mag]}$')
    ax[0].set_xlim(0.21,.99)
    ax[0].set_ylim(0.07,0.64)
    ax[1].hist(periods,
                bins=30,
                histtype='barstacked',
                lw=.5,
                color='k',
                edgecolor='w',
                alpha=0.6)
    ax[1].set_xlabel('$\mathrm{P\ [dias]}$')
    ax[1].set_ylabel('$\#\ \mathrm{estrelas}$')
    ax[1].set_xlim(0.21,.99)
    ax[1].set_ylim(0,220)
    ax_t = ax[1].secondary_xaxis('top')
    ax_t.tick_params(axis='x', direction='inout',length=6)
    ax_t.set_xticklabels([])
    #plt.tight_layout()
    plt.savefig('bailey_diagram.png',dpi=300,bbox_inches = 'tight',pad_inches=0.05)
    plt.show()



    #period vs metalicidade
    FeH = distances['[Fe/H]']
    plt.plot(FeH,periods,'.')
    plt.show()






    # Period vs Period
    our_periods = 1./d.rrl_fitparams.Freq
    ogle_periods = d.all_dat.loc[our_periods.index,'OGLE_P']
    vsx_periods  = d.all_dat.loc[our_periods.index,'VSX_P']
    viva_periods = d.all_dat.loc[our_periods.index,'vivaP']
   
    fig,ax = plt.subplots(1,1,figsize=[7,7])
    ms = 30
    ax.scatter(our_periods,vsx_periods,
                marker='s',
                color='r',
                s=ms*5,
                ec='r',
                fc='none',
                label='VSX',
                alpha=1)

    ax.scatter(our_periods,ogle_periods,
                marker='o',
                color='dodgerblue',
                s=ms*3,
                ec='k',
                fc='none',
                label='OGLE',
                alpha=1)
    ax.scatter(our_periods,viva_periods,
                marker='.',
                color='dodgerblue',
                s=ms*1,
                ec='none',
                fc='dodgerblue',
                label='VIVA',
                alpha=1)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0.1,2)
    ax.set_ylim(0.1,2)
    ax.set_xlabel(r'$\textrm{Nosso Período}$')
    ax.set_ylabel(r'$\textrm{Período do OGLE/ VSX/ VIVA}$')
    ax.legend()
    plt.tight_layout()
    #plt.savefig('period_vs_period_rrl.png',dpi=300,bbox_inches = 'tight',pad_inches=0.05)
    plt.show()





    # RRL Distance Histogram
    plt.rcParams.update({'font.size': 13})
    histrange = [2,15]
    histbin = 40
    rx = (distances.x + GC_dist)/1000
    rx_BEAM = (distances_BEAM.x + GC_dist)/1000

    msk1 = distances.ExtintionFlag == 1
    msk2 = distances.ExtintionFlag == 2   
    msk3 = distances.ExtintionFlag == 3
    fig,ax = plt.subplots(2,1,figsize=[7,7],sharex=True,gridspec_kw={'hspace':0})
    ax[0].hist(rx,
            bins=histbin,
            range=histrange,
            histtype='barstacked',
            lw=1.1,
            edgecolor='w',
            label='Soma',
            color='k',
            alpha=0.2)
    ax[0].hist(rx[msk1],
            bins=histbin,
            range=histrange,
            histtype='step',
            label='E(J-Ks)',
            color='orangered')
    ax[0].hist(rx[msk2],
            bins=histbin,
            range=histrange,
            histtype='step',
            label='E(H-Ks)',
            color='forestgreen')  
    ax[0].hist(rx[msk3],
            bins=histbin,
            range=histrange,
            histtype='step',
            label='BEAM',
            color='firebrick')
    ax[0].axvline(x=GC_dist/1000,ls='--',lw=1,c='gray')
    ax[0].legend()#prop={'size': 10})
    
    ax[1].hist(rx,
            bins=histbin,
            range=histrange,
            histtype='step',
            label='Soma',
            color='dodgerblue')
    ax[1].hist(rx_BEAM,
            bins=histbin,
            range=histrange,
            histtype='step',
            label='Somente BEAM',
            color='firebrick')
    ax[1].axvline(x=GC_dist/1000,ls='--',lw=1,c='gray')

    ax[0].set_ylabel('$\mathrm{\#\ estrelas}$')
    ax[1].set_ylabel('$\mathrm{\#\ estrelas}$')
    ax[1].set_xlabel('$d\ \mathrm{[kpc]}$')
    ax[1].legend()#prop={'size': 10})
    plt.tight_layout()
    plt.savefig('rrl_dist_hist.png',dpi=300, bbox_inches = 'tight',pad_inches=0.05)
    plt.show()






    # RRL Distance Histogram - bar oriented
    plt.rcParams.update({'font.size': 13})
    histrange = [-6.178,6.822]
    histbin = 40
    bar_angle = 20*np.pi/180

    rx = distances.x
    ry = distances.y
    alpha = np.arctan(ry/rx) - bar_angle
    R = (rx**2 + ry**2)**0.5
    rx2 = R*np.cos(alpha)
    d_x = rx 
    rx_theta = rx + np.tan(bar_angle) * ry 
    rx_theta2 = GC_dist*np.cos(bar_angle) + rx2

    fig,ax = plt.subplots(1,1,figsize=[7,4],sharex=True,gridspec_kw={'hspace':0})
    ax.hist(rx_theta/1000,
            bins=histbin,
            range=histrange,
            histtype='step',
            label=r'$\mathrm{x + y. tan \phi}$',
            lw=1.2,
            color='k')

    #ax.axvline(x=GC_dist/1000,ls='--',lw=1,c='gray')
    ax.axvline(x=0,ls='--',lw=1,c='gray')
    ax.hist(rx/1000,
            bins=histbin,
            range=histrange,
            histtype='barstacked',
            label=r'$\mathrm{x}$',
            color='k',
            lw=1.1,
            edgecolor='w',
            alpha=0.2)

    ax.set_ylabel('$\mathrm{\#\ estrelas}$')
    ax.set_xlabel('$\mathrm{x\ [kpc]}$')
    ax.legend()#prop={'size': 10})
    plt.tight_layout()
    plt.savefig('rrl_dist_hist2.png',dpi=300, bbox_inches = 'tight',pad_inches=0.05)
    plt.show()




    
    # [Fe/H] distribution
    import math_functions
    mu1 = -1.5
    mu2 = -1.0
    A1  = 200
    A2  = 100
    sigma1 = 0.5
    sigma2 = 0.5

    hist, bin_edges = np.histogram(distances['[Fe/H]'], bins=30)
    bin_centers = []
    n=0
    while n < len(bin_edges)-1:
        bin_centers.append((bin_edges[n]+bin_edges[n+1])/2)
        n+=1


    guess = [A1,mu1,sigma1,A2,mu2,sigma2]
    gauss_fit = leastsq(func=math_functions.double_gaussian_residuals,
                        x0=guess,
                        args=(bin_centers,hist))

    x = np.linspace(-3.5,1,200)
    y = math_functions.double_gaussian(x,gauss_fit[0])
    y1 = math_functions.single_gaussian(x,gauss_fit[0][:3])
    y2 = math_functions.single_gaussian(x,gauss_fit[0][3:])
    
    plt.rcParams.update({'font.size': 13})
    fig,ax=plt.subplots(figsize=(7,3.5))
    ax.hist(distances['[Fe/H]'],
             bins=30,
             histtype='barstacked',
             lw=.5,
             color='k',
             edgecolor='w',
             alpha=0.6)
    ax.plot(x,y,'r-')        
    ax.plot(x,y1,'k--')
    ax.plot(x,y2,'k--')
    ax.set_xlabel('$\mathrm{[Fe/H]}$')
    ax.set_ylabel('$\mathrm{\#\ estrelas}$')
    plt.tight_layout()
    plt.savefig('rrl_metalicity.png',dpi=300, bbox_inches = 'tight',pad_inches=0.05)
    plt.show()

    # Z distribution
    plt.hist((distances['Z']),bins=300,histtype='step')
    plt.xlabel('log(Z)')
    plt.ylabel('\# stars')
    plt.show()

    # M_Ks distributuion
    plt.hist(distances['M_Ks'],bins=30,histtype='step')
    plt.xlabel('Absolute Ks mag')
    plt.ylabel('\# stars')
    plt.show()

    # E[J-Ks] distribution
    plt.hist(distances['E(J-Ks)'],bins=40,histtype='step')
    plt.xlabel('E(J-Ks)')
    plt.ylabel('\# stars')
    plt.show()






    # cone view - polar:
    plt.rcParams.update({'font.size': 14})

    distances = d.get_distance(magoffset=0,campain='variability',method='color')
    rx = distances.x + GC_dist
    r = rx * np.cos(distances.gal_l.values*np.pi/180)
    l = distances.gal_l*np.pi/180

    from matplotlib.ticker import LogFormatter
    import matplotlib
    fig = plt.figure(figsize=[14,14])
    #plot RRL
    ax = fig.add_subplot(111, polar=True)

    bins=(int(15*0.6),int(120*0.6) )
    cmap = copy.copy(mpl.cm.get_cmap("jet"))# plt.cm.jet
    cmap.set_bad('w', 1.)
    cmap_multicolor = copy.copy(mpl.cm.get_cmap("jet")) # plt.cm.jet
    cmap_multicolor.set_bad('w', 1.)
    N, xedges, yedges = np.histogram2d(l,r/1000,bins=bins)
    img = ax.imshow(N.T,
                    norm=matplotlib.colors.LogNorm(),
                    origin='lower',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    aspect='auto',
                    interpolation='none',
                    cmap=cmap,
                    alpha=.6)
    

    #ax.scatter(l, r/1000,c='k', alpha=1, marker='.',s=2)
    ax.scatter(0,GC_dist/1000,marker='x',c='k',label="GC")
    #plot RC
    i = 0
    RC_dist1_mean = []
    RC_dist2_mean = []
    while i < 4:
        tile1 = Rc_peaks.index[i]
        tile2 = Rc_peaks.index[i+4]
        print(tile1, tile2)
        gal_l = Rc_peaks.loc[tile1,'tile_central_l']
        gal_b = Rc_peaks.loc[tile1,'tile_central_b']

        err_peak1  = Rc_peaks['RC_peak1_dist_sigma']
        mask1      = err_peak1 > 2000
        dist_peak1 = Rc_peaks['RC_peak1_dist']
        dist_peak1.loc[dist_peak1.index[mask1]] = np.nan
        err_peak1.loc[dist_peak1.index[mask1]] = np.nan

        err_peak2  = Rc_peaks['RC_peak2_dist_sigma']
        mask2      = err_peak2 > 2000
        dist_peak2 = Rc_peaks['RC_peak2_dist']
        dist_peak2.loc[dist_peak2.index[mask2]] = np.nan
        err_peak2.loc[dist_peak2.index[mask2]] = np.nan
        
        RC_dist1 = np.nanmean([dist_peak1.loc[tile1], dist_peak1.loc[tile2]])
        RC_dist1_err = np.sqrt(np.nansum([err_peak1.loc[tile1]**2, err_peak1.loc[tile2]**2]))/2
        RC_dist1_mean.append((gal_l*np.pi/180, RC_dist1/1000))

        RC_dist2 = np.nanmean([dist_peak2.loc[tile1], dist_peak2.loc[tile2]])
        RC_dist2_err = np.sqrt(np.nansum([err_peak2.loc[tile1]**2, err_peak2.loc[tile2]**2]))/2
        RC_dist2_mean.append((gal_l*np.pi/180, RC_dist2/1000))
        color = 'maroon'#'r'#plt.cm.Set1(0)#'m'#plt.cm.Accent(5)
        ax.errorbar(gal_l*np.pi/180,
                        RC_dist1/1000,
                        yerr=RC_dist1_err/1000,
                        marker="o",
                        mfc=color,
                        mec=color,
                        ecolor=color,
                        ms=5,
                        lw=1)

        ax.errorbar(gal_l*np.pi/180,
                        RC_dist2/1000,
                        yerr=RC_dist2_err/1000,
                        marker="o",
                        mfc=color,
                        mec=color,
                        ecolor=color,
                        ms=5,
                        lw=1)
        i+=1
    RC_dist1_mean = np.array(RC_dist1_mean).T
    RC_dist2_mean = np.array(RC_dist2_mean).T
    ax.plot(RC_dist1_mean[0],RC_dist1_mean[1],'-',c=color)
    ax.plot(RC_dist2_mean[0],RC_dist2_mean[1],'-',c=color)
    ax.set_thetamin(-1)
    ax.set_thetamax(8)
    ax.set_rlim((0, 12))
    
    
    formatter = LogFormatter(10, labelOnlyBase=False) 

    cbar_ax = plt.axes([0.4, 0.35, 0.2, 0.01])
    cb = fig.colorbar(img,
                    ticks=[1,2,10,15],
                    orientation='horizontal',
                    format=formatter,
                    shrink=0.6 ,
                    cax=cbar_ax)#ticks=[0, 1, 2, 3],format=r'$10^{%.1f}$',
    cb.set_label(r'n\'{u}mero por pixel')

    ax.text(s=r'$d\ \mathrm{[kpc]}$',  x=0.5, y=0.38,transform=ax.transAxes)
    ax.text(s=r'$l\ \mathrm{[graus]}$',x=1.05,y=.5,  transform=ax.transAxes)
    plt.tight_layout()
    #plt.savefig(f'polar_rrl_all.png',dpi=300,bbox_inches = 'tight',pad_inches=0.05)
    plt.show()
    plt.rcParams.update({'font.size': 12})






    # Separated by Extintion Determination
    
    rx = distances.x + GC_dist
    r = rx * np.cos(distances.gal_l.values*np.pi/180)
    l = distances.gal_l*np.pi/180
    masks = [msk1,msk2,msk3]
    labels = ['E(J-Ks)','E(H-Ks)','BEAM']
    for i in range(0,3):
        plt.rcParams.update({'font.size': 15})
        fig = plt.figure(figsize=[14,14])
        ax = fig.add_subplot(111, polar=True)
        ax.scatter(l[masks[i]], r[masks[i]]/1000,c='k', alpha=1, marker='.',s=2)
        ax.scatter(0,GC_dist/1000,marker='x',c='k',label="GC")
        # Red Clump Peaks position

        n = 0
        RC_dist1_mean = []
        RC_dist2_mean = []
        while n < 4:
            tile1 = Rc_peaks.index[n]
            tile2 = Rc_peaks.index[n+4]
            print(tile1, tile2)
            gal_l = Rc_peaks.loc[tile1,'tile_central_l']
            gal_b = Rc_peaks.loc[tile1,'tile_central_b']

            err_peak1  = Rc_peaks['RC_peak1_dist_sigma']
            mask1      = err_peak1 > 2000
            dist_peak1 = Rc_peaks['RC_peak1_dist']
            dist_peak1.loc[dist_peak1.index[mask1]] = np.nan
            err_peak1.loc[dist_peak1.index[mask1]] = np.nan

            err_peak2  = Rc_peaks['RC_peak2_dist_sigma']
            mask2      = err_peak2 > 2000
            dist_peak2 = Rc_peaks['RC_peak2_dist']
            dist_peak2.loc[dist_peak2.index[mask2]] = np.nan
            err_peak2.loc[dist_peak2.index[mask2]] = np.nan
            
            RC_dist1 = np.nanmean([dist_peak1.loc[tile1], dist_peak1.loc[tile2]])
            RC_dist1_err = np.sqrt(np.nansum([err_peak1.loc[tile1]**2, err_peak1.loc[tile2]**2]))/2
            RC_dist1_mean.append((gal_l*np.pi/180, RC_dist1/1000))

            RC_dist2 = np.nanmean([dist_peak2.loc[tile1], dist_peak2.loc[tile2]])
            RC_dist2_err = np.sqrt(np.nansum([err_peak2.loc[tile1]**2, err_peak2.loc[tile2]**2]))/2
            RC_dist2_mean.append((gal_l*np.pi/180, RC_dist2/1000))
            color = 'maroon'#'r'#plt.cm.Set1(0)#'m'#plt.cm.Accent(5)
            ax.errorbar(gal_l*np.pi/180,
                            RC_dist1/1000,
                            yerr=RC_dist1_err/1000,
                            marker="o",
                            mfc=color,
                            mec=color,
                            ecolor=color,
                            ms=5,
                            lw=1)

            ax.errorbar(gal_l*np.pi/180,
                            RC_dist2/1000,
                            yerr=RC_dist2_err/1000,
                            marker="o",
                            mfc=color,
                            mec=color,
                            ecolor=color,
                            ms=5,
                            lw=1)
            n+=1
        RC_dist1_mean = np.array(RC_dist1_mean).T
        RC_dist2_mean = np.array(RC_dist2_mean).T
        ax.plot(RC_dist1_mean[0],RC_dist1_mean[1],'-',c=color)
        ax.plot(RC_dist2_mean[0],RC_dist2_mean[1],'-',c=color)

        ax.set_thetamin(-1)
        ax.set_thetamax(8)
        ax.set_rlim((0, 12))
        ax.text(s='$d\ \mathrm{[kpc]}$',  x=0.5, y=0.38,transform=ax.transAxes)
        ax.text(s='$l\ \mathrm{[graus]}$',x=1.05,y=.5,  transform=ax.transAxes)
        ax.text(s=f'${labels[i]}$',         x=0,   y=0.5, transform=ax.transAxes)
        #ax.set_rorigin(-2000)
        plt.tight_layout()
        plt.savefig(f'polar_rrl_{labels[i]}_.png',dpi=300)
        #plt.show()
        plt.close()
        plt.rcParams.update({'font.size': 12})











    # RRL Milky Way MAP
    def line(x,theta,yshift):
        y = x * np.tan(theta) + yshift
        return y

    #sun_pixels = (2800,3870)
    sun_pixels = (2000,2760)
    distances = d.get_distance(magoffset=0,campain='variability',method='color')
    x = distances.x/1000
    y = distances.y/1000
    GC = GC_dist/1000
    
    import matplotlib.image as mpimg
    from astropy import units as u

    bins=(int(130*0.6),int(20*0.6))
    N, xedges, yedges = np.histogram2d(x,y,bins=bins)
    cmap = copy.copy(mpl.cm.get_cmap("jet"))# plt.cm.jet
    cmap.set_bad('w', 1.)
    cmap.set_bad(alpha=0)


    def plot_MW_map(mode='bulge',plot_clusters=True):
        '''
        mode = wide : wide view, includes sun and arms
        mode = full : total map
        mode = bulge : bulge view only'''
        # sigle plot
        if mode=='bulge':
            figsize = [8,8.5]
        else:
            figsize = [8,11]
        fig, ax = plt.subplots(1,1, figsize=figsize)
        #img = mpimg.imread(f'/home/botan/OneDrive/Doutorado/VVV_DATA/figuras_tese/ssc2008-10b1.tif')
        img = mpimg.imread(f'/home/botan/OneDrive/Doutorado/VVV_DATA/figuras_tese/eso1339g.tif')
        img_center =(int(img.shape[0]/2), int(img.shape[1]/2))
        pixel2pc_scale = GC/(sun_pixels[1] - img_center[1])
        xmin = +img_center[0]*pixel2pc_scale
        xmax = -img_center[0]*pixel2pc_scale
        ymin = -img_center[0]*pixel2pc_scale
        ymax =  img_center[0]*pixel2pc_scale

        ax.imshow(img,extent=[xmin,xmax,ymin,ymax])
        ax.scatter( y,
                    x,
                    marker='.',
                    s=12,
                    c='r',
                    edgecolors='none',
                    alpha=.4)
        
        ax.contour( yedges[:-1],
                    xedges[:-1],
                    N,
                    levels=3,
                    colors='k')
        img1 = ax.imshow(N,
                        norm=matplotlib.colors.LogNorm(),
                        #origin='lower',
                        extent=[ yedges[0], yedges[-1], xedges[-1], xedges[0]],
                        aspect='auto',
                        interpolation='none',
                        cmap=cmap,
                        alpha=0.3)

        ax.plot(0,0,'kx',label='GC')
        ax.text(x=-0.2,y=0,s='CG',ha='left',va='center')

        xp = np.array([-3,3])
        ax.plot(xp,line(xp,-20*np.pi/180,0)   , c='gray')
        ax.plot(xp,line(xp,-20*np.pi/180,2.8) , c='gray')
        ax.plot(xp,line(xp,-20*np.pi/180,-2.4), c='gray')



        n = 0
        RC_dist1_mean = []
        RC_dist2_mean = []
        while n < 4:
            tile1 = Rc_peaks.index[n]
            tile2 = Rc_peaks.index[n+4]
            print(tile1, tile2)
            gal_l = Rc_peaks.loc[tile1,'tile_central_l']
            gal_b = Rc_peaks.loc[tile1,'tile_central_b']

            err_peak1  = Rc_peaks['RC_peak1_dist_sigma']
            mask1      = err_peak1 > 2000
            dist_peak1 = Rc_peaks['RC_peak1_dist']
            dist_peak1.loc[dist_peak1.index[mask1]] = np.nan
            err_peak1.loc[dist_peak1.index[mask1]] = np.nan

            err_peak2  = Rc_peaks['RC_peak2_dist_sigma']
            mask2      = err_peak2 > 2000
            dist_peak2 = Rc_peaks['RC_peak2_dist']
            dist_peak2.loc[dist_peak2.index[mask2]] = np.nan
            err_peak2.loc[dist_peak2.index[mask2]] = np.nan
            
            RC_dist1 = np.nanmean([dist_peak1.loc[tile1], dist_peak1.loc[tile2]])
            RC_dist1_err = np.sqrt(np.nansum([err_peak1.loc[tile1]**2, err_peak1.loc[tile2]**2]))/2

            RC_dist2 = np.nanmean([dist_peak2.loc[tile1], dist_peak2.loc[tile2]])
            RC_dist2_err = np.sqrt(np.nansum([err_peak2.loc[tile1]**2, err_peak2.loc[tile2]**2]))/2
            
            x1,y1,z1 = d.cartezian_projections(RC_dist1,gal_l,0)
            x1err = RC_dist1_err*np.cos(gal_l*np.pi/180)
            y1err = RC_dist1_err*np.sin(gal_l*np.pi/180)
            RC_dist1_mean.append((x1/1000,y1/1000))
            
            x2,y2,z2 = d.cartezian_projections(RC_dist2,gal_l,0)
            x2err = RC_dist2_err*np.cos(gal_l*np.pi/180)
            y2err = RC_dist2_err*np.sin(gal_l*np.pi/180)
            RC_dist2_mean.append((x2/1000,y2/1000))

            color = 'maroon'#'r'#plt.cm.Set1(0)#'m'#plt.cm.Accent(5)
            ax.errorbar(y1/1000,
                        x1/1000,
                        xerr=y1err/1000,
                        yerr=x1err/1000,
                        marker="o",
                        mfc=color,
                        mec=color,
                        ecolor=color,
                        ms=5,
                        lw=1)

            ax.errorbar(y2/1000,
                        x2/1000,
                        xerr=y2err/1000,
                        yerr=x2err/1000,
                        marker="o",
                        mfc=color,
                        mec=color,
                        ecolor=color,
                        ms=5,
                        lw=1)
            n+=1
        RC_dist1_mean = np.array(RC_dist1_mean).T
        RC_dist2_mean = np.array(RC_dist2_mean).T
        ax.plot(RC_dist1_mean[1],RC_dist1_mean[0],'-',c=color)
        ax.plot(RC_dist2_mean[1],RC_dist2_mean[0],'-',c=color)



        if mode == 'wide':
            ax.set_xlim( 3, -3)
            ax.set_ylim(-8.2,  6)
            ax.plot(0,-8,'*',color='k',label='Sun')
            ax.text(x=-0.2,y=-8,s='Sol',ha='left',va='center')
        if mode == 'bulge':
            # clusters
            if plot_clusters:
                for n, cluster in enumerate(['NGC 6544', 'Djorg 2', 'Terzan 9', 'Terzan 10']):
                    gal_l_c = glob_clusters[cluster][0]
                    gal_b_c = glob_clusters[cluster][1]
                    d_c = cluster_distances[cluster][0]
                    xc, yc, zc = d.cartezian_projections(d_c,gal_l_c,gal_b_c)
                    ax.plot(yc/1000,xc/1000,'ko',markersize=10,markerfacecolor='none')
                    ax.text(x=yc/1000 ,y=xc/1000 - 0.1 ,s=n+1,c='k',ha='center',va='top',fontweight='bold')

            ax.set_xlim( 3, -3)
            ax.set_ylim(-3,  3)
            formatter = LogFormatter(10, labelOnlyBase=False) 
            cbar_ax = plt.axes([0.3, 0.88, 0.4, 0.01])
            cb = fig.colorbar(  img1,
                                ticks=[1,5,10,15,20,25],
                                orientation='horizontal',
                                format=formatter,
                                shrink=0.6 ,
                                cax=cbar_ax)
            cb.set_label(r'n\'{u}mero de estrelas')
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.xaxis.set_label_position('top')
        if mode == 'full':
            ax.set_xlim( 20, -20)
            ax.set_ylim(-20,  20)
            ax.plot(0,-8,'*',color='k',label='Sun')
            ax.text(x=-0.2,y=-8,s='Sol',ha='left',va='center')
        ax.set_aspect('equal')
        ax.set_xlabel('y [kpc]')
        ax.set_ylabel('x [kpc]')

        plt.savefig(f'MW_map_{mode}.png',dpi=300,pad_inches=0.05)
        plt.show()






    # multi plot
    fig, ax = plt.subplots(1,5, figsize=[20,10],tight_layout=True,gridspec_kw={'width_ratios': [13.5,3,3,3,3]})
    img = mpimg.imread(f'/home/botan/OneDrive/Doutorado/VVV_DATA/figuras_tese/ssc2008-10b1.tif')
    ax[0].imshow(img)
    lyr2pixel = (sun_pixels[1] - img.shape[1]/2.) / (GC_dist*u.parsec.to(u.lightyear))
    ax[0].scatter(  -ylyr*lyr2pixel+img.shape[1]/2.,
                    -xlyr*lyr2pixel+img.shape[1]/2.,
                    marker='.',
                    s=1,
                    c='r',
                    alpha=.4)
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    ax[1].imshow(img)
    ax[1].scatter(  -ylyr*lyr2pixel+img.shape[1]/2.,
                    -xlyr*lyr2pixel+img.shape[1]/2.,
                    marker='.',
                    s=1,
                    c='r',
                    alpha=.6)
    ax[1].set_xlim(2526,3080)
    ax[1].set_ylim(4130,1620)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_yticks([])
    ax[1].set_xticks([])

    ax[2].imshow(img)
    ax[2].scatter(  -ylyr[msk1]*lyr2pixel+img.shape[1]/2.,
                    -xlyr[msk1]*lyr2pixel+img.shape[1]/2.,
                    marker='.',
                    s=1,
                    c='r',
                    alpha=.6)
    ax[2].set_xlim(2526,3080)
    ax[2].set_ylim(4130,1620)
    ax[2].set_yticklabels([])
    ax[2].set_xticklabels([])
    ax[2].set_yticks([])
    ax[2].set_xticks([])

    ax[3].imshow(img)
    ax[3].scatter(  -ylyr[msk2]*lyr2pixel+img.shape[1]/2.,
                    -xlyr[msk2]*lyr2pixel+img.shape[1]/2.,
                    marker='.',
                    s=1,
                    c='r',
                    alpha=.6)
    ax[3].set_xlim(2526,3080)
    ax[3].set_ylim(4130,1620)
    ax[3].set_yticklabels([])
    ax[3].set_xticklabels([])
    ax[3].set_yticks([])
    ax[3].set_xticks([])

    ax[4].imshow(img)
    ax[4].scatter(  -ylyr[msk3]*lyr2pixel+img.shape[1]/2.,
                    -xlyr[msk3]*lyr2pixel+img.shape[1]/2.,
                    marker='.',
                    s=1,
                    c='r',
                    alpha=.6)
    ax[4].set_xlim(2526,3080)
    ax[4].set_ylim(4130,1620)
    ax[4].set_yticklabels([])
    ax[4].set_xticklabels([])
    ax[4].set_yticks([])
    ax[4].set_xticks([])
    plt.show()





    # cone view 2:
    for msk,label in zip([msk1,msk2,msk3],['E(J-Ks)','E(H-Ks)','BEAM']):
        fig = plt.figure(figsize=[15,15])
        ax = fig.add_subplot(111, polar=True)
        rx = distances.x + GC_dist
        r = rx * np.cos(distances.gal_l.values*np.pi/180)
        #ax.scatter(distances.gal_l[msk3]*np.pi/180, r[msk3], alpha=.7, marker='.',s=2,label='BEAM')
        #ax.scatter(distances.gal_l[msk2]*np.pi/180, r[msk2], alpha=.7, marker='.',s=2,label='E(H-Ks)')
        ax.scatter(distances.gal_l[msk]*np.pi/180, r[msk], alpha=.7, marker='.',s=2,label=label)
        #ax.scatter(distances.gal_l[msk2]*np.pi/180, r[msk2], alpha=.7, marker='.',s=2,label='E(H-Ks)')
        #ax.scatter(distances.gal_l[msk3]*np.pi/180, r[msk3], alpha=.7, marker='.',s=2,label='BEAM')
        ax.scatter(0,GC_dist,marker='x',c='k',label="GC")
        ax.set_thetamin(-1)
        ax.set_thetamax(8)
        ax.set_rmax(16000)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'polar_rrl_{label}.png',dpi=300)
        plt.show()



    fig = plt.figure(figsize=[8,8])
    ax = fig.add_subplot(111)
    rx = distances.x + GC_dist
    ry = distances.y
    ax.scatter(rx, ry, alpha=1, marker='.',s=2)
    ax.scatter(0,0,marker='*',label='Sun')
    ax.text(s="Sun",x=0,y=-60,ha='center')
    ax.scatter(GC_dist,0,marker='x',c='k',label="GC")
    ax.text(s="GC",x=GC_dist,y=-60,ha='center')
    plt.tight_layout()
    plt.show()



    # cone view - xy
    fig = plt.figure(figsize=[15,5])
    ax = fig.add_subplot(111)
    rx = distances.x
    ry = distances.y


    bins=(int(120*0.6),int(20*0.6) )
    cmap = copy.copy(mpl.cm.get_cmap("jet"))# plt.cm.jet
    cmap.set_bad('w', 1.)
    cmap_multicolor = copy.copy(mpl.cm.get_cmap("jet")) # plt.cm.jet
    cmap_multicolor.set_bad('w', 1.)
    N, xedges, yedges = np.histogram2d(rx,ry,bins=bins)
    img = ax.imshow(np.log10(N.T),
                    origin='lower',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    aspect='auto',
                    interpolation='nearest',
                    cmap=cmap)


    ax.scatter(rx, ry, alpha=.7, marker='.',s=2)
    
    ax.scatter(0,0,marker='x',c='k',label="GC")
    ax.set_aspect('equal', 'box')
    ax.text(s="GC",x=0,y=60,ha='center')
    plt.tight_layout()
    plt.show()
    
    



    # RRL Ks Histogram
    mag_range = [11,18]
    rrl_Ks_mag1 = d.all_dat.loc[rrl_ids].mag_Ks
    rrl_Ks_mag2 = d.all_dat.loc[rrl_ids].lc_mean
    fig, ax = plt.subplots(figsize=(7,3.5))
    fig.subplots_adjust(wspace=0,hspace=0)
    ax.hist(rrl_Ks_mag2,
                bins=40,
                label='Média da curva de luz',
                range=mag_range,
                histtype='barstacked',
                lw=.5,
                color='k',
                edgecolor='w',
                alpha=0.6)
    ax.hist(rrl_Ks_mag1,
                bins=40,
                label='Campanha de cor',
                range=mag_range,
                histtype='barstacked',
                lw=.5,
                color='dodgerblue',
                edgecolor='w',
                alpha=0.6)
    ax.set_xlabel(r"$\mathrm{K_s\ [mag]}$")
    ax.set_ylabel(r'$\mathrm{\#\ estrelas}$')
    ax.legend()#prop={'size': 10})
    plt.savefig('RRL_Ks_hist.png',dpi=300,bbox_inches='tight',pad_inches=0.05)
    plt.show()





    # RRL CMD
    def plot_RRL_CMD(passband='J',font_size=13):
        distances = d.get_distance(magoffset=0,campain='variability',method='color')
        if passband == 'J':
            #params dict [cmin,cmax,ymin,ymax,xmin,xmax]
            params_dict = { 'b293':[0.85,1.00,11.1,18.1,0.1,2.9],
                            'b294':[0.86,1.00,11.1,18.1,0.1,2.9],
                            'b295':[0.95,1.20,11.1,18.1,0.1,2.9],
                            'b296':[1.05,1.40,11.1,18.1,0.1,2.9],
                            'b307':[1.00,1.40,11.1,18.1,0.1,2.9],
                            'b308':[1.19,1.71,11.1,18.1,0.1,2.9],
                            'b309':[1.19,1.71,11.1,18.1,0.1,2.9],
                            'b310':[1.45,2.00,11.1,18.1,0.1,2.9]}
            extintion = distances['E(J-Ks)'].mean()
            reddening = extintion * 0.689
            xlabel='(J-Ks)'
            ckey = 'mag_J'
            arrow_xpos = 2.1


        if passband == 'H':
            params_dict = { 'b293':[0.85,1.00,11.1,18.1,-0.2,1.2],
                            'b294':[0.86,1.00,11.1,18.1,-0.2,1.2],
                            'b295':[0.95,1.20,11.1,18.1,-0.2,1.2],
                            'b296':[1.05,1.40,11.1,18.1,-0.2,1.2],
                            'b307':[1.00,1.40,11.1,18.1,-0.2,1.2],
                            'b308':[1.19,1.71,11.1,18.1,-0.2,1.2],
                            'b309':[1.19,1.71,11.1,18.1,-0.2,1.2],
                            'b310':[1.45,2.00,11.1,18.1,-0.2,1.2]}
            extintion = distances['E(H-Ks)'].mean()
            reddening = extintion * 1.888
            xlabel='(H-Ks)'
            ckey = 'mag_H'
            arrow_xpos = 0.9

        # CMD axes dict
        axes_dict   = { 'b293':[1,3],
                        'b294':[1,2],
                        'b295':[1,1],
                        'b296':[1,0],
                        'b307':[0,3],
                        'b308':[0,2],
                        'b309':[0,1],
                        'b310':[0,0]}

        
        plt.rcParams.update({'font.size': font_size})
        fig, axes = plt.subplots(2, 4, figsize=(16,8))
        fig.subplots_adjust(wspace=0,hspace=0)
        tiles = sorted(os.listdir(f'{path}/data/psf_ts/'))
        num = 0
        for tile in tiles:
            tileData = []
            chips = [_[:-3] for _ in os.listdir(f'{path}/data/psf_ts/{tile}/chips/') if _.endswith('.ts')]
            for chip in chips:
                chipData = pd.read_csv(f'{path}/data/psf_ts/{tile}/chips/{chip}.ts',index_col='ID')
                tileData.append(chipData)
        
            tileData = pd.concat(tileData)
            tileData = tileData.drop_duplicates()

            color = tileData[ckey] - tileData.mag_Ks

            msk   = ~color.isnull()
            mag   = tileData.mag_Ks
            mag   = mag[msk]
            color = color[msk]

            xlim= params_dict[tile][4:6]
            ylim= params_dict[tile][2:4]


            bins=(600,400)
            cmap = copy.copy(mpl.cm.get_cmap("jet"))# plt.cm.jet 
            cmap.set_bad('w', 1.)
            cmap_multicolor = copy.copy(mpl.cm.get_cmap("jet")) # plt.cm.jet
            cmap_multicolor.set_bad('w', 1.)
            N, xedges, yedges = np.histogram2d(color,mag,bins=bins)
            ax1 = axes_dict[tile][0]
            ax2 = axes_dict[tile][1]
            img = axes[ax1,ax2].imshow(np.log10(N.T), origin='lower',
                                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                        aspect='auto', interpolation='nearest', cmap=cmap)
            axes[ax1,ax2].text(s=f'Tile: {tile}',x=0.73,y=0.02,ha='left',transform=axes[ax1,ax2].transAxes)
            # RRL CMD
            tile_rrl = [_ for _ in rrl_ids if _ in mag.index]
            num+=len(tile_rrl)
        
            #rrl_mag = mag.loc[tile_rrl]
            rrl_mag = d.all_dat.loc[tile_rrl].lc_mean
            axes[ax1,ax2].scatter(  color.loc[tile_rrl],
                                    rrl_mag.loc[tile_rrl],
                                    marker='.',
                                    c='k',
                                    s=2)

            axes[ax1,ax2].set_xlim(xlim)
            axes[ax1,ax2].set_ylim(ylim)
            axes[ax1,ax2].set_xlabel(r'$\mathrm{%s}$'%xlabel)
            axes[ax1,ax2].set_ylabel(r'$\mathrm{K_s\ [mag]}$')
            axes[ax1,ax2].invert_yaxis()
            if ax2 > 0:
                axes[ax1,ax2].set_yticks([])
            for im in plt.gca().get_images():
                im.set_clim(0, 3)
            # reddening vector
            axes[ax1,ax2].annotate("", xy=(arrow_xpos+extintion, 11.4+reddening),
                                     xytext=(arrow_xpos, 11.4),
                                     arrowprops=dict(arrowstyle="->", color='r'))
            
        for ax in fig.get_axes():
            ax.label_outer()
        cbar_ax = plt.axes([0.91, 0.2, 0.01, 0.6])
        cb = fig.colorbar(img, 
                        ticks=[0, 1, 2, 3],
                        format=r'$10^{%i}$',
                        shrink=0.6 ,
                        cax=cbar_ax)
        cb.set_label(r'n\'{u}mero por pixel',rotation=90)
        #cb.set_label(r'$\mathrm{number\ in\ pixel}$',rotation=90)
        #plt.tight_layout()
        plt.savefig(f'CMD_RRL_{ckey}.png',dpi=200,bbox_inches='tight',pad_inches=0.05)
        plt.show()






    # GAIA Bailer Jones distances 
    plt.rcParams.update({'font.size': 13})
    import matplotlib.gridspec as gridspec
    rrl_ks_mag = d.all_dat.loc[rrl_ids].lc_mean
    rrl_dist = d.all_dat.loc[rrl_ids].rest /1000
    low_lim = d.all_dat.loc[rrl_ids].b_rest_x /1000
    upp_lim = d.all_dat.loc[rrl_ids].B_rest_xa /1000
    rrl_dist_err = np.array([rrl_dist.values - low_lim.values , upp_lim.values - rrl_dist.values])

    fig = plt.figure(figsize=[7,8])
    gs0 = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2,1])
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[0], wspace=0, hspace=0)
    
    # Gaia Distances vs mag
    ax1 = fig.add_subplot(gs1[0])
    ax1.errorbar( x=rrl_ks_mag.values, 
                y=rrl_dist.values, 
                yerr=rrl_dist_err,
                marker='.', capsize=2, 
                elinewidth=0.8,fmt='.', 
                mec='none', mfc='k',
                ms=8,  ecolor='r',alpha=.5)
    #ax.xaxis.set_minor_locator(AutoMinorLocator())
    #ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.axhline(GC_dist/1000,lw=.8,c='gray')
    #ax1.set_xlabel('$K_s\ \mathrm{[mag]}$')
    ax1.set_ylabel('$d\ \mathrm{[kpc]}$')
    ax1.set_xlim(10.9,16.1)
    ax1.set_xticklabels([])

    # Gaia magnitude histogram
    ax2 = fig.add_subplot(gs1[1])
    histrange = [11,16]
    rrl_mag = d.all_dat.loc[rrl_ids,'lc_mean']
    msk = ~(rrl_dist.isna())
    ax2.hist( rrl_mag,
                bins=30,
                range=histrange,
                histtype='barstacked',
                lw=.5,
                color='dodgerblue',
                edgecolor='w',
                alpha=0.7,
                label='RRL identificadas')
    ax2.hist( rrl_mag[msk],
                bins=30,
                range=histrange,
                histtype='barstacked',
                lw=.5,
                color='k',
                edgecolor='w',
                alpha=0.6,
                label='RRL com distância no Gaia')
    ax2.legend()
    ax2.set_xlabel('$K_\mathrm{s}\ \mathrm{[mag]}$')
    ax2.set_ylabel('$\#\ estrelas$')
    ax2.set_xlim(10.9,16.1)
    ax_t = ax2.secondary_xaxis('top')
    ax_t.tick_params(axis='x', direction='inout',length=6)
    ax_t.set_xticklabels([])

    # Histogram distances from Gaia
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[1])
    ax3 = fig.add_subplot(gs2[0])
    ax3.hist( rrl_dist[msk],
                bins=30,
                histtype='barstacked',
                lw=.5,
                color='k',
                edgecolor='w',
                alpha=0.6)
    ax3.set_xlabel('$d\ \mathrm{[kpc]}$')
    ax3.set_ylabel('$\#\ estrelas$')

    plt.tight_layout()
    plt.savefig(f'rrl_dist_BJ.png',
                dpi=300, 
                bbox_inches = 'tight',
                pad_inches = 0.05)

    plt.show()
    plt.close()



    # Gaia vs VVV distances
    fig = plt.figure(figsize=[8,8])
    ax = fig.add_subplot()
    ax.errorbar(x=distances.distance/1000, 
                y=rrl_dist, 
                xerr=distances.distanceSigma/1000,
                yerr=rrl_dist_err,
                marker='.', capsize=2, 
                elinewidth=0.8,fmt='.', 
                mec='none', mfc='k',
                ms=8,  ecolor='r',alpha=.5)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0,15)
    ax.set_ylim(0,15)
    ax.set_xlabel('$\mathrm{VVV} \ d \ \mathrm{[kpc]}$')
    ax.set_ylabel('$\mathrm{Gaia} \ d \ \mathrm{[kpc]}$')
    plt.tight_layout()
    plt.savefig(f'rrl_Gaia_vs_VVV.png',
                dpi=300, 
                bbox_inches = 'tight',
                pad_inches = 0.05)
    plt.show()
    



    #fig, ax = plt.subplots(4,1,figsize=[7,10],gridspec_kw={'height_ratios': [2, 1.5, 1.5,1.5]},tight_layout=True)
    fig.subplots_adjust(wspace = 0)
    ax[0].errorbar( x=rrl_ks_mag.values, 
                    y=rrl_dist.values, 
                    yerr=rrl_dist_err,
                    marker='.', capsize=2, 
                    elinewidth=0.8,fmt='.', 
                    mec='none', mfc='k',
                    ms=8,  ecolor='r',alpha=.5)
    #ax.xaxis.set_minor_locator(AutoMinorLocator())
    #ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax[0].axhline(GC_dist/1000,lw=.8,c='gray')
    #ax[0].set_xlabel('$K_s\ \mathrm{[mag]}$')
    ax[0].set_ylabel('$d\ \mathrm{[kpc]}$')
    ax[0].set_xlim(10.9,16.1)
    ax[0].set_xticklabels([])
    
    # histogram mag
    histrange = [11,16]
    rrl_mag = d.all_dat.loc[rrl_ids,'lc_mean']
    msk = ~(rrl_dist.isna())
    ax[1].hist( rrl_mag,
                bins=30,
                range=histrange,
                histtype='barstacked',
                lw=.5,
                color='dodgerblue',
                edgecolor='w',
                alpha=0.7,
                label='Todas as RRL identificadas')
    ax[1].hist( rrl_mag[msk],
                bins=30,
                range=histrange,
                histtype='barstacked',
                lw=.5,
                color='k',
                edgecolor='w',
                alpha=0.6,
                label='RRL com distância no Gaia')
    ax[1].legend()
    ax[1].set_xlabel('$K_\mathrm{s}\ \mathrm{[mag]}$')
    ax[1].set_ylabel('$\#\ estrelas$')
    ax[1].set_xlim(10.9,16.1)

    # histogram distance
    
    ax[2].hist( rrl_dist[msk],
                bins=30,
                histtype='barstacked',
                lw=.5,
                color='dodgerblue',
                edgecolor='w',
                alpha=0.7)
    ax[2].set_xlabel('$d\ \mathrm{[kpc]}$')
    ax[2].set_ylabel('$\#\ estrelas$')

    


    plt.savefig(f'rrl_dist_BJ.png',
                dpi=300, 
                bbox_inches = 'tight',
                pad_inches = 0.05)

    plt.show()
    plt.close()


