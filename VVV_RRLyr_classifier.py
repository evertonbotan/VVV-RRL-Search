# -*- coding: utf-8 -*-
"""
Created on Apr 04 2020

@author: Everton Botan
@supervisor: Roberto Saito

Search and classify RR Lyrae stars into VVV data.
"""
import os
import sys
import numpy as np
import math
import pandas as pd
from PyAstronomy.pyTiming import pyPDM
from astropy.timeseries import LombScargle
from astropy.io import ascii
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import match_coordinates_sky, search_around_sky
from scipy.optimize import curve_fit
from scipy import optimize, signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from termcolor import colored




class StarClassifier(object):
    
    def __init__(self, tile):
        # minVal: minimum period with same unit as time from light curve (float)
        # maxVal: maximum period with same unit as time from light curve (float)
        # periodogram: LombScargle or PDM
        # varIndex: std or chi2
        #super(StarClassifier,self).__init__(tile,raw_files=False)
        self.path = 'data/psf_ts/%s/'%(tile)
        self.tile = tile
        self.parameters_files = [fn for fn in os.listdir(f'{self.path}output/') if fn.endswith("clean")]
        self.ts_files = [fn for fn in os.listdir(f'{self.path}output/') if fn.endswith("ts")] 
        self.ts_files.sort()
        self.rr_lyr_z = 0.0025 #metalicity from Alonso-García (2014): Variable stars in VVV globular clusters I 
        self.zyjhk = ascii.read("data/zyjhk/zyjhk%s.cals"%(tile[1:])).to_pandas()
        self.zyjhk.rename(columns = {"ra":"RA","dec":"DEC"}, inplace = True)

    def read_tables(self,chip):
        self.ogle_iv_table = pd.read_csv("ogle_iv_bulge/ogle_iv.txt",sep="\t",index_col="ID",na_values=-99.99)
        self.vsx_table = pd.read_csv("vsx/vsx.csv",sep=",",index_col="ID")
        self.parameters  = pd.read_csv("%s/output/%s_LS_parameters_clean"%(self.path,chip[:-3]),delimiter=",",index_col="ID")
        data_table       = pd.read_csv(f'{self.path}output/{chip}',index_col=0,sep=' ')
        mag_color_cols   = [col for col in data_table.columns if col.split("_")[0] == "mag"]
        err_color_cols   = [col for col in data_table.columns if col.split("_")[0] == "er"]
        mag_cols         = [col for col in data_table.columns if col.split("_")[0] == "MAG"]
        err_cols         = [col for col in data_table.columns if col.split("_")[0] == "ERR"]
        coord_cols       = ["RA","DEC"]
        self.data_table  = data_table
        self.mag_colors  = data_table[mag_color_cols]
        self.err_colors  = data_table[err_color_cols]
        self.mag_table   = data_table[mag_cols]
        self.err_table   = data_table[err_cols]
        self.coord_table = data_table[coord_cols]
        self.obs_time    = np.loadtxt(f'{self.path}output/{chip.split(".")[0]+".mjd"}')
        self.ogle_type    = data_table["OGLE_Subtype"]
        self.vsx_type     = data_table["VSX_Type"]

    def read_data(self):
        tiles = sorted(os.listdir('/home/botan/OneDrive/Doutorado/VVV_DATA/data/psf_ts/'))
        duplicates = pd.read_csv('/home/botan/OneDrive/Doutorado/VVV_DATA/data/chip_overlap_ids.csv',index_col=0)
        
        rrl_fitparams = []
        for tile  in tiles:
            path = f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/psf_ts/{tile}/lc_plots/short_period/pos_visual_inspection'
            rrl_fitparams.append(pd.read_csv(f'{path}/{tile}_rrlyr_bona_parameters.csv',sep=',',index_col='ID'))
        rrl_fitparams = pd.concat(rrl_fitparams)
        id2drop = []
        for _ in rrl_fitparams.index: 
            if _ in duplicates.index: 
                for col in duplicates.columns: 
                    star2drop = duplicates.loc[_,col] 
                    if star2drop not in id2drop: 
                        id2drop.append(star2drop)
        self.rrl_ids = [_ for _ in rrl_fitparams.index if _ not in id2drop]
        
        self.rrl_fitparams = rrl_fitparams.loc[self.rrl_ids]
        self.BEAM_extintion = pd.read_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/all_variables_extintion.csv',index_col='ID')
        self.extintion3D = pd.read_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/3D_Extintion_Map/table1jk.csv')
        self.all_dat = pd.read_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/all_variables_match_vsx_ogle_gaia_viva.csv',index_col='ID')
        

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

    def Ks_mag(self,period,metallicity,offset=0):
        abs_ks = - 0.6365 - 2.347*np.log10(period) + 0.1747*np.log10(metallicity) + offset
        # this theoretical absolute magnitude has errors bellow survey photometric precision.
        # Thus it has been ignored.
        return abs_ks

    def H_mag(self,period,metallicity,offset=0):
        abs_h = - 0.5539 - 2.302*np.log10(period) + 0.1781*np.log10(metallicity) + offset
        return abs_h

    def J_mag(self,period,metallicity,offset=0):
        abs_j = - 0.2361 - 1.830*np.log10(period) + 0.1886*np.log10(metallicity) + offset
        return abs_j
    
    def Y_mag(self,period,metallicity,offset=0):
        abs_y = 0.0090 - 1.467*np.log10(period) + 0.1966*np.log10(metallicity) + offset
        return abs_y
    
    def Z_mag(self,period,metallicity,offset=0):
        abs_z = 0.1570 - 1.247*np.log10(period) + 0.2014*np.log10(metallicity) + offset
        return abs_z

    def extintion(self, magA, magAerr, magB, magBerr, abs_magA, abs_magB):
        #color excess
        extintion = (magA - magB) - (abs_magA - abs_magB)
        sigma     = np.sqrt(magAerr**2 + magBerr**2)
        return extintion,sigma

    def reddenig(self,extintion,extintionSigma,redIndex=0.689):
        # for AKs, redIndex = 0.689, for E(J-Ks) and Cardelli law
        # for AJ, redIndex = 1.692, for E(J-Ks) and Cardelli law
        red   = redIndex*extintion
        sigma = redIndex*extintionSigma
        return red,sigma

    def red_free_mag(self,mag,err,red_mag,red_err):
        mag_0 = mag - red_mag
        sigma = np.sqrt(err**2 + red_err**2)
        return mag_0,sigma

    def rrl_distance(self,Ks_0,ErrKs,AbsKs):
        dist  = 10**(1 + (Ks_0 - AbsKs)/5) #in parsec
        sigma = 2**(1 + (Ks_0 - AbsKs)/5) * 5**((Ks_0 - AbsKs)/5) * np.log(10) * ErrKs
        return dist,sigma

    def get_distances(self,MagKs,ErrKs,MagJ,ErrJ,MagH,ErrH,
                      beam_E_JK,beam_E_JK_err,period,
                      abundanceMode=1,magoffset=0):
        # MagKs is the Ks mean magnitude from light curve
        # MagJ and MagH are the magnitudes from color campain
        FeH, FeH_sigma   = self.Fe_abundance(period, mode=abundanceMode)
        metallicity      = self.metallicity_from_Fe_abundance(FeH,FeH_sigma,alpha=0.3)
        AbsKs            = self.Ks_mag(period,metallicity,offset=magoffset)
        if MagJ != -99:
            # if J magnitude is availeable we calculate extintion
            # from the difference between observed and intrinsec magnitude
            redIndex           = 0.689
            AbsJ               = self.J_mag(period,metallicity,offset=magoffset)
            E_JKs, E_JKS_sigma = self.extintion(MagKs,ErrKs,MagJ,ErrJ,AbsKs,AbsJ)
            AKs, AKs_sigma     = self.reddenig(E_JKs,E_JKS_sigma,redIndex=redIndex)
            extintion_flag     = 1
        else:
            if MagH != -99:
                # if J is missing but not H we calculate extintion
                # from the difference between observed and intrinsec magnitude
                redIndex           = 1.888
                AbsH               = self.H_mag(period,metallicity,offset=magoffset)
                E_HKs, E_HKS_sigma = self.extintion(MagKs,ErrKs,MagH,ErrH,AbsKs,AbsH)
                AKs, AKs_sigma     = self.reddenig(E_HKs,E_HKS_sigma,redIndex=redIndex)
                extintion_flag     = 2
            else:
                # if J and H magnitude is missing we used BEAM.
                redIndex       = 0.689
                E_JKs,         = beam_E_JK
                E_JKS_sigma    = beam_E_JK_err
                AKs,AKs_sigma  = self.reddenig(E_JKs,E_JKS_sigma,redIndex=redIndex)
                extintion_flag = 3

        Ks_0, Ks_0_err   = self.red_free_mag(MagKs,ErrKs,AKs,AKs_sigma)
        dist, dist_err   = self.rrl_distance(Ks_0,Ks_0_err,AbsKs)
        return dist, dist_err, extintion_flag
    
    def fourier_series(self, ):
        pass




    def rr_period_selection(self):
        periods = 1./self.parameters.best_freq
        #RRab
        msk = ((periods < 0.8) & (periods > 0.4))
        rrab_id = periods.index[msk]
        #RRc
        msk = ((periods < 0.4) & (periods > 0.2))
        rrc_id = periods.index[msk]
        return rrab_id, rrc_id

    def rr_color_selection(self):
        crange = {}
        #RRab
        stars = [fn[3:-4] for fn in os.listdir("%s/figures/"%self.path) if fn.endswith("png")]
        msk = 1
        return rrab_id, rrc_id

    def absMag2apMag(self,M,d):
        # M : absolute magnitude
        # d : distance in parsec
        m = 5*np.log10(d/10) + M
        return m

    def get_ogle_from_tiles(self):
        data = []
        i = 1
        for chip in self.ts_files:
            print("Hey, anxious?. Keep calm, I'm doing it: ", chip[:-3], " [%i/%i]"%(i,len(self.ts_files)))
            data_table = pd.read_csv(f'{self.path}output/{chip}',index_col="ID",sep=' ')
            msk = ~data_table.OGLE_ID.isna()
            out = data_table[msk]
            out.loc[out.index, "chip"] = chip[:-3]
            out.loc[out.index, "tile"] = self.tile
            data.append(out)
            i+=1
        data = pd.concat(data, sort=False)
        return data

    def get_vsx_from_tiles(self):
        data = []
        for chip in self.ts_files:
            data_table = pd.read_csv(f'{self.path}output/{chip}',index_col="ID",sep=' ')
            msk = ~data_table.VSX_ID.isna()
            out = data_table[msk]
            out.loc[out.index, "chip"] = chip[:-3]
            out.loc[out.index, "tile"] = self.tile
            data.append(out)
        data = pd.concat(data, sort=False)
        return data

    def plot_color(self, filter1="mag_J", filter2="mag_Ks",show=False):

        mag_j  = self.zyjhk[filter1]
        mag_ks = self.zyjhk[filter2]
        msk = ~(mag_j-mag_ks).isna()
        
        ogle_dat = pd.read_csv("ogle_dat.dat",sep=",", index_col="ID")

        rr_lyr =[]
        for tile in ["b293","b294","b295","b296"]:
            rr_lyr += [fn[3:-4] for fn in os.listdir("data/psf_ts/%s/figures/RRLyr/"%tile) if fn.endswith("png")]

        ogle_rrab = ogle_dat[ogle_dat.OGLE_Subtype == "RRab"].index
        ogle_rrc  = ogle_dat[ogle_dat.OGLE_Subtype == "RRc"].index
        rrab_id = [x for x in ogle_rrab if x in rr_lyr]
        rrc_id  = [x for x in ogle_rrc  if x in rr_lyr]

        plt.figure(figsize=[15,10])
        plt.hist2d((mag_j-mag_ks)[msk],mag_ks[msk],bins=500,norm=mpl.colors.LogNorm(),cmap=mpl.cm.copper,alpha=.2,lw=0)
        cbar = plt.colorbar()
        cbar.set_label('# stars')
        
        f1 = ogle_dat[filter1]
        f2 = ogle_dat[filter2]
        plt.scatter(f1.loc[rrc_id] - f2.loc[rrc_id], f2.loc[rrc_id], s=1,c="r",label="OGLE RRc") 
        plt.scatter(f1.loc[rrab_id] - f2.loc[rrab_id], f2.loc[rrab_id], s=1,c="b",label="OGLE RRab")
        
        plt.gca().invert_yaxis()
        minx = (mag_j-mag_ks)[msk].sort_values().iloc[50]
        maxx = (mag_j-mag_ks)[msk].sort_values().iloc[-50]
        plt.xlim(minx,maxx)
        plt.xlabel("%s - %s"%(filter1.split('_')[-1],filter2.split('_')[-1]))
        plt.ylabel("%s"%(filter2.split('_')[-1]))

        #isocroone
        isoc = np.genfromtxt("isocrone_age_12gyr_z_25e-4.dat",skip_header=11,skip_footer=1,names=True)
        f1_isoc = self.absMag2apMag(isoc[filter1], 8e3)
        f2_isoc = self.absMag2apMag(isoc[filter2], 8e3)
        plt.scatter(f1_isoc-f2_isoc,f2_isoc, 
                    marker="s", 
                    edgecolors="r", 
                    facecolors='none', 
                    label="isochrone Z = 0.0025 12 Gyr Av = 0.0")
        
        isoc = np.genfromtxt("isocrone_age_10gyr_z_25e-4.dat",skip_header=11,skip_footer=1,names=True)
        f1_isoc = self.absMag2apMag(isoc[filter1], 8e3)
        f2_isoc = self.absMag2apMag(isoc[filter2], 8e3)
        plt.scatter(f1_isoc-f2_isoc,f2_isoc, 
                    marker="s", 
                    edgecolors="b", 
                    facecolors='none', 
                    label="isochrone Z = 0.0025 10 Gyr Av = 0.0")

        plt.legend()
        
        # getting limits for color: 95% == 2 sigmas.
        c_rrab = f1.loc[rrab_id] - f2.loc[rrab_id]
        c_rrc  = f1.loc[rrc_id] - f2.loc[rrc_id]
        clip = sigma_clip(c_rrab,sigma=2).mask
        c_rrab = c_rrab[~clip]
        clip = sigma_clip(c_rrc,sigma=2).mask
        c_rrc = c_rrc[~clip]
        color = "%s - %s"%(filter1.split('_')[-1],filter2.split('_')[-1])
        print(colored("RRab: %.2f < %s < %.2f"%(c_rrab.min(), color, c_rrab.max()),"yellow"))
        print(" ")
        print(colored("RRc: %.2f < %s < %.2f"%(c_rrc.min(), color, c_rrc.max()),"yellow"))

        # plotting limits for color: 95% == 2 sigmas.
        ymin = mag_j[msk].min()
        ymax = mag_j[msk].max()

        plt.vlines(x=c_rrab.min(),ymin=ymin,ymax=ymax,colors="b",lw=.4)
        plt.vlines(x=c_rrab.max(),ymin=ymin,ymax=ymax,colors="b",lw=.4)

        plt.vlines(x=c_rrc.min(),ymin=ymin,ymax=ymax,colors="r",lw=.4)
        plt.vlines(x=c_rrc.max(),ymin=ymin,ymax=ymax,colors="r",lw=.4)
        
        plt.savefig("RRLyr_color_%s_%s.png"%(filter1.split('_')[-1],filter2.split('_')[-1]),dpi=300)
        if show:
            plt.show()
        plt.close()
        

        # histograms
        c_rrab = f1.loc[rrab_id] - f2.loc[rrab_id]
        c_rrc  = f1.loc[rrc_id] - f2.loc[rrc_id]
        c_rrab_mean = c_rrab.mean()
        c_rrab_median = c_rrab.median()
        c_rrc_mean = c_rrc.mean()
        c_rrc_median = c_rrc.median()
        plt.figure(figsize=[8,4])
        plt.hist(c_rrab,bins=10, label="RRab",histtype="step",color="b")
        plt.hist(c_rrc,bins=10, label="RRab",histtype="step",color="r")
        plt.axvline(c_rrab_median, color='b', linestyle='dashed', linewidth=1,label="median: %.2f"%c_rrab_median)
        plt.axvline(c_rrc_median,  color='r', linestyle='dashed', linewidth=1,label="median: %.2f"%c_rrc_median)
        #plt.axvline(c_rrab_mean, color='k', linestyle='dashed', linewidth=1)
        #plt.axvline(c_rrc_mean, color='k', linestyle='dashed', linewidth=1)
        plt.xlabel("%s - %s"%(filter1.split('_')[-1],filter2.split('_')[-1]))
        plt.ylabel("# stars")
        plt.legend()
        plt.tight_layout()
        plt.savefig("RRLyr_color_hist_%s_%s.png"%(filter1.split('_')[-1],filter2.split('_')[-1]),dpi=300)
        if show:
            plt.show()
        plt.close

        return {'RRab_min': round(c_rrab.min(), 2), 
                'RRab_max': round(c_rrab.max(), 2), 
                "RRc_min":  round(c_rrc.min(),  2), 
                "RRc_max":  round(c_rrc.max(),  2)}


    def plot_period_dist(self,show=True):
        ogle_dat = pd.read_csv("ogle_dat.dat",sep=" ", index_col="ID")

        rr_lyr =[]
        for tile in ["b293","b294","b295","b296"]:
            rr_lyr += [fn[3:-4] for fn in os.listdir("data/psf_ts/%s/figures/RRLyr/"%tile) if fn.endswith("png")]
        
        ogle_rrab = ogle_dat[ogle_dat.OGLE_Subtype == "RRab"].index
        ogle_rrc  = ogle_dat[ogle_dat.OGLE_Subtype == "RRc"].index
        rrab_id = [x for x in ogle_rrab if x in rr_lyr]
        rrc_id  = [x for x in ogle_rrc  if x in rr_lyr]
        
        p_rrab = ogle_dat.OGLE_P[rrab_id]
        p_rrc  = ogle_dat.OGLE_P[rrc_id]
        plt.figure(figsize=[8,4])
        plt.hist(p_rrab,bins=15, label="RRab",histtype="step",color="b")
        plt.hist(p_rrc,bins=15, label="RRab",histtype="step",color="r")
        plt.axvline(p_rrab.median(), color='b', linestyle='dashed', linewidth=1,label="median: %.2f"%p_rrab.median())
        plt.axvline(p_rrc.median(),  color='r', linestyle='dashed', linewidth=1,label="median: %.2f"%p_rrc.median())
        plt.xlabel("period [day]")
        plt.ylabel("# stars")
        plt.legend()
        plt.tight_layout()
        plt.savefig("RRLyr_period_hist.png",dpi=300)
        if show:
            plt.show()
        plt.close


    def plot_isochrone(self,x,y,filter1="Jmag",filter2="Ksmag",show=True):
        isoc = np.genfromtxt("isocrone_age_10gyr_z_25e-4.dat",skip_header=11,skip_footer=1,names=True)
        f1 = isoc[filter1] 
        f2 = isoc[filter2]
        mbol = isoc["mbolmag"]
        plt.figure(figsize=[7,7])
        plt.scatter(f1-f2,f2, 
                    marker="s", 
                    edgecolors="b", 
                    facecolors='none', 
                    label="isochrone Z = 0.0025 10 Gyr Av = 0.0")

        isoc = np.genfromtxt("isocrone_age_12gyr_z_5e-4.dat",skip_header=11,skip_footer=1,names=True)
        f1 = isoc[filter1] 
        f2 = isoc[filter2]
        mbol = isoc["mbolmag"]
        plt.scatter(f1-f2,f2, 
                    marker="s", 
                    edgecolors="r", 
                    facecolors='none', 
                    label="isochrone Z = 0.0005 12 Gyr Av = 0.0")
        

        ogle = pd.read_csv("ogle_gaia_dr2_dist.dat", index_col="ID")
        rrab = ogle[ogle.OGLE_Subtype == "RRab"]
        rrc  = ogle[ogle.OGLE_Subtype == "RRc"]

        p_rrab = rrab.OGLE_P
        p_rrc  = rrc.OGLE_P

        metallicity = 0.0025
        mJ  =  self.J_mag(p_rrc,metallicity)
        mKs = self.Ks_mag(p_rrc,metallicity)
        y = mKs
        x = mJ - mKs # rrab.mag_J - rrab.mag_Ks
        plt.scatter(x,y,
                    marker="o", 
                    edgecolors="k", 
                    facecolors='none', 
                    label="RRc [match com OGLE]")

        mJ  =  self.J_mag(p_rrab,metallicity)
        mKs = self.Ks_mag(p_rrab,metallicity)
        y = mKs
        x = mJ - mKs # rrab.mag_J - rrab.mag_Ks
        plt.scatter(x,y,
                    marker="^", 
                    edgecolors="gray", 
                    facecolors='none', 
                    label="RRab [match com OGLE]")
        

        #plt.ylim(-7,5.2)
        plt.gca().invert_yaxis() 
        plt.xlabel("$M_%s - M_%s$"%(filter1[:-3],filter2[:-3]))
        plt.ylabel("$M_%s$"%(filter2[:-3]))
        plt.legend()
        plt.tight_layout()

        plt.savefig("isocrhone.png",dpi=300)
        if show:
            plt.show()
        plt.close()
        

    def isoc_test(self):
        ogle = pd.read_csv("ogle_gaia_dr2_dist.dat", index_col="ID")
        rrab = ogle[ogle.OGLE_Subtype == "RRab"]
        rrc  = ogle[ogle.OGLE_Subtype == "RRc"]

        p_rrab = rrab.OGLE_P
        p_rrc  = rrc.OGLE_P

        metallicity = 0.0015
        mJ  =  self.J_mag(p_rrc,metallicity)
        mKs = self.Ks_mag(p_rrc,metallicity)
        y = mKs
        x = mJ - mKs # rrab.mag_J - rrab.mag_Ks
        self.plot_isochrone(x,y,filter1="Jmag",filter2="Ksmag",show=True)


    def plot_color_diagram(self):
        pass





if __name__ == "__main__":

    # find color ranges using OGLE RRLyr:
    sc = StarClassifier(tile="b293")
    #sc.plot_isochrone(x=None,y=None)

    i = 0
    filters = ["mag_Z","mag_Y","mag_J","mag_H","mag_Ks"] #ordered by lambda
    index = []
    c_limits = []
    while i < 5:
        j = 0
        while j < 4:
            if j < i: 
                f1 = filters[j]
                f2 = filters[i]
                print(f1,f2)
                lim = sc.plot_color(filter1=f1, filter2=f2)
                index.append("%s-%s"%(f1,f2))
                c_limits.append(lim)
            j+=1
        i+=1
    #color_ranges = pd.DataFrame(c_limits,index=index)
    #color_ranges.to_csv("RRLyr_color_ranges.dat",sep=" ")
    

    #get vvv and vsx from vvv:
    '''
    tiles = ["b293","b294","b295","b296"]
    ogle_data = []
    vsx_data  = []
    for tile in tiles:
        sc = StarClassifier(tile="b294")
        data = sc.get_ogle_from_tiles()
        ogle_data.append(data)

        data = sc.get_vsx_from_tiles()
        vsx_data.append(data)
    
    ogle_data = pd.concat(ogle_data, sort=False)
    ogle_data.to_csv("ogle_dat.dat",sep=" ")
    vsx_data  = pd.concat(vsx_data, sort=False)
    vsx_data.to_csv("vsx_dat.dat",sep=" ")
    '''

