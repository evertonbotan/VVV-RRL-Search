# -*- coding: utf-8 -*-
"""
Created on Nov 04 2020

@author: Everton Botan
@supervisor: Roberto Saito

Tools to work with Red Clumps
"""
import os
import sys
import copy
import math
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.signal import find_peaks

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

class RedClump(object):
    
    def __init__(self,Rv):
        self.gc_distance = 8178 # +- 13 pc https://www.aanda.org/articles/aa/full_html/2019/05/aa35656-19/aa35656-19.html
        self.Rv = Rv
        self.path = '/home/botan/OneDrive/Doutorado/VVV_DATA'

    def cartezian_projections(self,d,gal_l,gal_b):
        dx = d*np.cos(math.radians(gal_b))*np.cos(math.radians(gal_l))
        rx = dx - self.gc_distance
        ry = d*np.cos(math.radians(gal_b))*np.sin(math.radians(gal_l))
        rz = d*np.sin(math.radians(gal_b))
        return rx,ry,rz

    def red_clump_distance(self,Ks_mag,Ks_err,c,c_err):
        # Ruiz-Dern et al. (2018) https://ui.adsabs.harvard.edu/abs/2018A%26A...609A.116R/abstract
        c_0      = 0.66
        c_0_err  = 0.02
        MKs      = -1.605
        MKs_err  = 0.009
        # Minniti (2011) AJ Letters 73:L43
        mu       = Ks_mag - self.Rv * (c - c_0) - MKs
        mu_err   = np.sqrt(Ks_err**2 + c_err**2 + c_0_err**2 + MKs_err**2)
        dist     = 10**((mu + 5)/5)
        dist_err = 2**((mu + 5)/5) * 5**(mu/5)*np.log(10) * mu_err
        return dist,dist_err

    def find_RC_color_peak(self, color, color_mask, bins=50, show=False):
        '''
        Find RC peaks color and color sigma and 
        return fit parameters for color peak (gaussian)
        '''
        y               = color[color_mask]
        hist, bin_edges = np.histogram(y,bins=bins)
        flor            = hist > hist[0]

        binSize = bin_edges[1]-bin_edges[0]
        x = np.empty(len(bin_edges)-1)
        x[0] = bin_edges[0] + binSize/2
        i = 1
        while i < len(bin_edges)-1:
            x[i] = x[i-1] + binSize
            i+=1

        guess = [hist.max(),y.median(),0.5]
        fit   = leastsq(func=math_functions.single_gaussian_residuals,
                        x0=guess,
                        args=(x[flor],hist[flor]))
        if show:
            func  = math_functions.single_gaussian(x,fit[0])
            plt.hist(y,bins=bins)
            plt.plot(x,func,'-')
            plt.ylabel('\#\ stars')
            plt.xlabel('J-Ks')
            plt.show()
        return fit[0]

    def find_RC_mag_peak(self, mag, mag_mask, mu1, mu2, bins=100, show=False):
        '''
        find RC peaks in magnitudes
        renturn fit parameters for peaks (two gaussians) and
        Luminosity Function (exponential)
        '''
        hist, bin_edges = np.histogram(mag[mag_mask],bins=bins)
        binSize = bin_edges[1]-bin_edges[0]
        x = np.empty(len(bin_edges)-1)
        x[0] = bin_edges[0] + binSize/2
        i = 1
        while i < len(bin_edges)-1:
            x[i] = x[i-1] + binSize
            i+=1
        # exponential fit to Luminosity Function
        mask2fit = ((x<12.2) | ((x>15.5) & (x<16))) # Mask mag around RC
        guess    = [-1e4,3e3,0.1]
        lum_fit  = leastsq( func = math_functions.exponential_residuals,
                            x0 = guess,
                            args=(x[mask2fit],hist[mask2fit]))
        lum_func = math_functions.exponential(x,lum_fit[0])
        # RC peaks
        RC_peaks   = hist - lum_func
        mask2peaks = ((x>12)&(x<14.5))
        x_RC_peaks = x[mask2peaks]
        y_RC_peaks = RC_peaks[mask2peaks]

        guess = [RC_peaks.max(),mu1,0.5,0.7*RC_peaks.max(),mu2,0.2]
        peak_fit = leastsq( func=math_functions.double_gaussian_residuals,
                            x0=guess,
                            args=(x_RC_peaks,y_RC_peaks))
        if show:
            y = math_functions.double_gaussian(x,peak_fit[0])
            plt.hist(mag[mag_mask],bins=bins)
            plt.plot(x,y+lum_func,'-')
            plt.plot(x,lum_func,'k--')
            plt.ylabel('\#\ stars')
            plt.xlabel('J-Ks')
            plt.show()
        return peak_fit[0],lum_fit[0]

    def find_RC_dist_peak(self, distances, bins, show=False):
        '''
        find RC peaks in disttance
        renturn fit parameters for peaks (two gaussians)
        '''
        hist, bin_edges = np.histogram(distances,bins=bins)
        binSize = bin_edges[1]-bin_edges[0]
        x = np.empty(len(bin_edges)-1)
        x[0] = bin_edges[0] + binSize/2
        i = 1
        while i < len(bin_edges)-1:
            x[i] = x[i-1] + binSize
            i+=1
        # gaussian
        guess = [hist.max(), 8000 ,1000 , 0.5*hist.max(), 11000, 2000]
        peak_fit = leastsq( func=math_functions.double_gaussian_residuals,
                            x0=guess,
                            args=(x,hist))
        if show:
            y1 = math_functions.single_gaussian(x,peak_fit[0][:3])
            y2 = math_functions.single_gaussian(x,peak_fit[0][3:])
            plt.hist(distances,bins=bins)
            plt.plot(x,y1,'k--')
            plt.plot(x,y2,'r--')
            plt.ylabel('\#\ stars')
            plt.xlabel('d [pc]')
            plt.show()
        return peak_fit[0]

    def red_clump_inclination(self,method='2gaussian',plotHist=False):
        '''
        method = '1gaussian'
        method = '2gaussian'
        method = 'polynomial'
        '''
        # params dict [cmin,cmax,ymin,ymax,xmin,xmax]
        params_JKs = {  'b293':[0.85,1.00,11.01,15.49,0.7,2.6],
                        'b294':[0.86,1.00,11.01,15.49,0.7,2.6],
                        'b295':[0.95,1.20,11.01,15.49,0.7,2.6],
                        'b296':[1.05,1.35,11.01,15.49,0.7,2.6],
                        'b307':[1.00,1.40,11.01,15.49,0.7,2.6],
                        'b308':[1.19,1.71,11.01,15.49,0.7,2.6],
                        'b309':[1.19,1.71,11.01,15.49,0.7,2.6],
                        'b310':[1.45,1.80,11.01,15.49,0.7,2.6]}
        params_HKs = {  'b293':[0.19,0.32,11.01,15.49,0.1,0.9],
                        'b294':[0.19,0.32,11.01,15.49,0.1,0.9],
                        'b295':[0.23,0.36,11.01,15.49,0.1,0.9],
                        'b296':[0.29,0.45,11.01,15.49,0.1,0.9],
                        'b307':[0.22,0.45,11.01,15.49,0.1,0.9],
                        'b308':[0.30,0.59,11.01,15.49,0.1,0.9],
                        'b309':[0.32,0.62,11.01,15.49,0.1,0.9],
                        'b310':[0.45,0.70,11.01,15.49,0.1,0.9]}
        params_band = { 'J-Ks':params_JKs,
                        'H-Ks':params_HKs}
        # CMD axes dict
        axes_dict   = { 'b293':[1,3],
                        'b294':[1,2],
                        'b295':[1,1],
                        'b296':[1,0],
                        'b307':[0,3],
                        'b308':[0,2],
                        'b309':[0,1],
                        'b310':[0,0]}

        for color_band in list(params_band.keys()):#[:1]:
            params_dict = params_band[color_band]
            plt.rcParams.update({'font.size': 14})
            fig, axes = plt.subplots(2, 4, figsize=(16,8))
            fig.subplots_adjust(wspace=0.1)
            tiles = sorted(os.listdir(f'{self.path}/data/psf_ts/'))
            for tile in tiles:#['b309','b310','b296']:#tiles:#[:1]:
                tileData = []
                chips = [_[:-3] for _ in os.listdir(f'{self.path}/data/psf_ts/{tile}/chips/') if _.endswith('.ts')]
                for chip in chips:
                    chipData = pd.read_csv(f'{self.path}/data/psf_ts/{tile}/chips/{chip}.ts',index_col='ID')
                    tileData.append(chipData)
            
                tileData = pd.concat(tileData)
                magCols  = [_ for _ in tileData.columns if _[:3] == 'MAG']
                errCols  = [_ for _ in tileData.columns if _[:3] == 'ERR']
                err_msk  = ( tileData[errCols] > 0.2).values
                f = color_band.split('-')[0]
                color = tileData[f'mag_{f}'] - tileData.mag_Ks
                msk   = ~color.isnull()
                mag   = tileData.mag_Ks
                mag   = mag[msk]
                color = color[msk]


                yRCpeak = []
                xRCpeak = []
                if method == '1gaussian':
                    # Single Gaussian fit
                    num_bins = 20
                    cmin = params_dict[tile][0]
                    cmax = params_dict[tile][1]
                    n = cmin
                    while n < cmax:
                        dc   = abs(cmax-cmin)/10
                        cmsk = ((color > n) & (color <= n+dc) & (mag < 14))
                        hist, bin_edges = np.histogram(mag[cmsk],bins=num_bins)
                        binSize = bin_edges[1]-bin_edges[0]
                        x = [bin_edges[0] + binSize/2]
                        i = 1
                        while i < len(bin_edges)-1:
                            x.append(x[i-1] + binSize)
                            i+=1
                                    
                        guess = [500,13.2,0.5,]
                        fit = leastsq(math_functions.single_gaussian_residuals,guess,args=(x,hist))
                        params = fit[0]
                        yfit = math_functions.single_gaussian(x,params)
                        if plotHist:
                            fig,ax=plt.subplots()
                            ax.hist(mag[cmsk],num_bins)
                            ax.plot(x,yfit,'-')
                            plt.show()
                        yRCpeak.append(params[1])
                        xRCpeak.append(n)
                        n+=dc

                if method == '2gaussian':
                    # DOuble Gaussian
                    num_bins = 80
                    cmin = params_dict[tile][0]
                    cmax = params_dict[tile][1]
                    n = cmin
                    while n < cmax:
                        dc   = 0.05 #abs(cmax-cmin)/10
                        cmsk = ((color > n) & (color <= n+dc))# & (mag < 17.5))
                        
                        hist, bin_edges = np.histogram(mag[cmsk],bins=num_bins)
                        binSize = bin_edges[1]-bin_edges[0]
                        x = [bin_edges[0] + binSize/2]
                        i = 1
                        while i < len(bin_edges)-1:
                            x.append(x[i-1] + binSize)
                            i+=1

                        mu1 = 13.0 #params_dict[tile][6] # initial guess for fisrt peak mag
                        mu2 = 13.6 #params_dict[tile][7] # initial guess for second peak mag
                        peak_fit, lum_fit = self.find_RC_mag_peak(mag, cmsk, mu1, mu2, show=False)
                        #peak_fit, lum_fit = find_RC_mag_peak(1,mag, cmsk, mu1, mu2, bins=num_bins, show=False)
                        
                        x = np.arange(11,18,(18-12)/1000)
                        lum_func = math_functions.exponential(x,lum_fit)
                        RC_fit = math_functions.double_gaussian(x,peak_fit)
                        fitted_curve = RC_fit + lum_func
                        crop = x < 14.5
                        mag_peak = x[crop][np.where(fitted_curve[crop] == fitted_curve[crop].max())[0][0]]


                        if plotHist:
                            yaxis_ref = np.histogram(mag[cmsk],bins=num_bins)[0].max()
                            fig,ax=plt.subplots(figsize=[6,4])
                            ax.hist(x=mag[cmsk],
                                    bins=num_bins,
                                    histtype='barstacked',
                                    lw=0.5,
                                    color='dodgerblue',
                                    edgecolor='w',
                                    alpha=0.6)
                            ax.plot(x,RC_fit+lum_func,'r-',lw=1)
                            ax.plot(x,lum_func,'k--',lw=1)
                            ptxt = '{:#.3n}'.format(mag_peak)
                            ax.axvline(mag_peak,lw=0.8,c='gray')
                            ax.text(s=ptxt,x=mag_peak+0.2,y=0.95*yaxis_ref,ha='left')
                            title = '{:#.3n}'.format(n) + ' < J-Ks < ' + '{:#.3n}'.format(n+dc)
                            ax.text(s=f'Tile: {tile} | {title}', x=0.5, y=1.02, ha='center', transform=ax.transAxes)
                            ax.set_ylabel('Número de estrelas')
                            ax.set_xlabel('Ks [mag]')
                            ax.set_ylim(-yaxis_ref*0.01,yaxis_ref+yaxis_ref*0.04)
                            plt.tight_layout()
                            plt.savefig(f'{self.path}/figuras_tese/RC_peaks_{tile}_{n}.png',dpi=300)
                            plt.show()
                            plt.close()
                        yRCpeak.append(mag_peak)
                        xRCpeak.append(n)
                        n+=dc

                if method == 'polynomial':
                    # Polynomial fit
                    num_bins = 100
                    cmin = params_dict[tile][0]
                    cmax = params_dict[tile][1]
                    n = cmin
                    while n < cmax:
                        dc   = (cmax-cmin)/8
                        cmsk = ((color > n) & (color <= n+dc) & (mag < 17.5))
                        
                        hist, bin_edges = np.histogram(mag[cmsk],bins=num_bins)
                        binSize = bin_edges[1]-bin_edges[0]
                        x = [bin_edges[0] + binSize/2]
                        i = 1
                        while i < len(bin_edges)-1:
                            x.append(x[i-1] + binSize)
                            i+=1
                        x = np.array(x)
                        fit = np.polyfit(x, hist, 200)
                        yp = np.poly1d(fit)
                    
                        x2 = np.arange(mag[cmsk].min(),mag[cmsk].max(),(mag[cmsk].max() - mag[cmsk].min())/1000)
                        msk = ((x2>12.5)&(x2<14))
                        peaks,_ = find_peaks(yp(x2[msk]))
                        if plotHist:
                            fig,ax=plt.subplots()
                            ax.hist(mag[cmsk],num_bins)
                            ax.plot(x,yp(x),'-')
                            ax.plot(x2[msk][peaks],yp(x2[msk][peaks]),"*")
                            ax.plot(x2[msk][peaks[0]],yp(x2[msk][peaks[0]]),"*")
                            plt.show()
                        yRCpeak.append(x2[msk][peaks[0]])
                        xRCpeak.append(n)
                        n+=dc

                # CMD plot
                
                y = np.array(yRCpeak)
                x = np.array(xRCpeak)
                xlim= params_dict[tile][4:6]
                ylim= params_dict[tile][2:4]
                xlabel= color_band
                ylabel='Ks [mag]'

                guess = [0.6,13]
                c,cov = curve_fit(  f = math_functions.linear,
                                    xdata = x,
                                    ydata = y,
                                    p0 = guess,
                                    sigma = y*.01,
                                    absolute_sigma = False) 
                
                xfit = np.array(xlim)
                yfit = math_functions.linear(xfit,c[0],c[1])

                bins=(600,400)
                cmap = copy.copy(mpl.cm.get_cmap("jet"))# plt.cm.jet 
                cmap.set_bad('w', 1.)
                cmap_multicolor = copy.copy(mpl.cm.get_cmap("jet")) # plt.cm.jet
                cmap_multicolor.set_bad('w', 1.)
                clip = ~color.isnull()
                N, xedges, yedges = np.histogram2d(color[clip],mag[clip],bins=bins)
                ax1 = axes_dict[tile][0]
                ax2 = axes_dict[tile][1]
                img = axes[ax1,ax2].imshow(np.log10(N.T), origin='lower',
                                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                            aspect='auto', interpolation='nearest', cmap=cmap)              
                red_inc = '{:#.3n}'.format(c[0])
                if ax1==0:
                    if ax2==1 or ax2==2:
                        axes[ax1,ax2].plot(x,y,'k.')
                        axes[ax1,ax2].plot(xfit,yfit,'k-')
                        axes[ax1,ax2].text( s=f'Tile: {tile} | AKs/E({color_band}) = {red_inc}',
                                        x=0.5,
                                        y=1.025,
                                        ha='center',
                                        transform=axes[ax1,ax2].transAxes)
                    else:
                        axes[ax1,ax2].text( s=f'Tile: {tile}',
                                        x=0.025,
                                        y=1.025,
                                        ha='left',
                                        transform=axes[ax1,ax2].transAxes)

                else:
                    axes[ax1,ax2].text( s=f'Tile: {tile}',
                                        x=0.025,
                                        y=1.025,
                                        ha='left',
                                        transform=axes[ax1,ax2].transAxes)
                axes[ax1,ax2].set_xlim(xlim)
                axes[ax1,ax2].set_ylim(ylim)
                axes[ax1,ax2].set_xlabel(xlabel)
                axes[ax1,ax2].set_ylabel(ylabel)
                axes[ax1,ax2].invert_yaxis()
                for im in plt.gca().get_images():
                    im.set_clim(0, 3)
            for ax in fig.get_axes():
                ax.label_outer()
            cbar_ax = plt.axes([0.92, 0.2, 0.01, 0.6])
            cb = fig.colorbar(img, 
                            ticks=[0, 1, 2, 3],
                            format=r'$10^{%i}$',
                            shrink=0.6 ,
                            cax=cbar_ax)
            cb.set_label('Número por pixel',rotation=90)
            #cb.set_label(r'$\mathrm{number\ in\ pixel}$',rotation=90)
            #plt.tight_layout()
            plt.savefig(f'{self.path}/figuras_tese/red_clump_reddening_{color_band}.png',dpi=200)
            plt.show()
            plt.rcParams.update({'font.size': 12})
            plt.close()

    def find_RC_peaks(self,plot=False,show=False):
        # params dict [ymin,ymax,xmin,xmaxc,cmin,cmax,RC_peak1,RC_peak2]
        params_dict = { 'b293':[11,17.9,0.0,1.4,0.65,1.10,13.0,13.8],
                        'b294':[11,17.9,0.0,1.5,0.70,1.20,13.0,13.8],
                        'b295':[11,17.9,0.2,1.5,0.75,1.30,13.0,13.9],
                        'b296':[11,17.9,0.2,1.7,0.85,1.64,13.0,14.1],
                        'b307':[11,17.9,0.1,2.0,0.85,1.50,13.1,13.8],
                        'b308':[11,17.9,0.1,2.3,1.00,1.60,13.2,14.0],
                        'b309':[11,17.9,0.1,2.3,1.00,2.00,13.2,14.2],
                        'b310':[11,17.9,0.3,2.6,1.20,2.00,13.2,14.3]}

        tiles = sorted(os.listdir(f'{self.path}/data/psf_ts/'))
        cols = ['RC_peak1_Ks_mag','RC_peak1_Ks_sigma',
                'RC_peak1_color' ,'RC_peak1_color_sigma',
                'RC_peak1_dist'  ,'RC_peak1_dist_sigma',
                'RC_peak2_Ks_mag','RC_peak2_Ks_sigma',
                'RC_peak2_color' ,'RC_peak2_color_sigma',
                'RC_peak2_dist'  ,'RC_peak2_dist_sigma',
                'tile_central_l' ,'tile_central_b']
        RC_info = pd.DataFrame(index=tiles,columns=cols)
        
        for tile in tiles:#[:1]:
            tileData = []
            chips = [_[:-3] for _ in os.listdir(f'{self.path}/data/psf_ts/{tile}/chips/') if _.endswith('.ts')]
            for chip in chips:
                chipData = pd.read_csv(f'{self.path}/data/psf_ts/{tile}/chips/{chip}.ts',index_col='ID')
                tileData.append(chipData)
            tileData = pd.concat(tileData)

            ra     = tileData.RA
            dec    = tileData.DEC
            c_icrs = SkyCoord(ra=ra, dec=dec,unit=(u.deg, u.deg))
            c_gal  = c_icrs.galactic
            tileData.loc[tileData.index,'gal_l'] = c_gal.l.deg
            tileData.loc[tileData.index,'gal_b'] = c_gal.b.deg
        
            color = tileData.mag_J - tileData.mag_Ks
            msk = ~color.isnull()
            color = color[msk]
            mag = tileData.mag_Ks[msk]
            color_min = params_dict[tile][4]

            # get RC peaks magnitudes
            mag_mask = ((color > color_min))
            mu1 = params_dict[tile][6] # initial guess for fisrt peak mag
            mu2 = params_dict[tile][7] # initial guess for second peak mag
            peak_fit, lum_fit = self.find_RC_mag_peak(mag, mag_mask, mu1, mu2, show=False)
            
            # get RC peaks colors
            color_masks = []
            peak_colors = []
            i = 1
            while i < 6:
                peak_mag, peak_sigma = peak_fit[i], peak_fit[i+1]
                # RC peaks color and color sigma           
                color_mask = (((color > color_min) & (color < 2.6)) & ((mag > peak_mag - abs(peak_sigma)) & (mag < peak_mag + abs(peak_sigma))))
                color_fit  = self.find_RC_color_peak(color, color_mask, show=False)
                peak_colors += [color_fit[1], abs(color_fit[2])]
                color_masks.append(color_mask)
                i+=3

            # calculate distances
            dist1,dist1_sigma = self.red_clump_distance(peak_fit[1],peak_fit[2],peak_colors[0],abs(peak_colors[1]))
            dist2,dist2_sigma = self.red_clump_distance(peak_fit[4],peak_fit[5],peak_colors[2],abs(peak_colors[3]))

            # tile central l and b
            tile_l = (tileData.gal_l.max() - tileData.gal_l.min())/2 + tileData.gal_l.min()
            tile_b = (tileData.gal_b.max() - tileData.gal_b.min())/2 + tileData.gal_b.min()
        
            # save peaks info into a pandas DataFrame
            info = list(peak_fit[1:3]) + peak_colors[:2] + [dist1,dist1_sigma] + list(peak_fit[4:6]) + peak_colors[2:] + [dist2,dist2_sigma,tile_l,tile_b]
            RC_info.loc[tile,cols] = info

            if plot:
                # Plot CMD 
                xlim  = params_dict[tile][2:4]
                ylim  = params_dict[tile][:2]
                xlabel='J-Ks'
                ylabel='Ks [mag]'

                bins=(600,400)
                cmap = copy.copy(mpl.cm.get_cmap("jet"))# plt.cm.jet 
                cmap.set_bad('w', 1.)
                cmap_multicolor = copy.copy(mpl.cm.get_cmap("jet")) # plt.cm.jet
                cmap_multicolor.set_bad('w', 1.)
                clip = ~color.isnull()
                N, xedges, yedges = np.histogram2d(color[clip],mag[clip],bins=bins)
                fig, axes = plt.subplots(1, 2, figsize=(10,4))
                img = axes[0].imshow(   np.log10(N.T),
                                        origin='lower',
                                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                        aspect='auto',
                                        interpolation='nearest',
                                        cmap=cmap)
                axes[0].errorbar(   x=info[2],
                                    y=info[0],
                                    xerr=info[3],
                                    yerr=info[1],
                                    marker="o",
                                    mfc='k',
                                    mec='k',
                                    ecolor='k',
                                    ms=3,
                                    lw=.8,
                                    capsize=3)
                axes[0].errorbar(   x=info[6],
                                    y=info[4],
                                xerr=info[7],
                                    yerr=info[5],
                                    marker="o",
                                    mfc='k',
                                    mec='k',
                                    ecolor='k',
                                    ms=3,
                                    lw=.8,
                                    capsize=3)
                axes[0].set_xlim(xlim)
                axes[0].set_ylim(ylim)
                axes[0].set_xlabel(xlabel)
                axes[0].set_ylabel(ylabel)
                axes[0].invert_yaxis()
                axes[0].axvline(color_min,c='k',lw=1)
                axes[0].text(s=f'Tile: {tile}',x=0.5,y=1.02,ha='center',transform=axes[0].transAxes)
                cb = fig.colorbar(  img, 
                                    ax=axes[0],
                                    ticks=[0, 1, 2, 3],
                                    format=r'$10^{%i}$',
                                    shrink=0.6,
                                    orientation='vertical')
                cb.set_label(r'$\mathrm{Número\ por\ pixel}$',rotation=90)

                # to plot luminosity ans peaks functions
                x = np.arange(11,18,(18-12)/1000)
                lum_func = math_functions.exponential(x,lum_fit)
                RC_fit = math_functions.double_gaussian(x,peak_fit)
                    
                # mask test:
                #axes[0].plot(color[color_masks[0]],mag[color_masks[0]],'b.',ms=.8,alpha=.01)
                #axes[0].plot(color[color_masks[1]],mag[color_masks[1]],'b.',ms=.8,alpha=.01)
                yaxis_ref = np.histogram(mag[mag_mask],bins=100)[0].max() # reference value
                axes[1].hist(   x=mag[mag_mask],
                                bins=100,
                                histtype='barstacked',
                                lw=.5,
                                color='dodgerblue',
                                edgecolor='w',
                                alpha=0.6)#,range=range)
                axes[1].plot(x,RC_fit+lum_func,'r-',lw=1)
                axes[1].plot(x,lum_func,'k--',lw=1)
                axes[1].axvline(x=peak_fit[1],
                                ls='--',
                                c='gray',
                                lw=1)
                m1 = '{:#.4n}'.format(peak_fit[1])
                axes[1].text(   s=f'{m1}',
                                x=peak_fit[1],
                                y=.9*yaxis_ref)
                axes[1].axvline(x=peak_fit[4],
                                ls='--',
                                c='gray',
                                lw=1)
                m2 = '{:#.4n}'.format(peak_fit[4])
                axes[1].text(   s=f'{m2}',
                                x=peak_fit[4],
                                y=.8*yaxis_ref)
                axes[1].set_xlabel(ylabel)
                axes[1].set_ylabel('Número de estrelas')
                a = '{:#.2n}'.format(color_min)
                axes[1].text(s=f'J-Ks > {a}',x=0.5,y=1.02,ha='center',transform=axes[1].transAxes)
                axes[1].yaxis.set_label_position("right")
                axes[1].yaxis.tick_right()
                axes[1].set_ylim(-yaxis_ref*.01,yaxis_ref+yaxis_ref*.04)
                plt.tight_layout()
                plt.savefig(f'{self.path}/figuras_tese/{tile}_RC_bumps.png',dpi=200)
                if show:
                    plt.show()
                plt.close()
        return RC_info



    ''' ======================= WORK IN PROGRESS ========================'''


    def RC_peak_distance_distribution(self,plot=False,show=False):
        path = '/home/botan/OneDrive/Doutorado/VVV_DATA'
        params_dict = { 'b293':[11,17.9,0.0,1.4,0.65,1.10,13.0,13.8],
                        'b294':[11,17.9,0.0,1.5,0.70,1.20,13.0,13.8],
                        'b295':[11,17.9,0.2,1.5,0.75,1.30,13.0,13.9],
                        'b296':[11,17.9,0.2,1.7,0.85,1.64,13.0,14.1],
                        'b307':[11,17.9,0.1,2.0,0.85,1.50,13.1,13.8],
                        'b308':[11,17.9,0.1,2.3,1.00,1.60,13.2,14.0],
                        'b309':[11,17.9,0.1,2.3,1.00,2.00,13.2,14.2],
                        'b310':[11,17.9,0.3,2.6,1.20,2.00,13.2,14.3]}

        tiles = sorted(os.listdir(f'{path}/data/psf_ts/'))
        cols = ['mag_peak1','mag_err_peak1',
                'color_peark1','color_err_peark1',
                'distance1','distance_err1',
                'x1','y1','z1',
                'mag_peak2','err_peak2',
                'color_peark2','color_err_peark2',
                'distance2','distance_err2',
                'x2','y2','z2',
                'tile_l','tile_b']
        RC_info = pd.DataFrame(index=tiles,columns=cols)
        
        for tile in tiles:
            tileData = []
            chips = [_[:-3] for _ in os.listdir(f'{path}/data/psf_ts/{tile}/chips/') if _.endswith('.ts')]
            for chip in chips:
                chipData = pd.read_csv(f'{path}/data/psf_ts/{tile}/chips/{chip}.ts',index_col='ID')
                tileData.append(chipData)
            tileData = pd.concat(tileData)

            ra     = tileData.RA
            dec    = tileData.DEC
            c_icrs = SkyCoord(ra=ra, dec=dec,unit=(u.deg, u.deg))
            c_gal  = c_icrs.galactic
            tileData.loc[tileData.index,'gal_l'] = c_gal.l.deg
            tileData.loc[tileData.index,'gal_b'] = c_gal.b.deg

            color = tileData.mag_J - tileData.mag_Ks
            color_err = np.sqrt((tileData.er_J)**2 + (tileData.er_Ks)**2)
            msk = ~color.isnull()
            color = color[msk]
            color_err = color_err[msk]
            mag = tileData.mag_Ks[msk]
            err = tileData.er_Ks[msk]
            color_min = params_dict[tile][4]
            mag_max = 14.5
            #CMD_crop = ((color > color_min) & (mag < mag_max))

            # get RC peaks magnitudes
            mag_mask = ((color > color_min))
            mu1 = params_dict[tile][6] # initial guess for fisrt peak mag
            mu2 = params_dict[tile][7] # initial guess for second peak mag
            peak_fit, lum_fit = self.find_RC_mag_peak(mag, mag_mask, mu1, mu2, show=False)
            
            peak_sigma = 2
            peaks_lim = [peak_fit[1] - peak_sigma*peak_fit[2], peak_fit[4] + peak_sigma*peak_fit[5]]
            CMD_crop = ((color > color_min) & ((mag > peaks_lim[0])&(mag < peaks_lim[1])))

            # get RC peaks colors
            color_masks = []
            peak_colors = []
            i = 1
            while i < 6:
                peak_mag, peak_sigma = peak_fit[i], peak_fit[i+1]
                # RC peaks color and color sigma           
                color_mask = (((color > color_min) & (color < 2.6)) & ((mag > peak_mag - abs(peak_sigma)) & (mag < peak_mag + abs(peak_sigma))))
                color_fit  = self.find_RC_color_peak(color, color_mask, show=False)
                peak_colors += list(abs(color_fit[1:]))
                color_masks.append(color_mask)
                i+=3

            # get peaks distances
            Rv = 0.689
            binsize = 50
            dist, dist_sigma = self.red_clump_distance(mag,err,Rv,color,color_err)
            #gc_dist  = self.gc_distance
            dist_peaks = self.find_RC_dist_peak(distances=dist[CMD_crop],bins=binsize)
            

            # distance using peak in mag
            dist2,dist_sigma2 = self.red_clump_distance(peak_fit[1],peak_fit[2],Rv,peak_colors[0],abs(peak_colors[1]))
            dist3,dist_sigma3 = self.red_clump_distance(peak_fit[4],peak_fit[5],Rv,peak_colors[2],abs(peak_colors[3]))
            
            tile_l = (tileData.gal_l.max() - tileData.gal_l.min())/2
            tile_b = (tileData.gal_b.max() - tileData.gal_b.min())/2

            cartesian2 = self.cartezian_projections(dist2,tile_l,tile_b,self.gc_distance)
            cartesian3 = self.cartezian_projections(dist3,tile_l,tile_b,self.gc_distance)

            params = (list(peak_fit[1:3]) + peak_colors[:2] + [dist2,dist_sigma2] + list(cartesian2)
                    + list(peak_fit[4:]) + peak_colors[2:] + [dist3,dist_sigma3] + list(cartesian3) +
                    [tile_l, tile_b])
            RC_info.loc[tile,cols] = params

            if plot:
                # PLOT CMD ans HIST
                xlim  = params_dict[tile][2:4]
                ylim  = params_dict[tile][:2]
                xlabel='J-Ks'
                ylabel='Ks [mag]'

                bins=(600,400)
                cmap = copy.copy(mpl.cm.get_cmap("jet"))# plt.cm.jet 
                cmap.set_bad('w', 1.)
                cmap_multicolor = copy.copy(mpl.cm.get_cmap("jet")) # plt.cm.jet
                cmap_multicolor.set_bad('w', 1.)
                N, xedges, yedges = np.histogram2d(color,mag,bins=bins)

                fig, axes = plt.subplots(1, 2, figsize=(10,4))
                img = axes[0].imshow(   np.log10(N.T),
                                        origin='lower',
                                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                        aspect='auto',
                                        interpolation='nearest',
                                        cmap=cmap)
                axes[0].add_patch(Rectangle(xy=(color_min,mag_max),
                                            width=xlim[1],
                                            height=peaks_lim[0]-peaks_lim[1],
                                            fc ='red', 
                                            ec ='none',
                                            lw = 0,
                                            alpha=0.3) )
                axes[0].set_xlim(xlim)
                axes[0].set_ylim(ylim)
                axes[0].set_xlabel(xlabel)
                axes[0].set_ylabel(ylabel)
                axes[0].invert_yaxis()
                axes[0].text(s=f'Tile: {tile}',x=0.5,y=1.02,ha='center',transform=axes[0].transAxes)
                cb = fig.colorbar(  img, 
                                    ax=axes[0],
                                    ticks=[0, 1, 2, 3],
                                    format=r'$10^{%i}$',
                                    shrink=0.6,
                                    orientation='vertical')
                cb.set_label(r'$\mathrm{Número\ por\ pixel}$',rotation=90)

                x = np.arange(dist[CMD_crop].min(),dist[CMD_crop].max(),(dist[CMD_crop].max() - dist[CMD_crop].min())/1000)
                y1 = math_functions.single_gaussian(x,dist_peaks[:3])
                y2 = math_functions.single_gaussian(x,dist_peaks[3:])

                axes[1].hist(   x=dist[CMD_crop]/1000,
                                bins=binsize,
                                histtype='barstacked',
                                lw=.5,
                                color='dodgerblue',
                                edgecolor='w',
                                alpha=0.6)#,range=range)
                
                axes[1].plot(x/1000,y1,'r--',lw=.8)
                axes[1].plot(x/1000,y2,'r--',lw=.8)
                axes[1].plot(x/1000,y1+y2,'k-',lw=.8)

                axes[1].axvline(dist2/1000,ls='--',lw=.8,c='gray')
                axes[1].axvline(dist3/1000,ls='--',lw=.8,c='gray')
                axes[1].set_xlabel('d [kpc]')
                axes[1].set_ylabel('Número de estrelas')
                axes[1].yaxis.set_label_position("right")
                axes[1].yaxis.tick_right()
                plt.tight_layout()
                plt.savefig(f'{path}/figuras_tese/{tile}_RC_dist_hist.png',dpi=200)
                if show:
                    plt.show()
                plt.close()
        return RC_info



    def cmd(self):
        #params dict [cmin,cmax,ymin,ymax,xmin,xmax]
        params_dict = { 'b293':[0.85,1.00,11,17.9,0.7,2.6],
                        'b294':[0.86,1.00,11,17.9,0.7,2.6],
                        'b295':[0.95,1.20,11,17.9,0.7,2.6],
                        'b296':[1.05,1.40,11,17.9,0.7,2.6],
                        'b307':[1.00,1.40,11,17.9,0.7,2.6],
                        'b308':[1.19,1.71,11,17.9,0.7,2.6],
                        'b309':[1.19,1.71,11,17.9,0.7,2.6],
                        'b310':[1.45,2.00,11,17.9,0.7,2.6]}
        params_HKs = { 'b293':[0.17,0.29,11,17.9,0.01,1.0],
                        'b294':[0.19,0.32,11,17.9,0.01,1.0],
                        'b295':[0.23,0.36,11,17.9,0.01,1.0],
                        'b296':[0.27,0.45,11,17.9,0.01,1.0],
                        'b307':[0.18,0.39,11,17.9,0.01,1.0],
                        'b308':[0.28,0.59,11,17.9,0.01,1.0],
                        'b309':[0.28,0.62,11,17.9,0.01,1.0],
                        'b310':[0.42,0.70,11,17.9,0.01,1.0]}
        # CMD axes dict
        axes_dict   = { 'b293':[1,3],
                        'b294':[1,2],
                        'b295':[1,1],
                        'b296':[1,0],
                        'b307':[0,3],
                        'b308':[0,2],
                        'b309':[0,1],
                        'b310':[0,0]}
        filters = ['mag_H','mag_J']
        for band in filters:

            fig, axes = plt.subplots(2, 4, figsize=(16,8))
            tiles = sorted(os.listdir('/home/botan/OneDrive/Doutorado/VVV_DATA/data/psf_ts/'))
            for tile in tiles:#[:1]:
                tileData = []
                chips = [_[:-3] for _ in os.listdir(f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/psf_ts/{tile}/chips/') if _.endswith('.ts')]
                for chip in chips:
                    chipData = pd.read_csv(f'/home/botan/OneDrive/Doutorado/VVV_DATA/data/psf_ts/{tile}/chips/{chip}.ts',index_col='ID')
                    tileData.append(chipData)
            
                tileData = pd.concat(tileData)

                color = tileData[band] - tileData.mag_Ks
                msk   = ~color.isnull()
                mag   = tileData.mag_Ks
                mag   = mag[msk]
                color = color[msk]


                xlim= params_dict[tile][4:6]
                ylim= params_dict[tile][2:4]
                xlabel=f'{band[-1]}-Ks'
                ylabel='Ks [mag]'

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

                axes[ax1,ax2].text(s=f'Tile: {tile}',x=(xlim[0]+xlim[1])/2,y=10.9,ha='center')
                #axes[ax1,ax2].set_xlim(xlim)
                axes[ax1,ax2].set_ylim(ylim)
                axes[ax1,ax2].set_xlabel(xlabel)
                axes[ax1,ax2].set_ylabel(ylabel)
                axes[ax1,ax2].invert_yaxis()
                for im in plt.gca().get_images():
                    im.set_clim(0, 3)
            for ax in fig.get_axes():
                ax.label_outer()
            cbar_ax = plt.axes([0.92, 0.2, 0.01, 0.6])
            cb = fig.colorbar(img, 
                            ticks=[0, 1, 2, 3],
                            format=r'$10^{%i}$',
                            shrink=0.6 ,
                            cax=cbar_ax)
            cb.set_label(r'$\mathrm{número\ por\ pixel}$',rotation=90)
            #cb.set_label(r'$\mathrm{number\ in\ pixel}$',rotation=90)
            #plt.tight_layout()
            plt.savefig(f'cmd_{band[-1]}-Ks.png',dpi=200)
            plt.show()




if __name__ == "__main__":
    import importlib
    importlib.reload(sys.modules['math_functions'])
    
    
    d = RedClump(Rv=0.689)
    # red clump distance, color and magnitude peaks.
    RC_info = d.find_RC_peaks(show=False)

    # Red Clump distance histograms
    RC_info = d.RC_peak_distance_distribution(plot=False,show=False)
