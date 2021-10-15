# -*- coding: utf-8 -*-
"""
Created on Apr 13 2020

@author: Everton Botan
@supervisor: Roberto Saito

It's a Python3 program that perform best match on-sky between catalogs. It's an assimetrycal macth.
    it returns:
        idx1: Indices into cat1_coord that matches to the corresponding element of idx2. Shape matches idx2.
        idx2: Indices into cat2_coord that matches to the corresponding element of idx1. Shape matches idx1.
        d2d : The on-sky (angle) separation between the coordinates. Shape matches idx1 and idx2.
"""
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u 
from astropy.coordinates import match_coordinates_sky, search_around_sky

class Match(object):

    def __init__(self):
        '''
        https://docs.astropy.org/en/stable/coordinates/matchsep.html#matching-catalogs
        c1:       catalog - numpy 2D array with RA and DEC.
        ra_u:     unit for RA: "deg" por "hourangle".
        c2:       your table - numpy 2D array with RA and DEC.
        max_sep:  maximium value for the separation
        '''
        self.units = {"deg":u.deg, "hourangle":u.hourangle}
            
    def _units(self,cat,ra_u):
        if not ra_u in  self.units.keys():
            RuntimeWarning("Wrong RA unit. Exiting!")
        else:
            c = SkyCoord(cat.T[0], 
                         cat.T[1],
                         unit=(self.units[ra_u], "deg"))
        return c

    def match(self, c1, ra_u1, c2, ra_u2, max_sep):
        c1 = self._units(c1,ra_u1)
        c2 = self._units(c2,ra_u2)
        idx, d2d, d3d = match_coordinates_sky(c1, c2)
        sep_constraint = d2d < max_sep * u.arcsec
        #if len(idx)>0:
        idx1 = np.argwhere(sep_constraint).T[0]
        idx2 = idx[sep_constraint]
        sep  = d2d[sep_constraint]
        return idx1, idx2, sep.arcsec

    def search_around(self,c1, ra_u1, c2, ra_u2, max_sep):
        idx1, idx2, d2d, d3d = search_around_sky( c1, c2, max_sep * u.arcsec )
        return idx1, idx2, d2d.arcsec

    def separation(self,c1, ra_u1, c2, ra_u2, max_sep):
        '''
        c1: catalog
        c2: individual source
        return:
            idx: index from catalog that match max_sep
            sep: separation
        '''
        c1 = self._units(c1,ra_u1)
        c2 = self._units(c2,ra_u2)
        d2d = c1.separation(c2)
        c1msk = d2d < max_sep*u.arcsec
        idx = np.where(c1msk)[0]
        sep = d2d[c1msk]
        return idx, sep.arcsec


if __name__ == '__main__':
    import pandas as pd
    from astropy.io import ascii
    c2 = ascii.read("data/zyjhk/zyjhk294.cals").to_pandas()
    c22 = c2[["ra","dec"]]  
    c1 = pd.read_csv("ogle_iv_bulge/ogle_iv.csv",sep=",") 
    c11 = c1[["RA","DEC"]]
    bm = Match() 
    idx1, idx2, sep = bm.match(c11.values,"hourangle",c22.values,"deg",1)
    print(idx1, idx2, sep)
