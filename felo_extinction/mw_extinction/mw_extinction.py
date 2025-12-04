# Copyright (C) 2024  Yuming Fu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from dustmaps.sfd import SFDQuery
from dustmaps.csfd import CSFDQuery
from dustmaps.pg2010 import PG2010Query
# from dust_extinction.parameter_averages import F99, G23
from dust_extinction import parameter_averages
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

class MilkyWayExtinction:
    def __init__(self, extinction_model='G23', Rv=3.1):
        if extinction_model not in parameter_averages.__all__:
            raise ValueError('Invalid extinction model')
        self.ext = getattr(parameter_averages, extinction_model)(Rv=Rv)
        self.csfd = CSFDQuery()
        self.pg10 = PG2010Query()
        
    def get_ebv(self, ra=None, dec=None, coord=None):
        if (ra is None or dec is None) and coord is None:
            raise ValueError("At least one of ra, dec, or coord must be provided.")
        if coord is not None:
            coords = coord
        else:
            coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        ebv_csfd = self.csfd(coords)
        corr_pg10 = self.pg10(coords)
        ebv = ebv_csfd - corr_pg10
        self.ebv = ebv
        return ebv
        
    def correct_extinction(self, wave, flux, flux_err=None, ebv=None):
        if ebv is None:
            if not hasattr(self, 'ebv'):
                raise ValueError("EBV not calculated. Run get_ebv first.")
            ebv = self.ebv
        if not hasattr(wave, 'unit'):
            print("Assuming wavelength is in Angstroms")
            wave = wave * u.angstrom
        flux_corr = flux / self.ext.extinguish(wave, ebv)
        if flux_err is not None:
            flux_err_corr = flux_err * flux_corr / flux
        else:
            flux_err_corr = None
        return flux_corr, flux_err_corr