# 
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

import numpy as np
from scipy import interpolate

import astropy.units as u
from astropy.modeling import Fittable1DModel, Parameter

from dust_extinction.helpers import _get_x_in_wavenumbers

def smc_like_model(x, c1, c2, c3, x0, gamma):
    drude = c3 * (x**2 / ((x**2 - x0**2)**2 + (x * gamma)**2))
    return c1 + c2 * x + drude

def bump_model(x, c1, c2, c3, x0, gamma):
    return c1 + c2 * x + (c3 * x**2 / ((x**2 - x0**2)**2 + (x * gamma)**2))

class FM90_Z22Ax(Fittable1DModel):
    r"""
    Fitzpatrick & Massa (1990) 6 parameter ultraviolet shape model

    Parameters
    ----------
    C1: float
       y-intercept of linear term

    C2: float
       slope of liner term

    C3: float
       strength of "2175 A" bump (true amplitude is C3/gamma^2)

    xo: float
       centroid of "2175 A" bump

    gamma: float
       width of "2175 A" bump

    Notes
    -----
    From Fitzpatrick & Massa (1990, ApJS, 72, 163)

    Only applicable at UV wavelengths

    Example showing a FM90 curve with components identified.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.shapes import FM90

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(3.8,8.6,0.1)/u.micron

        ext_model = FM90()
        ax.plot(x,ext_model(x),label='total')

        ext_model = FM90(C3=0.0, C4=0.0)
        ax.plot(x,ext_model(x),label='linear term')

        ext_model = FM90(C1=0.0, C2=0.0, C4=0.0)
        ax.plot(x,ext_model(x),label='bump term')

        ext_model = FM90(C1=0.0, C2=0.0, C3=0.0)
        ax.plot(x,ext_model(x),label='FUV rise term')

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$E(\lambda - V)/E(B - V)$')

        ax.legend(loc='best')
        plt.show()
    """
    n_inputs = 1
    n_outputs = 1

   #  C1 = Parameter(description="linear term: y-intercept", default=-0.18)
   # #  C1 = -0.17933325626620644
   # #  C2 = 0.19155855400351673
   #  C2 = Parameter(description="linear term: slope", default=0.18)
   #  C3 = Parameter(description="bump: amplitude", default=1.0, min=0.2, max=1.0)
   #  xo = Parameter(description="bump: centroid", default=4.6, min=4.6, max=4.6)
   #  gamma = Parameter(description="bump: width", default=0.9, min=0.9, max=1.2)
   #  C4 = Parameter(description="FUV break amplitude", default=0.4, min=0.1, max=0.8)
   #  C5 = Parameter(description="FUV break", default=4.5, min=4.5, max=5.0)
    C1 = Parameter(default=-0.14, min=-1.0, max=10.0, description="linear y-intercept")
    C2 = Parameter(default=0.14, min=-1.0, max=1.0, description="linear slope")
    C3 = Parameter(default=1.0, min=0.0, max=10.0, description="bump amplitude")
    xo = Parameter(default=4.6, description="bump centroid", fixed=True)  # small range around 2175 Å
    gamma = Parameter(default=1.14, min=0.1, max=5.0, description="bump width")
    C4 = Parameter(default=0.5, min=0.0, max=5, description="FUV amplitude")
   #  C5 = Parameter(default=5.0, min=3.0, max=5.5, description="FUV break position") # felo
    C5 = Parameter(default=5, min=1.0, max=7.8, description="FUV break position")


   #  x_range = x_range_FM90

    @staticmethod
    def evaluate(in_x, C1, C2, C3, xo, gamma, C4, C5):
        """
        FM90 function

        Parameters
        ----------
        in_x: float
           expects either x in units of wavelengths or frequency
           or assumes wavelengths in wavenumbers [1/micron]

           internally wavenumbers are used

        Returns
        -------
        Ax: np array (float)
            A(x) extinction curve [mag]

        Raises
        ------
        ValueError
           Input x values outside of defined range
        """
        x = _get_x_in_wavenumbers(in_x)

        # # check that the wavenumbers are within the defined range
        # _test_valid_x_range(x, x_range_FM90, "FM90")

        # linear term
        Ax = C1 + C2 * x

        # bump term
        x2 = x**2
      #   Ax += C3 * (x2 / ((x2 - xo**2) ** 2 + x2 * (gamma**2)))``
        # piecewise FUV break, when x > C5, add C4(x− C5)**2
        Ax += np.where(x > C5, C4 * (x - C5) ** 2, 0)
        return Ax
     
     
class FM90_NoBump(Fittable1DModel):
    r"""
    Fitzpatrick & Massa (1990)-style UV shape *without* the 2175 Å bump.

    This model keeps only:
    - a linear term C1 + C2 * x
    - a far-UV curvature term C4 * F(x)

    with the standard FM90/F99 far-UV function F(x):

        F(x) = 0                                   for x <= 5.9
        F(x) = (x - 5.9)^2 + 0.287 (x - 5.9)^3     for x > 5.9

    so that

        A(x) = C1 + C2 * x + C4 * F(x).

    Parameters
    ----------
    C1 : float
        y-intercept of the linear term.

    C2 : float
        slope of the linear term.

    C4 : float
        amplitude of the far-UV curvature term.  Can be positive
        (extra FUV steepening) or negative (FUV flattening), depending
        on the desired behavior.

    Notes
    -----
    - Only intended for UV wavelengths, typically x ~ 3–8 μm^-1.
    - No 2175 Å bump term is included; this is for bump-less sightlines.
    """

    n_inputs = 1
    n_outputs = 1

    C1 = Parameter(
        default=-0.14,
        min=-5.0,
        max=10.0,
        description="linear y-intercept",
    )
    C2 = Parameter(
        default=0.14,
        min=-5.0,
        max=5.0,
        description="linear slope",
    )
    # allow negative C4 so you can flatten the FUV if needed
    C4 = Parameter(
        default=0.0,
        min=-5.0,
        max=5.0,
        description="far-UV curvature amplitude",
    )

    @staticmethod
    def evaluate(in_x, C1, C2, C4):
        """
        Evaluate the FM90_NoBump function.

        Parameters
        ----------
        in_x : float or ndarray
            Input x values. Can be:
              - wavelength with astropy units, or
              - wavenumber [1/μm] as a plain float/array.
            Internally converted to wavenumber via _get_x_in_wavenumbers.

        Returns
        -------
        Ax : ndarray
            A(x) extinction curve (e.g. E(λ - V)/E(B - V) or similar).
        """
        x = _get_x_in_wavenumbers(in_x)

        # linear term
        Ax = C1 + C2 * x

        # FM90/F99 far-UV curvature term
        x0_fuv = 5.9
        F_x = np.zeros_like(x)
        mask = x > x0_fuv
        if np.any(mask):
            y = x[mask] - x0_fuv
            # standard F(x) from F99: 0.5392 y^2 + 0.05644 y^3
            # here we use the "normalized" form y^2 + 0.287 y^3
            # so that C4 absorbs the overall scale.
            F_x[mask] = y**2 + 0.287 * y**3

        Ax = Ax + C4 * F_x

        return Ax
