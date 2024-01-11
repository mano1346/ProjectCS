# From https://github.com/poliastro/poliastro/blob/main/contrib/satgpio.py
"""
Author: Juan Luis Cano Rodríguez

Code to read GP data from Celestrak using the HTTP API and python-sgp4.

Requires some extra dependencies:

  $ pip install httpx sgp4

This is similar to https://gitlab.com/librespacefoundation/python-satellitetle,
but uses the XML API instead and returns a `Satrec` object from sgp4 directly.

"""

import io
import json
import xml.etree.ElementTree as ET

import httpx
from sgp4 import exporter, omm
from sgp4.api import Satrec


def _generate_url(catalog_number, international_designator, name):
    params = {
        "CATNR": catalog_number,
        "INTDES": international_designator,
        "NAME": name,
    }
    param_names = [
        param_name
        for param_name, param_value in params.items()
        if param_value is not None
    ]
    if len(param_names) != 1:
        raise ValueError(
            "Specify exactly one of catalog_number, international_designator, or name"
        )
    param_name = param_names[0]
    param_value = params[param_name]
    url = (
        "https://celestrak.org/NORAD/elements/gp.php?"
        f"{param_name}={param_value}"
        "&FORMAT=XML"
    )
    return url


def _segments_from_query(url):
    response = httpx.get(url)
    response.raise_for_status()

    if response.text == "No GP data found":
        raise ValueError(
            f"Query '{url}' did not return any results, try a different one"
        )
    tree = ET.parse(io.StringIO(response.text))
    root = tree.getroot()

    yield from omm.parse_xml(io.StringIO(response.text))


def load_gp_from_celestrak(
    *, catalog_number=None, international_designator=None, name=None
):
    """Load general perturbations orbital data from Celestrak.

    Returns
    -------
    Satrec
        Orbital data from specified object.

    Notes
    -----
    This uses the OMM XML format from Celestrak as described in [1]_.

    References
    ----------
    .. [1] Kelso, T.S. "A New Way to Obtain GP Data (aka TLEs)"
       https://celestrak.org/NORAD/documentation/gp-data-formats.php

    """
    # Assemble query, raise an error if malformed
    url = _generate_url(catalog_number, international_designator, name)

    # Make API call, raise an error if data is malformed
    for segment in _segments_from_query(url):
        # Initialize and return Satrec object
        sat = Satrec()
        omm.initialize(sat, segment)

        yield sat


def print_sat(sat, name):
    """Prints Satrec object in convenient form."""
    print(json.dumps(exporter.export_omm(sat, name), indent=2))


sat = list(load_gp_from_celestrak(name="STARLINK"))[500]
print_sat(sat, "STARLINK")

from astropy import units as u
from astropy.time import Time

now = Time.now()
now.jd1, now.jd2

error, r, v = sat.sgp4(now.jd1, now.jd2)
assert error == 0

import numpy as np

from astropy.coordinates import CartesianRepresentation, CartesianDifferential

from poliastro.util import time_range

times = time_range(now, end=now + (1 << u.h), periods=3)


errors, rs, vs = sat.sgp4_array(times.jd1, times.jd2)
assert (errors == 0).all()

# print(CartesianRepresentation(rs << u.km, xyz_axis=-1))

# print(CartesianDifferential(vs << (u.km / u.s), xyz_axis=-1))


from warnings import warn

from astropy.coordinates import TEME, GCRS

from poliastro.ephem import Ephem
from poliastro.frames import Planes


def ephem_from_gp(sat, times):
    errors, rs, vs = sat.sgp4_array(times.jd1, times.jd2)
    if not (errors == 0).all():
        warn(
            "Some objects could not be propagated, " "proceeding with the rest",
            stacklevel=2,
        )
        rs = rs[errors == 0]
        vs = vs[errors == 0]
        times = times[errors == 0]

    cart_teme = CartesianRepresentation(
        rs << u.km,
        xyz_axis=-1,
        differentials=CartesianDifferential(
            vs << (u.km / u.s),
            xyz_axis=-1,
        ),
    )
    cart_gcrs = (
        TEME(cart_teme, obstime=times).transform_to(GCRS(obstime=times)).cartesian
    )

    return Ephem(cart_gcrs, times, plane=Planes.EARTH_EQUATOR)


from poliastro.bodies import Earth
from poliastro.plotting import OrbitPlotter3D

from itertools import islice

epochs = time_range(now, end=now + (90 << u.minute))
starlink_ephem = ephem_from_gp(sat, epochs)

plotter = OrbitPlotter3D()
plotter.set_attractor(Earth)
plotter.plot_ephem(starlink_ephem, color="#333", label=sat.satnum, trail=True)

for n_starlink in islice(load_gp_from_celestrak(name="STARLINK"), 25):
    n_starlink_ephem = ephem_from_gp(n_starlink, epochs)
    plotter.plot_ephem(
        n_starlink_ephem, color="#666", label=n_starlink.satnum, trail=True
    )

plotter.show()
