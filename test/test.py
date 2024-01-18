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

<<<<<<< Updated upstream
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
=======
    return Orbit.from_vectors(
        Earth, r << u.km, v << u.km / u.s, epoch=Time(jd1, jd2, format="jd")
    )


satellite_names = []
orbits = []
max_count = 5
latest_epoch = Time(0, 0, format="jd")

with open(os.path.join(os.path.dirname(__file__), "starlink.xml")) as xml:
    segments = omm.parse_xml(xml)
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
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
=======
        orbit = get_orbit_for_satellite(sat)
        if orbit.epoch > latest_epoch:
            latest_epoch = orbit.epoch

        orbits.append(orbit)
        satellite_names.append(segment["OBJECT_NAME"])

        count += 1
        if count >= max_count:
            break


satellite_coords = []
from poliastro.plotting.interactive import OrbitPlotter3D
import plotly.graph_objects as go


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "redraw": False,
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
    }


plotter = OrbitPlotter3D()
plotter.set_attractor(Earth)


fig = go.Figure()

add_earth = False
i = 0
for orbit_i, name in zip(orbits, satellite_names):
    orbit_i = orbit_i.propagate(latest_epoch - orbit_i.epoch)
    orbits[i] = orbit_i
    trace = OrbitPlotter3D().plot(orbit_i, color="orange")
    if not add_earth:
        trace.data[1]["name"] = name
        trace.data[2]["name"] = name + "-rev"
        fig.add_trace(trace.data[0])
        fig.add_trace(trace.data[3])
        fig.add_trace(trace.data[1])
        fig.add_trace(trace.data[2])
        add_earth = True
    else:
        trace.data[1]["name"] = name
        trace.data[2]["name"] = name + "-rev"
        fig.add_trace(trace.data[1])
        fig.add_trace(trace.data[2])
    i += 1

nameDataindexdict = {}
for dt in fig.data:
    nameDataindexdict[dt["name"]] = fig.data.index(dt)

frames = []
n_frames = 100
for k in range(n_frames):
    new_data = []
    new_traces = []
    i = 0
    for orbit_i, name in zip(orbits, satellite_names):
        orbit_i = orbit_i.propagate(1 << u.min)
        orbits[i] = orbit_i
        trace = OrbitPlotter3D().plot(orbit_i, color="orange")
        trace.data[1]["name"] = name
        trace.data[2]["name"] = name + "-rev"
        new_data.append(trace.data[1])
        new_data.append(trace.data[2])
        new_traces.append(nameDataindexdict[name])
        new_traces.append(nameDataindexdict[name + "-rev"])

        i += 1

    frames.append(
        go.Frame(
            data=new_data,
            traces=new_traces,
            name=f"fr{k}",
        )
    )
fig.update(frames=frames)

sliders = [
    {
        "pad": {"b": 10, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {
                "args": [None, frame_args(0)],
                "label": str(k),
                "method": "animate",
            }
            for k, f in enumerate(fig.frames)
        ],
    }
]
fig.update_layout(
    updatemenus=[
        {
            "buttons": [
                {
                    "args": [None, frame_args(0)],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [[None], frame_args(0)],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "type": "buttons",
            "x": 0.1,
            "y": 0,
        }
    ],
    sliders=sliders,
)

fig.update_layout(sliders=sliders)
fig.show()
>>>>>>> Stashed changes
