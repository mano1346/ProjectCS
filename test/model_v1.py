from astropy import units as u
from astropy.time import Time
from astropy.coordinates import CartesianRepresentation

from poliastro.bodies import Earth

from poliastro.twobody import Orbit, propagation
from poliastro.core.propagation import func_twobody

from sgp4 import omm
from sgp4.api import Satrec

import numpy as np
import pandas as pd

import os

from scipy.spatial.distance import pdist


def get_orbit_for_satellite(sat: Satrec):
    jd1, jd2 = sat.jdsatepoch, sat.jdsatepochF
    error, r, v = sat.sgp4(jd1, jd2)
    assert error == 0

    return Orbit.from_vectors(
        Earth, r << u.km, v << u.km / u.s, epoch=Time(jd1, jd2, format="jd")
    )


satellite_names = []
orbits = []
max_count = 10
latest_epoch = Time(0, 0, format="jd")

with open(os.path.join(os.path.dirname(__file__), "starlink.xml")) as xml:
    segments = omm.parse_xml(xml)

    count = 0
    for segment in segments:
        sat = Satrec()
        omm.initialize(sat, segment)

        orbit = get_orbit_for_satellite(sat)
        if orbit.epoch > latest_epoch:
            latest_epoch = orbit.epoch

        orbits.append(orbit)
        satellite_names.append(segment["OBJECT_NAME"])

        count += 1
        if count >= max_count:
            break

for i in range(len(orbits)):
    orbits[i] = orbits[i].propagate(latest_epoch - orbits[i].epoch)
    
simulation_length = 100
satellite_coords = [[] for _ in range(simulation_length)]

for timestep in range(simulation_length):
    for i in range(len(orbits)):
        orbits[i] = orbit = orbits[i].propagate(
            1 << u.min, method=propagation.FarnocchiaPropagator()
        )
        satellite_coords[timestep].append(orbit.r.value)

    distances = pdist(satellite_coords[timestep])

    print(np.mean(distances))


from satellite_visualization import visualize_data

visualize_data(satellite_coords)
