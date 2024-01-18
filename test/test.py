from astropy import units as u
from astropy.time import Time
from astropy.coordinates import CartesianRepresentation

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from sgp4 import omm
from sgp4.api import Satrec

import numpy as np
import pandas as pd

import os


def get_orbit_for_satellite(sat: Satrec):
    jd1, jd2 = sat.jdsatepoch, sat.jdsatepochF
    error, r, v = sat.sgp4(jd1, jd2)
    assert error == 0

    return Orbit.from_vectors(Earth, r << u.km, v << u.km / u.s, epoch = Time(jd1, jd2, format = 'jd'))


satellite_names = []
orbits = []
max_count = 50
latest_epoch = Time(0, 0, format = 'jd')

with open(os.path.join(os.path.dirname(__file__), 'starlink.xml')) as xml:
    segments = omm.parse_xml(xml)

    count = 0
    for segment in segments:
        sat = Satrec()
        omm.initialize(sat, segment)

        orbit = get_orbit_for_satellite(sat)

        if (orbit.epoch > latest_epoch):
            latest_epoch = orbit.epoch

        orbits.append(orbit)
        satellite_names.append(segment['OBJECT_NAME'])

        count += 1
        if count >= max_count:
            break


def distance_between_orbits(orbit1 : Orbit, orbit2 : Orbit):
    coords1, coords2 = np.array(orbit1.r.value), np.array(orbit2.r.value)

    return np.power(coords1 - coords2, 2).sum() ** 0.5


for i in range(len(orbits)):
    orbits[i] = orbits[i].propagate(latest_epoch - orbit.epoch)


satellite_coords = [list()] * len(orbits)
for _ in range(50):
    for i in range(len(orbits)):
        orbits[i] = orbit = orbits[i].propagate(1 << u.min)
        # satellite_coords[i].append(orbit.r)
    
    distances = []
    for i, orbit in enumerate(orbits[:-1]):
        for other_orbit in orbits[i+1: ]:
            assert orbit != other_orbit
            distance = distance_between_orbits(orbit, other_orbit)
            distances.append(distance)
    
    print(np.mean(distances))

    
# from poliastro.plotting.interactive import OrbitPlotter3D

# plotter = OrbitPlotter3D()
# plotter.set_attractor(Earth)
# for coords, name in zip(satellite_coords, satellite_names):
#     plotter.plot_trajectory(CartesianRepresentation(coords << u.km, xyz_axis=-1), label=name)

# plotter.show()

