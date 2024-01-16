from astropy import units as u
from astropy.time import Time
from astropy.coordinates import CartesianRepresentation

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from sgp4 import omm
from sgp4.api import Satrec

import numpy as np

import os


def get_orbit_for_satellite(sat: Satrec):
    jd1, jd2 = sat.jdsatepoch, sat.jdsatepochF
    error, r, v = sat.sgp4(jd1, jd2)
    assert error == 0

    return Orbit.from_vectors(Earth, r << u.km, v << u.km / u.s, epoch = Time(jd1, jd2, format = 'jd'))


satellite_names = []
orbits = []
max_count = 5
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


satellite_coords = []

for orbit in orbits:
    orbit = orbit.propagate(latest_epoch - orbit.epoch)

    coords = []
    for i in range(20):
        orbit = orbit.propagate(1 << u.min)
        coords.append(orbit.r)

    coords = np.array(coords)
    satellite_coords.append(coords)


from poliastro.plotting.interactive import OrbitPlotter3D

plotter = OrbitPlotter3D()
plotter.set_attractor(Earth)
for coords, name in zip(satellite_coords, satellite_names):
    plotter.plot_trajectory(CartesianRepresentation(coords << u.km, xyz_axis=-1), label=name)

plotter.show()