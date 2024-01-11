from astropy import units as u
from astropy.time import Time

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from sgp4 import omm
from sgp4.api import Satrec

import numpy as np

with open("starlink.xml") as xml:
    segments = omm.parse_xml(xml)

    for segment in segments:
        print(segment['EPOCH'])
        sat = Satrec()
        omm.initialize(sat, segment)
        break

jd1, jd2 = sat.jdsatepoch, sat.jdsatepochF
e, r, v = sat.sgp4(jd1, jd2)

orbit = Orbit.from_vectors(Earth, r << u.km, v << u.km / u.s, epoch = Time(jd1, jd2, format = 'jd'))

coords = []
for _ in range(50):
    orbit = orbit.propagate(1 << u.min)
    coords.append(orbit.r.value)

coords = np.array(coords)

#orbit.plot()

from poliastro.plotting.interactive import OrbitPlotter3D
from astropy.coordinates import CartesianRepresentation

representation = CartesianRepresentation(coords[:, 0], coords[:, 1], coords[:, 2], unit = u.km)

plotter = OrbitPlotter3D()
plotter.set_attractor(Earth)
plotter.plot_trajectory(representation)