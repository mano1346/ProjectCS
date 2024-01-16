#%matplotlib widget

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
max_count = 10
latest_epoch = Time(0, 0, format = 'jd')

with open(os.path.join(os.path.dirname(__file__), 'starlink.xml')) as xml:
    segments = omm.parse_xml(xml)

    count = 0
    for segment in segments:
        print(count)
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
df = pd.DataFrame(columns=['time','x','y','z'])


from poliastro.maneuver import Maneuver

dv = [1000, 0, 0] << (u.m / u.s)
imp = Maneuver.impulse(dv)


for orbit in orbits:
    orbit = orbit.propagate(latest_epoch - orbit.epoch)
    orbit = orbit.apply_maneuver(imp)

    coords = []
    for i in range(300):
        orbit = orbit.propagate(1 << u.min)
        coords.append(orbit.r)
        df.loc[len(df.index)] = ([i] + list(orbit.r.value))
        print(i)

    coords = np.array(coords)
    satellite_coords.append(coords)


# from poliastro.plotting.interactive import OrbitPlotter3D

# plotter = OrbitPlotter3D()
# plotter.set_attractor(Earth)
# for coords, name in zip(satellite_coords, satellite_names):
#     plotter.plot_trajectory(CartesianRepresentation(coords << u.km, xyz_axis=-1), label=name)

# plotter.show()

# Code below largely taken from
# https://stackoverflow.com/a/41609238
# Author: ImportanceOfBeingErnest
# Jan 12 2017
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import ipympl


def update_graph(num):
    data=df[df['time']==num]
    graph._offsets3d = (data.x, data.y, data.z)
    title.set_text('3D Test, time={}'.format(num))
    return num


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

data=df[df['time']==0]
graph = ax.scatter(data.x, data.y, data.z)

ani = animation.FuncAnimation(fig, update_graph, 300, 
                               interval=100, blit=False)

plt.show()