import os
import datetime
import numpy as np
import time

from sgp4.api import Satrec, jday
from sgp4.conveniences import sat_epoch_datetime
from sgp4 import omm

from vallado_propagator import vallado_propagate as propagate
from scipy.spatial.distance import pdist

satellite_names = []
satellites = []
max_count = 5600
latest_epoch = 0
k = 3.986004418 * (10**5)

with open(os.path.join(os.path.dirname(__file__), "starlink_11_01.xml")) as xml:
    segments = omm.parse_xml(xml)

    count = 0
    for segment in segments:
        sat = Satrec()
        omm.initialize(sat, segment)

        satellites.append(sat)
        satellite_names.append(segment["OBJECT_NAME"])

        if sat.epochdays > latest_epoch:
            latest_epoch = sat.epochdays

        count += 1
        if count >= max_count:
            break

jd, jdF = jday(2024, 1, 11, 0, 0, 0)


def get_pos_satellite(sat):
    error, r, v = sat.sgp4(jd, jdF)
    assert error == 0

    return r, v




def propagate_n_satellites(sat_r, sat_v, tof):
    sat_new_r = []
    sat_new_v = []
    for i in range(len(sat_r)):
        r, v = propagate(k, sat_r[i], sat_v[i], tof, numiter=350)
        sat_new_r.append(r)
        sat_new_v.append(v)

    return sat_new_r, sat_new_v



from octree import generate_octree

pairs = []
for value1 in range(len(satellites)):
    for value2 in range(value1 + 1, len(satellites)):
        pairs.append((value1, value2))
pairs = np.array(pairs)

bins = [10, 20, 30, 40, 50]

def get_hist_data(sat_positions):
    distances = pdist(sat_positions)
    sat_hist_count = []
    for bin_threshold in bins:
        sat_hist_count.append(len(np.unique(pairs[(distances < bin_threshold).nonzero()[0]].flatten())))
    return sat_hist_count



# satrec = satellites[0]
# print(satrec.v)
start = time.process_time()

simulation_length = 100
tof = 1
satellite_r = [[] for _ in range(simulation_length + 1)]
satellite_v = [[] for _ in range(simulation_length + 1)]

r0 = []
v0 = []
for sat in satellites:
    r, v = get_pos_satellite(sat)
    r0.append(np.array(r))
    v0.append(np.array(v))

satellite_r[0].extend(r0)
satellite_v[0].extend(v0)

hist_counts = [get_hist_data(satellite_r[0])]
for i in range(simulation_length):
    ri, vi = propagate_n_satellites(satellite_r[i], satellite_v[i], tof)
    satellite_r[i + 1].extend(ri)
    satellite_v[i + 1].extend(vi)

    hist_count = get_hist_data(ri)

    print(hist_count)
    hist_counts.append(hist_count)

print(time.process_time() - start)
from satellite_visualization import visualize_data

visualize_data(satellite_r, hist_counts, bins)

