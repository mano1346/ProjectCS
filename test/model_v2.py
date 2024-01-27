import os
import datetime
import numpy as np
import time

from sgp4.api import Satrec, jday
from sgp4.conveniences import sat_epoch_datetime
from sgp4 import omm

from poliastro.twobody.propagation.vallado import vallado as propagate

# from vallado_propagator import vallado_propagate as propagate

from scipy.spatial.distance import pdist

satellite_names = []
satellites = []
max_count = 5600
random_satellite_count = 0
latest_epoch = 0
k = 3.986004418 * (10**5)

with open(os.path.join(os.path.dirname(__file__), "starlink_11_01.xml")) as xml:
    segments = omm.parse_xml(xml)

    count = 0
    for segment in segments:
        if count >= max_count:
            break
        
        count += 1

        sat = Satrec()
        omm.initialize(sat, segment)

        satellites.append(sat)
        satellite_names.append(segment["OBJECT_NAME"])

        if sat.epochdays > latest_epoch:
            latest_epoch = sat.epochdays

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

generate_histogram = True
if generate_histogram:
    pairs = []
    for value1 in range(len(satellites) + random_satellite_count):
        for value2 in range(value1 + 1, len(satellites) + random_satellite_count):
            pairs.append((value1, value2))
    pairs = np.array(pairs)

bins = [10, 20, 30, 40, 50]


def get_hist_data(sat_positions):
    distances = pdist(sat_positions)
    sat_hist_count = []
    for bin_threshold in bins:
        sat_hist_count.append(
            len(np.unique(pairs[(distances < bin_threshold).nonzero()[0]].flatten()))
        )
    return sat_hist_count


def normalize_vector(vec):
    return vec / ((vec ** 2).sum() ** 0.5)

# Implemented according to https://math.stackexchange.com/questions/137362/how-to-find-perpendicular-vector-to-another-vector#comment316390_137362 
def perpendicular_vector(vec):
    for i in range(3):
        if vec[i] != 0:
            new_vec = np.array([0,0,0])
            new_vec[(i + 1) % 3] = -vec[i]
            new_vec[i] = vec[(i + 1) % 3]

            return new_vec


from scipy.spatial.transform import Rotation

def create_random_satellite(height):
    x,y,z = 0,0,0
    while x+y+z == 0:
        x, y, z = np.random.normal(), np.random.normal(), np.random.normal()
    
    pos = np.array([x,y,z])
    pos_normalized = normalize_vector(pos)
    pos = pos_normalized * height

    speed_factor = (k/height) ** 0.5

    velocity = perpendicular_vector(pos)
    velocity = normalize_vector(velocity) * speed_factor

    rotation = Rotation.from_rotvec(np.pi*2*np.random.random() * pos_normalized)
    velocity = rotation.apply(velocity)

    return pos, velocity


# def apply_impulse()

start = time.process_time()

simulation_length = 100
tof = 60
satellite_r = [[] for _ in range(simulation_length + 1)]
satellite_v = [[] for _ in range(simulation_length + 1)]

r0 = []
v0 = []
for sat in satellites:
    r, v = get_pos_satellite(sat)
    r0.append(np.array(r))
    v0.append(np.array(v))

for _ in range(random_satellite_count):
    r, v = create_random_satellite(6371 + 530)
    r0.append(r)
    v0.append(v)

satellite_r[0].extend(r0)
satellite_v[0].extend(v0)

if generate_histogram:
    hist_counts = [get_hist_data(satellite_r[0])]
for i in range(simulation_length):
    ri, vi = propagate_n_satellites(satellite_r[i], satellite_v[i], tof)
    satellite_r[i + 1].extend(ri)
    satellite_v[i + 1].extend(vi)

    if generate_histogram:
        hist_count = get_hist_data(ri)

        print(hist_count)
        hist_counts.append(hist_count)

print(time.process_time() - start)
from satellite_visualization import visualize_data, plot_hist_counts

if generate_histogram:
    plot_hist_counts(hist_counts, 4)
    visualize_data(satellite_r, hist_counts, bins)
else:
    visualize_data(satellite_r)
