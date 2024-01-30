# import cProfile

# cProfile.run("foo()")
import os
import numpy as np
import re
import time
import datetime

from sgp4.api import Satrec
from sgp4 import omm

from numba import njit as jit

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import TEME, GCRS
from astropy.coordinates import CartesianRepresentation, CartesianDifferential

from poliastro.core.perturbations import atmospheric_drag_exponential, J2_perturbation
from poliastro.constants import rho0_earth, H0_earth
from poliastro.bodies import Earth
from poliastro.twobody.propagation.vallado import vallado as propagate

from scipy.spatial.distance import pdist

import pickle


R = Earth.R.to(u.km).value
k = Earth.k.to(u.km**3 / u.s**2).value
J2 = Earth.J2.value
C_D = 2.2
rho0 = rho0_earth.to(u.kg / u.km**3).value  # kg/km^3
H0 = H0_earth.to(u.km).value

A_over_m_sats = []


def A_M_for_sat_v(sat_id):
    if sat_id[0] == 2023 and sat_id[1] in (26, 56, 67, 79, 96):
        return 5.6 * (10 ** (-6)) / 800

    elif sat_id[0] == 2019 or sat_id[0] == 2020:
        return 5.6 * (10 ** (-6)) / 260

    elif sat_id[0] == 2021:
        if sat_id[1] <= 44:
            return 5.6 * (10 ** (-6)) / 260
        else:
            return 5.6 * (10 ** (-6)) / 300

    elif sat_id[0] == 2022:
        return 5.6 * (10 ** (-6)) / 300

    elif sat_id[0] == 2022:
        return 5.6 * (10 ** (-6)) / 300

    elif sat_id[0] == 2023:
        if sat_id[1] <= 99:
            return 5.6 * (10 ** (-6)) / 300
        elif sat_id[1] < 170:
            return 5.6 * (10 ** (-6)) / 800
        else:
            return None
    else:
        return None


satellite_names = []
satellites = []
max_count = 5600
random_satellite_count = 0
latest_epoch = 0

with open(os.path.join(os.path.dirname(__file__), "starlink_23_01.xml")) as xml:
    segments = omm.parse_xml(xml)

    count = 0
    for segment in segments:
        sat_id = re.findall("\d+", segment["OBJECT_ID"])
        sat_id = [int(i) for i in sat_id]
        A_m = A_M_for_sat_v(sat_id)
        if A_m != None:
            sat = Satrec()
            omm.initialize(sat, segment)

            satellites.append(sat)
            satellite_names.append(segment["OBJECT_NAME"])
            A_over_m_sats.append(A_m)

            count += 1
            if count >= max_count:
                break


def get_pos_satellite(sat, t):
    error, r, v = sat.sgp4(t.jd1, t.jd2)
    assert error == 0

    # teme = CartesianRepresentation(
    #     r << u.km,
    #     xyz_axis=-1,
    #     differentials=CartesianDifferential(
    #         v << (u.km / u.s),
    #         xyz_axis=-1,
    #     ),
    # )
    # gcrs = TEME(teme, obstime=t).transform_to(GCRS(obstime=t))

    # r = (gcrs.cartesian.x.value, gcrs.cartesian.y.value, gcrs.cartesian.z.value)
    # v = (gcrs.velocity.d_x.value, gcrs.velocity.d_y.value, gcrs.velocity.d_z.value)

    return np.array(r), np.array(v)


@jit
def a_perturbations(t0, state, k, J2, R, C_D, A_over_m, H0, rho0, perturbation_chance):
    if perturbation_chance < np.random.random():
        return np.array([0.0, 0.0, 0.0])

    per1 = J2_perturbation(t0, state, k, J2, R)
    per2 = atmospheric_drag_exponential(t0, state, k, R, C_D, A_over_m, H0, rho0)
    return per1 + per2


def propagate_n_satellites(sat_r, sat_v, tof, curr_time):
    sat_new_r = []
    sat_new_v = []
    for i in range(len(sat_r)):
        dv = a_perturbations(
            curr_time,
            np.concatenate([sat_r[i], sat_v[i]]),
            k,
            J2,
            R,
            C_D,
            A_over_m_sats[i],
            H0,
            rho0,
            perturbation_chance=1.0,
        )
        r, v = propagate(k, sat_r[i], sat_v[i] + dv, tof, numiter=350)
        sat_new_r.append(r)
        sat_new_v.append(v)

    return sat_new_r, sat_new_v


from octree import generate_octree

generate_histogram = True
if generate_histogram:
    file_name = ""
    while file_name == "":
        file_name = input("File name for histogram data: ")
        if os.path.exists(os.path.join(os.path.dirname(__file__), f"{file_name}.pkl")):
            file_name = ""
            print("A file with this name already exists.\n")

    pairs = []
    for value1 in range(len(satellites) + random_satellite_count):
        for value2 in range(value1 + 1, len(satellites) + random_satellite_count):
            pairs.append((value1, value2))
    pairs = np.array(pairs)

bins = [5, 10, 20, 30, 40]


def get_hist_data(sat_positions):
    distances = pdist(sat_positions)
    sat_hist_count = []
    for bin_threshold in bins:
        sat_hist_count.append(
            len(np.unique(pairs[(distances < bin_threshold).nonzero()[0]].flatten()))
        )
    return sat_hist_count


def normalize_vector(vec):
    return vec / ((vec**2).sum() ** 0.5)


# Implemented according to https://math.stackexchange.com/questions/137362/how-to-find-perpendicular-vector-to-another-vector#comment316390_137362
def perpendicular_vector(vec):
    for i in range(3):
        if vec[i] != 0:
            new_vec = np.array([0, 0, 0])
            new_vec[(i + 1) % 3] = -vec[i]
            new_vec[i] = vec[(i + 1) % 3]

            return new_vec


from scipy.spatial.transform import Rotation


def create_random_satellite(height):
    x, y, z = 0, 0, 0
    while x + y + z == 0:
        x, y, z = np.random.normal(), np.random.normal(), np.random.normal()

    pos = np.array([x, y, z])
    pos_normalized = normalize_vector(pos)
    pos = pos_normalized * height

    speed_factor = (k / height) ** 0.5

    velocity = perpendicular_vector(pos)
    velocity = normalize_vector(velocity) * speed_factor

    rotation = Rotation.from_rotvec(np.pi * 2 * np.random.random() * pos_normalized)
    velocity = rotation.apply(velocity)

    return pos, velocity


start_time = datetime.datetime(2024, 1, 23, 10, 0, 0)
start_time_in_s = start_time.timestamp()
t_start = Time(start_time, format="datetime", scale="utc")

simulation_length = 3600
tof = 1
satellite_r = [[] for _ in range(simulation_length + 1)]
satellite_v = [[] for _ in range(simulation_length + 1)]

start = time.process_time()
r0 = []
v0 = []
for sat in satellites:
    r, v = get_pos_satellite(sat, t_start)
    r0.append(r)
    v0.append(v)

for _ in range(random_satellite_count):
    r, v = create_random_satellite(6371 + 530)
    r0.append(r)
    v0.append(v)

satellite_r[0].extend(r0)
satellite_v[0].extend(v0)

print(f"initializing time :{time.process_time() - start}")
if generate_histogram:
    hist_counts = [get_hist_data(satellite_r[0])]

start = time.process_time()
print("Start simulation")
for i in range(simulation_length):
    ri, vi = propagate_n_satellites(
        satellite_r[i], satellite_v[i], tof, start_time_in_s + (i * tof)
    )
    satellite_r[i + 1].extend(ri)
    satellite_v[i + 1].extend(vi)

    if generate_histogram:
        hist_count = get_hist_data(ri)

        # print(hist_count)
        hist_counts.append(hist_count)

print(f"Simulation time :{time.process_time() - start}")
from satellite_visualization import visualize_data

if generate_histogram:
    with open(
        os.path.join(os.path.dirname(__file__), f"{file_name}.pkl"), "xb"
    ) as file:
        pickle.dump(hist_counts, file)

if generate_histogram:
    visualize_data(satellite_r, hist_counts, bins)
else:
    visualize_data(satellite_r)
