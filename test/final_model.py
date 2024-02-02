"""
University of Amsterdam

Course: Project Computational Science
Authors: Emmanuel Mukeh, Justin Wong & Arjan van Staveren

This code runs a simulation that propagates satellites for a given
time and returns a visualisation and histogram data of the amount
of times satellites where in a certain proximity of eachother for
each timestep.

"""

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
from poliastro.core.propagation import vallado

from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation

import pickle


# Constants needed for the perturbations
R = Earth.R.to(u.km).value
k = Earth.k.to(u.km**3 / u.s**2).value
J2 = Earth.J2.value
C_D = 2.2
rho0 = rho0_earth.to(u.kg / u.km**3).value  # kg/km^3
H0 = H0_earth.to(u.km).value



def A_M_for_sat_v(sat_id):
    """ Get frontal area (A) over mass (M) ratio based on the satellites year and launch
        number, relating to which version of the Starlink it is and returns the corresponding ratio.
    """

    # If a sat_id is from a launch after 2023_170 it's filterd out since that satellite might not be
    # operating in it's final orbit yet.

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

# Initialize the max amount of satellites, and how many random satellites, you want to simulate
max_count = 5600
random_satellite_count = 0
satellites = []
A_over_m_sats = []


# List to store all the A/m ratio's for the satellites from the first date
A_over_m_sats = []

with open(os.path.join(os.path.dirname(__file__), "starlink_23_01.xml")) as xml:

    segments = omm.parse_xml(xml)  # Convert a segment of the xml to the satellite's orbital data in OMM format
    count = 0
    for segment in segments:

        # Obtain the year and launch number from the satellites object id
        sat_id = re.findall("\d+", segment["OBJECT_ID"])
        sat_id = [int(i) for i in sat_id]

        A_m = A_M_for_sat_v(sat_id)
        if A_m != None:

            # Add Satellite as Satrec object to the list
            sat = Satrec()
            omm.initialize(sat, segment)
            satellites.append(sat)
            A_over_m_sats.append(A_m)

            count += 1
            if count >= max_count:
                break


def get_pos_satellite(sat, t):
    """ Get position and velocity using the SGP4 perdiction
        model, see: https://pypi.org/project/sgp4/
    """
    error, r, v = sat.sgp4(t.jd1, t.jd2)
    assert error == 0

    return np.array(r), np.array(v)


@jit
def dv_perturbations(t0, state, k, J2, R, C_D, A_over_m, H0, rho0):
    """ Calculate J2 perturbation and atmospheric drag perturbations using
        the formulas from: 
        Curtis, Howard, Orbital mechanics for engineering students, Butterworth-Heinemann, 2013.
    
        See poliastro for more: https://docs.poliastro.space/en/stable/autoapi/poliastro/core/perturbations/index.html
    """

    return J2_perturbation(t0, state, k, J2, R) + atmospheric_drag_exponential(t0, state, k, R, C_D, A_over_m, H0, rho0)


def propagate(k, r0, v0, tof, numiter):
    """ Calculate new position and velocity by getting the Lagrange coefficients with the
        Newton-Raphson method.
        
        Formulas from: 
        Curtis, Howard, Orbital mechanics for engineering students, Butterworth-Heinemann, 2013.
    
        See poliastro for more: https://docs.poliastro.space/en/stable/autoapi/poliastro/core/propagation/vallado/index.html
    """

    # Compute Lagrange coefficients
    f, g, fdot, gdot = vallado(k, r0, v0, tof, numiter)

    r = f * r0 + g * v0
    v = fdot * r0 + gdot * v0

    return r, v


def propagate_n_satellites(sat_r, sat_v, tof, curr_time):
    """ Propagate all satellites and qdd the acceleration 
        due to the perturbations. Return lists of new positions
        and velocity's for all satellites
    """
    sat_new_r = []
    sat_new_v = []
    for i in range(len(sat_r)):
        dv = dv_perturbations(
            curr_time + tof,
            np.concatenate([sat_r[i], sat_v[i]]),
            k,
            J2,
            R,
            C_D,
            A_over_m_sats[i],
            H0,
            rho0,
        )
        r, v = propagate(k, sat_r[i], sat_v[i] + dv, tof, numiter=350)
        sat_new_r.append(r)
        sat_new_v.append(v)

    return sat_new_r, sat_new_v



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


# Initialize start time/epoch
start_time = datetime.datetime(2024, 1, 23, 10, 0, 0)
start_time_in_s = start_time.timestamp()
t_start = Time(start_time, format="datetime", scale="utc")


# Initialize simulation length and timestep (Time of flight)
simulation_length = 100
tof = 1
satellite_r = [[] for _ in range(simulation_length + 1)]
satellite_v = [[] for _ in range(simulation_length + 1)]

# Propagate the satellites to the start epoch
r0 = []
v0 = []
for sat in satellites:
    r, v = get_pos_satellite(sat, t_start)
    r0.append(r)
    v0.append(v)


# Add random satellites
for _ in range(random_satellite_count):
    r, v = create_random_satellite(6371 + 530)
    r0.append(r)
    v0.append(v)

satellite_r[0].extend(r0)
satellite_v[0].extend(v0)

if generate_histogram:
    hist_counts = [get_hist_data(satellite_r[0])]


# Start of the simulation
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
        hist_counts.append(hist_count)


print(f"Simulation time :{time.process_time() - start}")



from satellite_visualization import visualize_data

# Create file with the histogram data
if generate_histogram:
    with open(
        os.path.join(os.path.dirname(__file__), f"{file_name}.pkl"), "xb"
    ) as file:
        pickle.dump(hist_counts, file)

# Visualize the simulation
if generate_histogram:
    visualize_data(satellite_r, hist_counts, bins)
else:
    visualize_data(satellite_r)
