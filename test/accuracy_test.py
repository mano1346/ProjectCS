"""
University of Amsterdam

Course: Project Computational Science
Authors: Emmanuel Mukeh, Justin Wong & Arjan van Staveren

This code runs a accuracy test for the our model by computing the difference
in position between a dataset from one date and a dataset from a later date, 
where the first dataset is propagated to the later date. 

"""

import os
import datetime
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib


from sgp4.api import Satrec
from sgp4 import omm

from numba import njit as jit

from astropy import units as u
from astropy.time import Time

from poliastro.core.perturbations import atmospheric_drag_exponential, J2_perturbation
from poliastro.constants import rho0_earth, H0_earth
from poliastro.bodies import Earth
from poliastro.core.propagation import vallado


# Constants needed for the perturbations
R = Earth.R.to(u.km).value
k = Earth.k.to(u.km**3 / u.s**2).value
J2 = Earth.J2.value
C_D = 2.2
rho0 = rho0_earth.to(u.kg / u.km**3).value
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

# Initialize the max amount of satellites you want to simulate
max_count = 5600
satellites_date_1 = []

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
            satellites_date_1.append(sat)
            A_over_m_sats.append(A_m)

            count += 1
            if count >= max_count:
                break


satellites_date_2 = []

with open(os.path.join(os.path.dirname(__file__), "starlink_25_01.xml")) as xml:

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
            satellites_date_2.append(sat)

            count += 1
            if count >= max_count:
                break

# Initialize start time/epoch, which has to be on the date the first dataset is from
start_time = datetime.datetime(2024, 1, 23, 0, 0, 0)
t_start = Time(start_time, format="datetime", scale="utc")
curr_time = start_time

# Initialize end time/epoch, which has to be on the date the second dataset is from
end_time = datetime.datetime(2024, 1, 25, 0, 0, 0)
t_end = Time(end_time, format="datetime", scale="utc")


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


def propagate(k, r, v, tof, numiter):
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
            curr_time.timestamp() + tof,
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


# Propagate the satellites from the first date to the start epoch
r0 = []
v0 = []
for sat in satellites_date_1:
    r, v = get_pos_satellite(sat, t_start)
    r0.append(r)
    v0.append(v)


# Propagate the satellites from start time to end time
i = 0
curr_time = start_time
tof = 1                    # Time of flight / timestep in seconds

ri_prev, vi_prev = r0, v0
while curr_time < end_time:
    ri, vi = propagate_n_satellites(ri_prev, vi_prev, tof, curr_time)
    ri_prev, vi_prev = ri, vi
    curr_time += datetime.timedelta(seconds=tof)
    i += 1


# Propagate the satellites from the second date to the end epoch
r0_new = []
v0_new = []
for sat in satellites_date_2:
    r, v = get_pos_satellite(sat, t_end)
    r0_new.append(r)
    v0_new.append(v)


# Calculating position & velocity deviations between the data from the first date and the second date
    
diff_positions_list = []
diff_velocities_list = []

for i in range(len(ri)):
    diff_position = np.abs(r0_new[i] - ri[i])
    diff_velocity = np.abs(v0_new[i] - vi[i])

    diff_positions_list.append(diff_position)
    diff_velocities_list.append(diff_velocity)

differences_x = [array[0] for array in diff_positions_list]
differences_y = [array[1] for array in diff_positions_list]
differences_z = [array[2] for array in diff_positions_list]


# Plot position deviations

matplotlib.rc("xtick", labelsize=20)
matplotlib.rc("ytick", labelsize=20)

fig, axes = plt.subplots(3, sharex=True, sharey=True, figsize=(11, 11))
axes[0].plot(range(len(differences_x)), differences_x, color="blue", label="x-position")
axes[0].legend(loc="upper left", fontsize=20)
axes[1].plot(
    range(len(differences_y)), differences_y, color="green", label="y-position"
)
axes[1].legend(loc="upper left", fontsize=20)
axes[2].plot(range(len(differences_z)), differences_z, color="red", label="z-position")
axes[2].legend(loc="upper left", fontsize=20)

fig.suptitle(
    f"""Position deviation of {len(differences_x)} satellites between satellite data 
    from 2 days before a certain epoch, which was propagated 
    using our model, and satellite data from that epoch""",
    fontsize=24,
)

fig.supxlabel("Satellite number", fontsize=22)
fig.supylabel("Position deviation in km", fontsize=22)

plt.savefig("accuracy_test_model.png")
plt.show()
