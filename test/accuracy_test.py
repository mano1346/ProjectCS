import os
import datetime
import numpy as np
import re


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
from poliastro._math.linalg import norm

import matplotlib.pyplot as plt

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


satellite_names_new = []
satellites_new = []
with open(os.path.join(os.path.dirname(__file__), "starlink_25_01.xml")) as xml:
    segments = omm.parse_xml(xml)

    count = 0
    for segment in segments:
        sat_id = re.findall("\d+", segment["OBJECT_ID"])
        sat_id = [int(i) for i in sat_id]
        A_m = A_M_for_sat_v(sat_id)
        if A_m != None:
            sat = Satrec()
            omm.initialize(sat, segment)

            satellites_new.append(sat)
            satellite_names_new.append(segment["OBJECT_NAME"])

            count += 1
            if count >= max_count:
                break

# N = 10
# satellite_names = satellite_names[-N:]
# satellites = satellites[-N:]
# A_over_m_sats = A_over_m_sats[-N:]
# satellite_names_new = satellite_names_new[-N:]
# satellites_new = satellites_new[-N:]

start_time = datetime.datetime(2024, 1, 24, 22, 0, 0)
t_start = Time(start_time, format="datetime", scale="utc")
curr_time = start_time

end_time = datetime.datetime(2024, 1, 25, 0, 0, 0)
t_end = Time(end_time, format="datetime", scale="utc")


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
def a_perturbations(t0, state, k, J2, R, C_D, A_over_m, H0, rho0):
    per1 = J2_perturbation(t0, state, k, J2, R)
    per2 = atmospheric_drag_exponential(t0, state, k, R, C_D, A_over_m, H0, rho0)
    return per1 + per2, A_over_m


def propagate_n_satellites(sat_r, sat_v, tof, curr_time):
    sat_new_r = []
    sat_new_v = []
    for i in range(len(sat_r)):
        dv, per2 = a_perturbations(
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

    distances = pdist(sat_new_r)
    return sat_new_r, sat_new_v


tof = 1
satellite_r = []
satellite_v = []

r0 = []
v0 = []
for sat in satellites:
    r, v = get_pos_satellite(sat, t_start)
    r0.append(r)
    v0.append(v)

satellite_r.append(r0)
satellite_v.append(v0)

i = 0
curr_time = start_time
while curr_time < end_time:
    ri, vi = propagate_n_satellites(satellite_r[i], satellite_v[i], tof, curr_time)
    satellite_r.append(ri)
    satellite_v.append(vi)
    curr_time += datetime.timedelta(seconds=tof)
    i += 1

r0_new = []
v0_new = []
for sat in satellites_new:
    r, v = get_pos_satellite(sat, t_end)
    r0_new.append(r)
    v0_new.append(v)

diff_positions_list = []
diff_velocities_list = []

for i in range(len(ri)):
    diff_position = np.abs(r0_new[i] - satellite_r[-1][i])
    diff_velocity = np.abs(v0_new[i] - satellite_v[-1][i])

    diff_positions_list.append(diff_position)
    diff_velocities_list.append(diff_velocity)

differences_x = [array[0] for array in diff_positions_list]
differences_y = [array[1] for array in diff_positions_list]
differences_z = [array[2] for array in diff_positions_list]

fig, axes = plt.subplots(3, sharex=True, sharey=True, figsize=(10, 5))
axes[0].plot(range(len(differences_x)), differences_x, color="blue", label="x-position")
axes[0].legend(loc="upper left")
axes[1].plot(
    range(len(differences_y)), differences_y, color="green", label="y-position"
)
axes[1].legend(loc="upper left")
axes[2].plot(range(len(differences_z)), differences_z, color="red", label="z-position")
axes[2].legend(loc="upper left")

fig.suptitle(
    f"""Position deviation of {len(differences_x)} satellites between satellite data from 2 days before a certain 
epoch, which was propagated using our model, and satellite data from that epoch""",
    fontsize=15,
)

fig.supxlabel("Satellite number", fontsize=13)
fig.supylabel("Position deviation in km", fontsize=13)

plt.savefig("accuracy_test_model.png")
# plt.show()
