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
from astropy.coordinates import TEME, GCRS
from astropy.coordinates import CartesianRepresentation, CartesianDifferential

from poliastro.core.perturbations import atmospheric_drag_exponential, J2_perturbation
from poliastro.constants import rho0_earth, H0_earth
from poliastro.bodies import Earth
from poliastro.twobody.propagation.vallado import vallado as propagate

from scipy.spatial.distance import pdist


satellite_names = []
satellites = []
max_count = 100
latest_epoch = 0


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

# N = 1000
# satellite_names = satellite_names[-N:]
# satellites = satellites[-N:]
# satellite_names_new = satellite_names_new[-N:]
# satellites_new = satellites_new[-N:]

start_time = datetime.datetime(2024, 1, 23, 0, 0, 0)
t_start = Time(start_time, format="datetime", scale="utc")
curr_time = start_time

end_time = datetime.datetime(2024, 1, 25, 0, 0, 0)
t_end = Time(end_time, format="datetime", scale="utc")


def get_pos_satellite(sat, t):
    error, r, v = sat.sgp4(t.jd1, t.jd2)
    assert error == 0

    return np.array(r), np.array(v)


r0 = []
v0 = []
for sat in satellites:
    r, v = get_pos_satellite(sat, t_end)
    r0.append(r)
    v0.append(v)

r0_new = []
v0_new = []
for sat in satellites_new:
    r, v = get_pos_satellite(sat, t_end)
    r0_new.append(r)
    v0_new.append(v)

diff_positions_list = []
diff_velocities_list = []

for i in range(len(r0)):
    diff_position = np.abs(r0_new[i] - r0[i])
    diff_velocity = np.abs(v0_new[i] - r0[i])

    diff_positions_list.append(diff_position)
    diff_velocities_list.append(diff_velocity)

differences_x = [array[0] for array in diff_positions_list]
differences_y = [array[1] for array in diff_positions_list]
differences_z = [array[2] for array in diff_positions_list]

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
    using the SGP4 model, and satellite data from that epoch""",
    fontsize=24,
)

fig.supxlabel("Satellite number", fontsize=22)
fig.supylabel("Position deviation in km", fontsize=22)

plt.savefig("accuracy_test_sgp4_model.png")
plt.show()
