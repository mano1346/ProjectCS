"""
University of Amsterdam

Course: Project Computational Science
Authors: Emmanuel Mukeh, Justin Wong & Arjan van Staveren

This code runs a accuracy test for the SGP4 model by computing the difference
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

from astropy import units as u
from astropy.time import Time


def A_M_for_sat_v(sat_id):
    """Get frontal area (A) over mass (M) ratio based on the satellites year and launch
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


# Initialize end time/epoch, which has to be on the date the second data is from
end_time = datetime.datetime(2024, 1, 25, 0, 0, 0)
t_end = Time(end_time, format="datetime", scale="utc")


# Use SGP4 model to predict the position of a satellite on a given epoch (t)
def get_pos_satellite(sat, t):
    """ Get position and velocity using the SGP4 perdiction
        model, see: https://pypi.org/project/sgp4/
    """
    error, r, v = sat.sgp4(t.jd1, t.jd2)
    assert error == 0

    return np.array(r), np.array(v)


# Propagate the satellites from the first date to the end epoch
r0_date1 = []
v0_date1 = []
for sat in satellites_date_1:
    r, v = get_pos_satellite(sat, t_end)
    r0_date1.append(r)
    v0_date1.append(v)

# Propagate the satellites from the second date to the end epoch
r0_date2 = []
v0_date2 = []
for sat in satellites_date_2:
    r, v = get_pos_satellite(sat, t_end)
    r0_date2.append(r)
    v0_date2.append(v)


# Calculating position & velocity deviations between the data from the first date and the second date

diff_positions_list = []
diff_velocities_list = []

for i in range(len(r0_date1)):
    diff_position = np.abs(r0_date2[i] - r0_date1[i])
    diff_velocity = np.abs(v0_date2[i] - v0_date1[i])

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
    using the SGP4 model, and satellite data from that epoch""",
    fontsize=24,
)

fig.supxlabel("Satellite number", fontsize=22)
fig.supylabel("Position deviation in km", fontsize=22)

plt.savefig("accuracy_test_sgp4_model.png")
plt.show()
