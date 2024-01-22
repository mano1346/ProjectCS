import os
import datetime
import numpy as np
import time

from sgp4.api import Satrec, jday
from sgp4.conveniences import sat_epoch_datetime
from sgp4 import omm

from scipy.spatial.distance import pdist

satellite_names = []
sattlites = []
max_count = 5600


with open(os.path.join(os.path.dirname(__file__), "starlink.xml")) as xml:
    segments = omm.parse_xml(xml)

    count = 0
    for segment in segments:
        sat = Satrec()
        omm.initialize(sat, segment)

        sattlites.append(sat)
        satellite_names.append(segment["OBJECT_NAME"])

        count += 1
        if count >= max_count:
            break


def time_to_jd_jdf(time):
    return jday(time.year, time.month, time.day, time.hour, time.minute, time.second)


def get_pos_satellite(sat):
    # Use the epoch from the Satrec object
    jd, jdF = sat.jdsatepoch, sat.jdsatepochF
    error, r, v = sat.sgp4(jd, jdF)
    assert error == 0

    return r, v


def propagate_satellite(sat, time_delta):
    # Use the epoch from the Satrec object
    jd_datetime = sat_epoch_datetime(sat)
    pro_jd, pro_jdF = time_to_jd_jdf(jd_datetime + time_delta)

    error, r, v = sat.sgp4(pro_jd, pro_jdF)
    assert error == 0

    return r, v


def propagate_n_satellites(sattlites, time_delta):
    coords = []
    for sat in sattlites:
        r, v = propagate_satellite(sat, time_delta)
        coords.append(r)

    distances = pdist(coords)
    return coords, distances


start = time.process_time()

simulation_length = 500
timestep = datetime.timedelta(seconds=1)
sattlite_coords = []
time_delta = timestep

for i in range(simulation_length):
    coords, distances = propagate_n_satellites(sattlites, time_delta)
    sattlite_coords.append(coords)
    # print(np.mean(distances))

    time_delta += timestep


print(time.process_time() - start)
