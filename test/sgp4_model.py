import os
import datetime
import numpy as np
import time

from sgp4.api import Satrec, jday
from sgp4.conveniences import sat_epoch_datetime
from sgp4 import omm

from scipy.spatial.distance import pdist

satellite_names = []
satellites = []
max_count = 560
latest_epoch = 0


with open(os.path.join(os.path.dirname(__file__), "starlink.xml")) as xml:
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


def propagate_n_satellites(satellites, time_delta):
    coords = []
    for sat in satellites:
        r, v = propagate_satellite(sat, time_delta + datetime.timedelta(days = latest_epoch - sat.epochdays))
        coords.append(r)

    distances = pdist(coords)
    return coords, distances


start = time.process_time()

simulation_length = 1000
timestep = datetime.timedelta(seconds=1)
satellite_coords = [[] for _ in range(simulation_length)]
time_delta = timestep

for i in range(simulation_length):
    coords, distances = propagate_n_satellites(satellites, time_delta)
    satellite_coords[i].extend(coords)

    # print(np.mean(distances))

    time_delta += timestep
    print(time_delta)


from satellite_visualization import visualize_data

visualize_data(satellite_coords)

print(time.process_time() - start)