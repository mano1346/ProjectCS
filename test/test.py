from astropy import units as u
from astropy.time import Time
from astropy.coordinates import CartesianRepresentation

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from sgp4 import omm
from sgp4.api import Satrec

import numpy as np

import os


def get_orbit_for_satellite(sat: Satrec):
    jd1, jd2 = sat.jdsatepoch, sat.jdsatepochF
    error, r, v = sat.sgp4(jd1, jd2)
    assert error == 0

    return Orbit.from_vectors(
        Earth, r << u.km, v << u.km / u.s, epoch=Time(jd1, jd2, format="jd")
    )


satellite_names = []
orbits = []
max_count = 5
latest_epoch = Time(0, 0, format="jd")

with open(os.path.join(os.path.dirname(__file__), "starlink.xml")) as xml:
    segments = omm.parse_xml(xml)

    count = 0
    for segment in segments:
        sat = Satrec()
        omm.initialize(sat, segment)

        orbit = get_orbit_for_satellite(sat)
        if orbit.epoch > latest_epoch:
            latest_epoch = orbit.epoch

        orbits.append(orbit)
        satellite_names.append(segment["OBJECT_NAME"])

        count += 1
        if count >= max_count:
            break


satellite_coords = []
from poliastro.plotting.interactive import OrbitPlotter3D
import plotly.graph_objects as go


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "redraw": False,
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
    }


plotter = OrbitPlotter3D()
plotter.set_attractor(Earth)


fig = go.Figure()

add_earth = False
i = 0
for orbit_i, name in zip(orbits, satellite_names):
    orbit_i = orbit_i.propagate(latest_epoch - orbit_i.epoch)
    orbits[i] = orbit_i
    trace = OrbitPlotter3D().plot(orbit_i, color="orange")
    if not add_earth:
        trace.data[1]["name"] = name
        trace.data[2]["name"] = name + "-rev"
        fig.add_trace(trace.data[0])
        fig.add_trace(trace.data[3])
        fig.add_trace(trace.data[1])
        fig.add_trace(trace.data[2])
        add_earth = True
    else:
        trace.data[1]["name"] = name
        trace.data[2]["name"] = name + "-rev"
        fig.add_trace(trace.data[1])
        fig.add_trace(trace.data[2])
    i += 1

nameDataindexdict = {}
for dt in fig.data:
    nameDataindexdict[dt["name"]] = fig.data.index(dt)

frames = []
n_frames = 100
for k in range(n_frames):
    new_data = []
    new_traces = []
    i = 0
    for orbit_i, name in zip(orbits, satellite_names):
        orbit_i = orbit_i.propagate(1 << u.min)
        orbits[i] = orbit_i
        trace = OrbitPlotter3D().plot(orbit_i, color="orange")
        trace.data[1]["name"] = name
        trace.data[2]["name"] = name + "-rev"
        new_data.append(trace.data[1])
        new_data.append(trace.data[2])
        new_traces.append(nameDataindexdict[name])
        new_traces.append(nameDataindexdict[name + "-rev"])

        i += 1

    frames.append(
        go.Frame(
            data=new_data,
            traces=new_traces,
            name=f"fr{k}",
        )
    )
fig.update(frames=frames)

sliders = [
    {
        "pad": {"b": 10, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {
                "args": [None, frame_args(0)],
                "label": str(k),
                "method": "animate",
            }
            for k, f in enumerate(fig.frames)
        ],
    }
]
fig.update_layout(
    updatemenus=[
        {
            "buttons": [
                {
                    "args": [None, frame_args(0)],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [[None], frame_args(0)],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "type": "buttons",
            "x": 0.1,
            "y": 0,
        }
    ],
    sliders=sliders,
)

fig.update_layout(sliders=sliders)
fig.show()
