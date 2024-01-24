import os
import spiceypy as spice
from datetime import datetime, timedelta
from sgp4.api import Satrec, WGS72
from sgp4.conveniences import sat_epoch_datetime
from sgp4 import omm

# Load SPICE kernels
spice.furnsh(
    r"C:\Users\31653\Desktop\ProjectCS\test\SpiceKernels\de438.bsp"
)  # Example SPICE kernel for planetary ephemeris
spice.furnsh(r"C:\\Users\\arjan\\OneDrive\\Documents\\GitHub\\ProjectCS\\test\\SpiceKernels")
spice.furnsh(r"C:\Users\31653\Desktop\ProjectCS\test\SpiceKernels\jup344.bsp")


def get_satellite_state_at_epoch(satrec):
    # Convert OMM epoch to Ephemeris Time (ET)
    omm_epoch_str = sat_epoch_datetime(sat).strftime("%Y-%m-%d %H:%M:%S.%f")
    et = spice.str2et(omm_epoch_str)

    # Get the state vector at the epoch using SPICE
    state, _ = spice.spkgeo(satrec.satnum, et, ref="J2000", obs=0)
    position = state[:3]
    velocity = state[3:]

    return position, velocity


def propagate_satellite_spice(satrec, time_delta):
    # Convert OMM epoch to Ephemeris Time (ET)
    omm_epoch_str = sat_epoch_datetime(sat).strftime("%Y-%m-%d %H:%M:%S.%f")[
        :-3
    ]  # Remove microseconds
    et = spice.str2et(omm_epoch_str)
    print(et)

    # Propagate using SPICE
    state, _ = spice.spkgeo(satrec.satnum, et, "J2000", time_delta.total_seconds())
    position = state[:3]
    velocity = state[3:]

    return position, velocity


satellite_names = []
orbits = []
max_count = 10

with open(os.path.join(os.path.dirname(__file__), "starlink.xml")) as xml:
    segments = omm.parse_xml(xml)

    count = 0
    for segment in segments:
        sat = Satrec()
        omm.initialize(sat, segment)

        orbits.append(sat)
        satellite_names.append(segment["OBJECT_NAME"])

        count += 1
        if count >= max_count:
            break

satrec = orbits[0]
# Get the current state of the satellite at its epoch
initial_position, initial_velocity = get_satellite_state_at_epoch(satrec)

print(f"Initial Position at Epoch: {initial_position}")
print(f"Initial Velocity at Epoch: {initial_velocity}")

# Propagate the satellite using SpiceyPy for 10 minutes
propagation_time_delta = timedelta(minutes=10)
propagated_position, propagated_velocity = propagate_satellite_spice(
    satrec, time_delta=propagation_time_delta
)

print(f"Position after 10 minutes using SpiceyPy: {propagated_position}")
print(f"Velocity after 10 minutes using SpiceyPy: {propagated_velocity}")

# Unload SPICE kernels
spice.unload(r"C:\Users\31653\Desktop\ProjectCS\test\SpiceKernels\de438.bsp")
spice.unload(r"C:\Users\31653\Desktop\ProjectCS\test\SpiceKernels\naif0012.tls")
spice.unload(r"C:\Users\31653\Desktop\ProjectCS\test\SpiceKernels\jup344.bsp")
