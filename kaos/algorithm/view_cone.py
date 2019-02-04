"""Implementation of Viewing cone algorithm"""
import numpy as np
from numpy import cross
from numpy.linalg import norm

import mpmath as mp

from ..constants import SECONDS_PER_DAY, ANGULAR_VELOCITY_EARTH, THETA_NAUGHT
from ..tuples import TimeInterval
from ..errors import ViewConeError


import matplotlib.pyplot as plt


def cart2sp(x, y, z):
    """Converts data from Cartesian coordinates into spherical.

    Args:
        x (scalar): X-component of data.
        y (scalar): Y-component of data.
        z (scalar): Z-component of data.

    Returns:
        Tuple (r, theta, phi) of data in spherical coordinates.
    """
    r = mp.sqrt(x**2+y**2+z**2)
    theta = mp.asin(z/r)
    phi = mp.atan2(y, x)
    return (r, theta, phi)

def reduce_poi(site_eci, sat_pos, sat_vel, q_magnitude, poi, a):
    """Performs a series of viewing cone calculations and shrinks the input POI

    Args:
      site_eci(Vector3D) = site location in ECI at the start of POI
      sat_pos(Vector3D) = position of satellite (at an arbitrary time)
      sat_vel(Vector3D) = velocity of satellite (at the same arbitrary time as sat_pos)
      q_magnitude(int) = maximum orbital radius
      poi(TimeInterval) = period of interest

    Returns:
      list of TimeIntervals that the orbit is inside viewing cone

    Raises:
      ValueError: on unexpected input
      ViewConeError: on inconclusive result from Viewing cone
    """

    for ass in a:
        plt.plot((ass[0],ass[1]),(0,0),'m-')


    if poi.start > poi.end:
        raise ValueError("poi.start is after poi.end")

    # Tracking how much of the POI has been processed
    cur_end = poi.start
    # Estimate of maximum m
    expected_final_m = mp.ceil((poi.end - poi.start)/SECONDS_PER_DAY)+1
    # print(expected_final_m)
    # Find the intervals to cover the input POI
    interval_list = []
    m = 0
    while (cur_end < poi.end) and (m < expected_final_m):
        try:
            t_1, t_2, t_3, t_4 = _view_cone_calc(site_eci, sat_pos, sat_vel, q_magnitude, m, poi.start)
            # Validate the intervals
            # if (t_3 > t_1) or (t_2 > t_4):
            #     # Unexpected order of times
            #     raise ViewConeError("Viewing Cone internal error")
            # Add intervals to the list
            interval_list.append(TimeInterval(poi.start+t_3, poi.start+t_1))
            interval_list.append(TimeInterval(poi.start+t_2, poi.start+t_4))
            m += 1
            cur_end = poi.start + t_4
        except ValueError:
            # The case were the formulas have less than 4 roots
            raise ViewConeError("Unsupported viewing cone and orbit configuration.")

    plt.show()
    # Adjusting the intervals to fit inside the input POI and return
    return _trim_poi_segments(interval_list, poi)

def _trim_poi_segments(interval_list, poi):
    """Semi-private: Adjusts list of intervals so that all intervals fit inside the poi

        Args:
          interval_list(list of TimeIntervals) = the interval to be trimmed
          poi(TimeInterval) = period of interest, reference for trimming

        Returns:
          List of TimeIntervals that fit inside the poi
    """
    ret_list = []
    for interval in interval_list:
        if (interval.start > poi.end) or (interval.end < poi.start):
            # Outside the input POI
            continue
        elif (interval.start < poi.end) and (interval.end > poi.end):
            ret_list.append(TimeInterval(interval.start, poi.end))
        elif (interval.end > poi.start) and (interval.start < poi.start):
            ret_list.append(TimeInterval(poi.start, interval.end))
        else:
            ret_list.append(TimeInterval(interval.start, interval.end))

    return ret_list


def _view_cone_calc(site_eci, sat_pos, sat_vel, q_magnitude, m, interval_start):
    """Semi-private: Performs the viewing cone visibility calculation for the day defined by m.

    Note: This function is based on a paper titled "rapid satellite-to-site visibility determination
    based on self-adaptive interpolation technique"  with some variation to account for interaction
    of viewing cone with the satellite orbit.

    Args:
      site_eci(Vector3D) = site location in ECI at the start of POI
      sat_pos(Vector3D) = position of satellite (at the same time as sat_vel)
      sat_vel(Vector3D) = velocity of satellite (at the same time as sat_pos)
      q_magnitude(int) = maximum orbital radius

    Returns:
      Returns 4 numbers representing times at which the orbit is tangent to the viewing cone,

    Raises:
      ValueError: if any of the 4 formulas has a complex answer. This happens when the orbit and
      viewing cone do not intersect or only intersect twice.
      Note: With more analysis it should be possible to find a correct interval even in the case
      where there are only two intersections but this is beyond the current scope of the project.
    """
    values = []
    times = []
    gamma_line = []
    gamma2_line = []

    THETA_NAUGHT = 0 * (mp.pi/180)
    # Get geocentric angles from site ECI
    site_eci = np.array(site_eci) * mp.mpf(1.0)
    r_site_magnitude, lat_geoc, lon_geoc = cart2sp(site_eci[0], site_eci[1], site_eci[2])

    # P vector (also referred  to as orbital angular momentum in the paper) calculations
    p_unit_x, p_unit_y, p_unit_z = cross(sat_pos, sat_vel) / (mp.norm(sat_pos) * mp.norm(sat_vel))

    # Formulas from paper:
    # Note: each Txxx represents an intersection between viewing cone and the orbit
    gamma = THETA_NAUGHT + mp.asin((r_site_magnitude * mp.sin((mp.pi / 2) + THETA_NAUGHT)) / q_magnitude)
    arcsin_term = lambda g:(mp.asin((mp.cos(g) - p_unit_z * mp.sin(lat_geoc)) /
                  (mp.sqrt((p_unit_x ** 2) + (p_unit_y ** 2)) * mp.cos(lat_geoc))))
    arctan_term = mp.atan2(p_unit_x,p_unit_y)

    time_1 = (1 / ANGULAR_VELOCITY_EARTH) * (arcsin_term(gamma) - lon_geoc - arctan_term + 2 * mp.pi * m)
    time_2 = (1 / ANGULAR_VELOCITY_EARTH) * (mp.pi - arcsin_term(gamma) - lon_geoc - arctan_term + 2 * mp.pi * m)

    # Second set
    gamma2 = mp.pi - gamma
    # time_3 = (1/ANGULAR_VELOCITY_EARTH) * (arcsin_term(gamma2) - lon_geoc - arctan_term + 2 * mp.pi * m)
    # time_4 = (1/ANGULAR_VELOCITY_EARTH) * (mp.pi - arcsin_term(gamma2) - lon_geoc - arctan_term + 2 * mp.pi * m)
    time_4 = 0
    time_3 = 0

    result = lambda time:(p_unit_x * mp.cos(lat_geoc) * mp.cos(ANGULAR_VELOCITY_EARTH*time+lon_geoc)
                        + p_unit_y * mp.cos(lat_geoc) * mp.sin(ANGULAR_VELOCITY_EARTH*time+lon_geoc) 
                        + p_unit_z * mp.sin(lat_geoc))

    # print ("result 1: ", result(time_1))
    # print ("result 2: ", result(time_2))
    # print ("cos(gamma): ", mp.cos(gamma))


    for t in range(interval_start,interval_start+(m+1)*24*60*60,100):
        values.append(result(t))
        gamma_line.append(mp.cos(gamma))
        gamma2_line.append(mp.cos(gamma2))
        times.append(t)

    print("time 1: ",time_1)
    print("time 2: ",time_2)
    print("time 3: ",time_3)
    print("time 4: ",time_4)
    # plt.plot(times, values, "r--", times, gamma_line, "g--", [time_1,time_2], [result(time_1),result(time_2)], "bs")
    plt.plot(times, values, "r--", times, gamma_line, "g--",times, gamma2_line, "k--",
             [time_1+interval_start,time_2+interval_start], 
             [result(time_1+interval_start),
             result(time_2+interval_start)], "bs",
             [time_3+interval_start,time_4+interval_start], 
             [result(time_3+interval_start),
             result(time_4+interval_start)], "bs")

    # Check for complex answers
    # if(filter(lambda time: not isinstance(time, mp.mpf),[time_1,time_2])):
    #     raise ValueError()

    return time_1, time_2, time_3, time_4
