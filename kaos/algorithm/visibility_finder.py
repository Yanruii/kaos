"""This module contains all functions required to perform the self adating hermite computations."""

import numpy as np

from .interpolator import Interpolator
from .coord_conversion import lla_to_eci

class VisibilityFinder(object):

    """An adaptive visibility finder used to determine the visibility interval of a satellite."""

    def __init__(self, satellite_id, site, interval):
        """Args:
            sattelite_id (integer): Satellite ID in the database
            site (tuple:float): The site location as a lat/lon tupple
            interval (tuple:float): The search window as a start_time, end_time tuple
        """
        self.satellite_id = satellite_id
        self.site = site
        self.interval = interval

        self.sat_irp = Interpolator(satellite_id)

    def visibility(self, posix_time):
        """Calculate the visibility function of the satellite and the site at a given time.

        Args:
            posix_time (float): The time to evaluate the visibility function at

        Returns:
            The value of the visibility function evaluated at the provided time.
        """
        site_pos = lla_to_eci(self.site[0], self.site[1], 0, posix_time)[0]
        site_normal_pos = site_pos/np.linalg.norm(site_pos)
        sat_pos = self.sat_irp.interpolate(posix_time)[0]
        sat_site = np.subtract(sat_pos, site_pos)

        return np.dot(sat_site, site_normal_pos) / np.linalg.norm(sat_site)

    def visibility_first_derivative(self, time):
        """Calculate the derivative of the visibility function of the satellite and the site at a
        given time.

        Args:
            time (float): The UNIX time to evaluate the derivative visibility function at.

        Returns:
            The value of the visibility function evaluated at the provided time.
        """
        sat_pos_vel = self.sat_irp.interpolate(time)
        site_pos_vel = lla_to_eci(self.site[0], self.site[1], 0, time)

        sat_site_pos = np.subtract(sat_pos_vel[0], site_pos_vel[0])
        sat_site_vel = np.subtract(sat_pos_vel[1], site_pos_vel[1])

        site_normal_pos = site_pos_vel[0] / np.linalg.norm(site_pos_vel[0])
        site_normal_vel = site_pos_vel[1] / np.linalg.norm(site_pos_vel[1])

        first_term = ((1 / np.linalg.norm(sat_site_pos)) *
                      (np.dot(sat_site_vel, site_normal_pos) +
                       np.dot(sat_site_pos, site_normal_vel)))

        second_term = ((1/np.linalg.norm(sat_site_pos) ** 3) *
                       np.dot(sat_site_pos, sat_site_vel) *
                       np.dot(sat_site_pos, site_normal_pos))

        return  first_term - second_term


    def bound_time_step_error(self, interval, error):
        # First we compute the maximum of the fourth derivative as per Eq 8 in the referenced
        # paper
        visibility_4_prime_max = self.visibility_fourth_derivative(interval[1], interval)

        # Then we use the error and Eq 9 to calculate the new time_step.
        return pow((16 * error) / (visibility_4_prime_max / 24), 0.25)



    def visibility_fourth_derivative(self, time, sub_interval):
        """Calculate the fourth derivative of the visibility function of the satellite and the site
        at a given time.

        Args:
            time_interval (

        Returns:
            The value of the visibility function evaluated at the provided time.

        Note:
            This function uses the approximation defined in the Rapid Satellite-to-Site Visibility
            paper.
        """
        #pylint: disable=too-many-locals

        start_time, end_time = sub_interval
        interval_length = end_time - start_time
        mid_time = start_time + (interval_length / 2)

        # In order to approximate the fourth order derivative, we need to evaluate both the
        # visibility function and its first derivative at 3 points:
        #   1- The interval start
        #   2- The interval midpoint
        #   3- The interval end
        visibility_start = self.visibility(start_time)
        visibility_mid = self.visibility(mid_time)
        visibility_end = self.visibility(end_time)

        visibility_d_start = self.visibility_first_derivative(start_time)
        visibility_d_mid = self.visibility_first_derivative(mid_time)
        visibility_d_end = self.visibility_first_derivative(end_time)

        # Calculating the a5 and a4 constants used in the approximation
        a5 = (((24 / interval_length ** 5) * (visibility_start - visibility_end)) +
              ((4 / interval_length**4) *
               (visibility_d_start + (4 * visibility_d_mid) + visibility_d_end)))

        # Since a4's computation is complex, it was split into several parts
        a4_first_term = ((4 / interval_length ** 4) *
                         (visibility_start + (4 * visibility_mid) + visibility_end))
        a4_second_term = ((4 / interval_length ** 4) *
                          ((visibility_d_start * ((2 * start_time) + (3 * end_time))) +
                           ((10 * visibility_d_mid) * (start_time + end_time)) +
                           (visibility_d_end * ((3 * start_time) + (2 * end_time)))))
        a4_third_term = ((24 / interval_length ** 5) *
                         ((visibility_start * ((2 * start_time) + (3 * end_time))) -
                          (visibility_end * ((3 * start_time) + (2 * end_time)))))
        a4 = a4_first_term - a4_second_term - a4_third_term

        # Using the above co-efficients we can determine the approximation as per Eq 5 of the cited
        # paper
        return (120 * a5 * time) + (24 * a4)


    def find_roots(self, interval,):
        """TODO: Docstring for find_roots.
        """
        pass

    def determine_visibility(self, error=0.01, tolerance_ratio=0.1, max_iter=1000):
        """Using the self adapting interpolation algorithm described in the cited paper, this
        function returns the subintervals for which the satellites have visibility.

        The accuracy of this function is tuned by changing:
            * error
            * tolerance_ratio
            * max_iter

        The error in each interpolation sub period is defined by an approximate error tolerance.
        This error tolerance is approximate since the algorithm will deem the accuracy sufficient
        when for a give interpolation sub period either:
            * The max number of iterations is exceeded
            * The tolerance ratio is exceeded

        Args:
            error (float, optional): Tolerance value of approximated error. Defaults to 0.01
            tolerance_ratio (float, optional): The tolerance ratio of the interval time step.
                                               Defaults to 0.1
            max_iter (int, optional): The maximum number of iterations per sub interval. Defaults to
                                      1000

        Returns:
            The subintervals over which the site is visible.

        Note:
            This function assumes a viewing angle of 180 degrees
        """
        start_time, end_time = self.interval

        # Initialize the algorithm variables
        subinterval_start = start_time
        # The subinterval_end is set to start the initial loop iteration
        subinterval_end = start_time
        # Defines the length of the initial subinterval (h)
        prev_time_step = 100

        while subinterval_end < end_time:
            new_time_step_1 = prev_time_step
            # Hack loop since python does not support do-while
            iter_num = 0
            while True:
                subinterval_end = subinterval_start + new_time_step_1
                new_time_step_2 = self.bound_time_step_error((subinterval_start, subinterval_end),
                                                             error)
                if (abs(new_time_step_2 - new_time_step_1) / new_time_step_1) <= tolerance_ratio:
                    break

                if (iter_num >= max_iter) and (new_time_step_1 <= new_time_step_2):
                    break

                new_time_step_1 = new_time_step_2
                iter_num += 1

            # At this stage for the current interpolation stage the time step is sufficiently small
            # to keep the error low
            new_time_step = new_time_step_1
            subinterval_end = subinterval_start + new_time_step

            # TODO Approximate V(t) on the current subinterval and resolve its roots
            # TODO Look at C(t) expand and collect terms such that the resulting expression is an
            # array like expression where each index is the coeefcient at a particular power

            # Set the start time and time step for the next interval
            subinterval_start = subinterval_end
            prev_time_step = new_time_step
