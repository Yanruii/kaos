"""Testing the visibility_finder."""
from collections import namedtuple
import re

from ddt import ddt, data

from kaos.algorithm.visibility_finder import VisibilityFinder
from kaos.models import Satellite
from kaos.models.parser import parse_ephemeris_file
from kaos.utils.time_conversion import utc_to_unix
import mpmath as mp

from .. import KaosTestCase

AccessTestInfo = namedtuple('AccessTestInfo', 'sat_name, target, accesses')


@ddt
class TestVisibilityFinder(KaosTestCase):
    """Test the visibility finder using specialized test files. These files are generated from STK
    and modified to include all relevant data about actual expected access times.
    """

    @classmethod
    def setUpClass(cls):
        super(TestVisibilityFinder, cls).setUpClass()
        parse_ephemeris_file("ephemeris/Radarsat2.e")

    # pylint: disable=line-too-long
    @staticmethod
    def parse_access_file(file_path):
        """Reads a KAOS access test file, these files follow the following format:

            ====================================================================================================
            Satellite Name: <Sat Name>
            Target Point: <lon>, <lat>
            ====================================================================================================
            record number, access start, access_end, access_duration
            ....

        Args:
            file_path (string): The path of the KAOS access test file.

        Returns:
            An AccessTestInfo tuple.
        """
        with open(file_path) as access_file:
            access_info_text = access_file.read()

        section_regex = re.compile(r'={99}', re.MULTILINE)
        access_info = section_regex.split(access_info_text)

        # Parse the header
        sat_name = re.search(r'Satellite Name: ([a-zA-Z0-9]+)', access_info[1]).groups()[0]
        target = [float(point) for point in
                  re.search(r'Target Point: (.*)', access_info[1]).groups()[0].split(',')]
        # Parse the access times
        accesses = []
        raw_access_data = access_info[2].splitlines()
        for access in raw_access_data[1:]:
            access = access.split(',')
            # Parse the start and end time
            start_time = utc_to_unix(access[1].rstrip().lstrip(), '%d %b %Y %H:%M:%S.%f')
            end_time = utc_to_unix(access[2].rstrip().lstrip(), '%d %b %Y %H:%M:%S.%f')
            accesses.append((start_time, end_time))

        return AccessTestInfo(sat_name, target, accesses)
    # pylint: enable=line-too-long

    @data(('test/algorithm/vancouver.test', (1514764802, 1514772000), 60),
          ('test/algorithm/vancouver.test', (1514768543, 1514772000), 60),
          ('test/algorithm/vancouver.test', (1514768340, 1514768400), 60),
          ('test/algorithm/vancouver.test', (1514768543, 1514769143), 60),
          ('test/algorithm/vancouver.test', (1515160800, 1515164400), 60))
    def test_full_visibility(self, test_data):
        """Tests that the visibility finder produces the same results as the access file.

        Args:
            test_data (tuple): A three tuple containing the:
                                1 - The path of KAOS access test file
                                2 - A tuple of the desired test duration
                                3 - The maximum tolerated deviation in seconds
        Note:
            Ranges provided must not start or end at an access boundary. This is a result of the
            strict checking method provided.
        """
        access_file, interval, max_error = test_data

        access_info = self.parse_access_file(access_file)
        finder = VisibilityFinder(Satellite.get_by_name(access_info.sat_name)[0].platform_id,
                                  access_info.target, interval)
        access_times = finder.determine_visibility()

        for predicted_access in access_times:
            found = False
            for actual_access in access_info.accesses:
                if (interval[0] > actual_access[0]) and (interval[1] < actual_access[1]):
                    if ((abs(interval[0] - predicted_access[0]) < max_error) and
                            (abs(interval[1] - predicted_access[1]) < max_error)):
                        found = True
                        break
                if interval[0] > actual_access[0]:
                    if ((abs(interval[0] - predicted_access[0]) < max_error) and
                            (abs(actual_access[1] - predicted_access[1]) < max_error)):
                        found = True
                        break
                elif interval[1] < actual_access[1]:
                    if ((abs(actual_access[0] - predicted_access[0]) < max_error) and
                            (abs(interval[1] - predicted_access[1]) < max_error)):
                        found = True
                        break

                if ((abs(actual_access[0] - predicted_access[0]) < max_error) and
                        (abs(actual_access[1] - predicted_access[1]) < max_error)):
                    found = True
                    break

            if not found:
                raise Exception('Wrong access: {}'.format(predicted_access))

    @data(('test/algorithm/vancouver.test', (1514764802, 1514772000), 60),
          ('test/algorithm/vancouver.test', (1514768543, 1514772000), 60),
          ('test/algorithm/vancouver.test', (1514768340, 1514768400), 60),
          ('test/algorithm/vancouver.test', (1514768543, 1514769143), 60),
          ('test/algorithm/vancouver.test', (1515160800, 1515164400), 60))
    def test_brute_force(self, test_data, time_step=60):
        """Tests that the visibility finder produces the same results as the access file.
           This is a brute force method. The self-adapting part is dropped. The root finding part
           is dropped. This method checks the intersection between the Visibility curve and
           sin(minimum_elevation_angle) and updates the visibility intervals.

        Args:
            test_data (tuple): A three tuple containing the:
                                1 - The path of KAOS access test file
                                2 - A tuple of the desired test duration
                                3 - The maximum tolerated deviation in seconds
            time_step (int)

        Note:
            Ranges provided must not start or end at an access boundary. This is a result
            of the strict checking method provided.
        """

        mp.dps = 100
        access_file, interval, max_error = test_data

        access_info = self.parse_access_file(access_file)
        finder = VisibilityFinder(Satellite.get_by_name(access_info.sat_name)[0].platform_id,
                                  access_info.target, interval)

        access_times = []
        subinterval_start = interval[0]
        subinterval_end = subinterval_start + time_step

        if mp.mpf(finder.visibility(subinterval_start)) > 0:
            visibility_last_sign = 1
            start_time = interval[0]
        else:
            visibility_last_sign = 0

        while subinterval_end < interval[1]:
            visibility = mp.mpf(finder.visibility(subinterval_start))
            # going from invisible to visible
            if visibility > 0 and visibility_last_sign == 0:
                start_time = subinterval_start
            # going from visible to invisible
            if visibility < 0 and visibility_last_sign == 1:
                access_times.append((start_time, subinterval_start))
            # ending the checking
            if (subinterval_end + time_step) > interval[1]:
                if not access_times:
                    access_times.append((start_time, interval[1]))
                elif start_time != access_times[len(access_times)-1][0]:
                    access_times.append((start_time, interval[1]))
            if visibility > 0:
                visibility_last_sign = 1
            if visibility < 0:
                visibility_last_sign = 0
            subinterval_start = subinterval_end
            subinterval_end += time_step
        #deal with the case where the poi is less than the step time:
        if len(access_times) == 0 and finder.visibility(subinterval_start) > 0:
            access_times.append((interval[0], interval[1]))

        for predicted_access in access_times:
            found = False
            for actual_access in access_info.accesses:
                if (interval[0] > actual_access[0]) and (interval[1] < actual_access[1]):
                    if ((abs(interval[0] - predicted_access[0]) < max_error) and
                            (abs(interval[1] - predicted_access[1]) < max_error)):
                        found = True
                        break
                if interval[0] > actual_access[0]:
                    if ((abs(interval[0] - predicted_access[0]) < max_error) and
                            (abs(actual_access[1] - predicted_access[1]) < max_error)):
                        found = True
                        break
                elif interval[1] < actual_access[1]:
                    if ((abs(actual_access[0] - predicted_access[0]) < max_error) and
                            (abs(interval[1] - predicted_access[1]) < max_error)):
                        found = True
                        break

                if ((abs(actual_access[0] - predicted_access[0]) < max_error) and
                        (abs(actual_access[1] - predicted_access[1]) < max_error)):
                    found = True
                    break

            if not found:
                raise Exception('Wrong access: {}'.format(predicted_access))