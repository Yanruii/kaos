import pytest, unittest

from ddt import ddt,data

from kaos.algorithm import coord_conversion
from kaos.tuples import Vector3D

@ddt
class Test_lla_to_ecef(unittest.TestCase):

    @data((0.0,0.0,0.0,6378137.0,0.0,0.0),
          (49.2827,-123.1207,0.0,-2277772.9,-3491338.7,4811126.5),
          (43.7615,-79.41107,0.0,847846.8,-4535275.1,4388991.1),
          (39.9138,116.3636,0.0,-2175414.8,4389344.3,4070649.0),
          (27.9861,86.9226,8848,303011.566,5636116.783,2979297.096))
    def test_lla_to_ecef(self,test_data):
        """
        Test for lat,lon,height to ECEF conversion with 0.1 meter accuracy
        test_data format:
          input lat, input lon, hight,  expectd x, expected y, expected z
        Values generated using: https://www.ngs.noaa.gov/NCAT/
        """
        ecef = coord_conversion.lla_to_ecef(test_data[0],test_data[1],test_data[2])
        self.assertAlmostEqual(ecef[0], test_data[3], places=1)
        self.assertAlmostEqual(ecef[1], test_data[4], places=1)
        self.assertAlmostEqual(ecef[2], test_data[5], places=1)

@ddt
class Test_geod_to_geoc_lat(unittest.TestCase):

    @data((0.0,0.0),(10.0,9.9344),(25,24.8529),(45.0,44.8076),(-11,-10.9281),
          (-63.55648,-63.4027),(89.99,89.9899),(-89.9799,-89.9798),(53.2576,53.0729),
          (90,90),(-90,-90))
    def test_geod_to_geoc_lat(self,test_data):
        """
        Test for latitude conversion from geodetic to geocentric with 0.0001 accuracy
        test_data format:
          latitude(geodetic-WGS84), expected latitude(geocentric)
        Values generated using Matlab: geod2geoc(lat,0,'WGS84')
        """
        geoc_lat_deg = coord_conversion.geod_to_geoc_lat(test_data[0])
        self.assertAlmostEqual(geoc_lat_deg,test_data[1],places=4)

@ddt
class Test_lla_to_eci(unittest.TestCase):

    @data((0,0,0,946684800,(-1.1040230676e+6,6.28185996625e+6,145.726867)),
        (10,120,0,946684800,(-4.81449119049e+6,-4.0352367821e+6,1.100005365e+6)),
        (-25,80,0,946684800,(-5.78394065413e+6,3.324200655e+3,-2.6792309935e+6)),
        (-70.54,-12.53,0,946684800,(9.5436e+4,2.1293e+6,-5.9913e+6)),
        (60.21,-80.46,0,946684800,(2.9943e+6,1.0607e+6,5.5122e+6)))
    def test_lla_to_eci(self,test_data):
        """
        Test for lat,lon,alt conversion to GCRS
        Note the 200m delta, see function docstring for details.

        Test_data format:
          lat,lon,alt,time_posix,(GCRS expected x,GCRS expected y,GCRS expected z)
        Values generated using Matlab: lla2eci([lat,lon,alt],time) which is in a J2000 FK5 frame
        """
        loc_eci = coord_conversion.lla_to_eci(test_data[0], test_data[1], test_data[2],
                                              test_data[3])[0]
        self.assertAlmostEqual(loc_eci[0], test_data[4][0], delta=200)
        self.assertAlmostEqual(loc_eci[1], test_data[4][1], delta=200)
        self.assertAlmostEqual(loc_eci[2], test_data[4][2], delta=200)

@ddt
class Test_eccf_to_eci(unittest.TestCase):

    @data((1514764800,
           (-1.1923013839603376e+05, 7.1372890010702536e+06, 6.9552517228703119e+05),
           (1.6248481050346918e+03, 7.5140287637908330e+02, -7.3372220261131497e+03),
           (-6.9980497691646582e+06, -1.4019786400312854e+06, 7.0754554424135364e+05),
           (-9.4202033738527109e+02, 9.5296010534027573e+02, -7.3355694593015414e+03)),
          (1514766900,
           (3.9804738738891535e+05, -3.6047780125065618e+06, -6.1941370541235292e+06),
           (-1.7738930891952052e+03, -6.3698768084068670e+03, 3.5949496617909012e+03),
           (3.2642112748636086e+06, 1.5584081196010089e+06, -6.1997168462066837e+06),
           (6.4919176217824288e+03, 6.5176028122607818e+02, 3.5837784721647768e+03)))
    def test_lla_to_eci(self, test_data):
        """
        Test for ECEF conversion to GCRS
        Note the 1m delta, see function docstring for details.

        test_data format:
          posix_time, ecef pos, ecef vel, expected eci pos, expected eci vel

        Values generated using stk.
        """
        time, test_pos, test_vel, real_pos, real_vel = test_data
        pos_vel_pair = coord_conversion.ecef_to_eci(test_pos, test_vel, time)
        self.assertEqual(real_pos, pytest.approx(pos_vel_pair[0][0], 1))
        self.assertEqual(real_vel, pytest.approx(pos_vel_pair[0][1], 1))
