"""This module contains all date/time related conversion code."""

import datetime
import calendar

from astropy.time import Time


def utc_to_unix(time_string, string_format='%Y%m%dT%H:%M:%S.%f'):
    """Takes a string input of a specified date time format and converts it to a UNIX time stamp.

    Args:
        time_string (str): String representation of UTC time in the stated format. Must be
                           greater than 19700101.
        string_format (str): The strptime format of the provided time string. Defaults to
                             'YYYYMMDDTHH:MM:SS.MS'  (where T is the letter 'T')

    Returns:
        The UNIX time stamp representation of the input time string.

    Raises:
        ValueError: If the string is malformed or the date represented by the string is invalid
                    in the POSIX epoch.

    Note:
        This function returns the UNIX time stamps to second precision only.
    """

    date_time = datetime.datetime.strptime(time_string, string_format)
    return calendar.timegm(date_time.utctimetuple())


def jdate_to_utc(jdate):
    """Takes a date in Julian format and converts it to a UTC formatted time stamp.

    Args:
        jdate (float): Float representing the number of days since January 1, 4713 BC.

    Returns:
        A string formatted as 'YYYYMMDDTHH:MM:SS.MS' (where T is the letter 'T') representing
        the UTC timestamp of the Jdate.

    Raises:
        ValueError: If the string is malformed or the date represented by the string is invalid
                    in the JDate format.

    Note:
        This function returns the UTC time stamps to millisecond precision.
    """
    jdate_timestamp = Time(jdate, format='jd')
    iso_date = ''.join(str(s) for s in jdate_timestamp.iso.split()[0].split('-'))
    iso_date += 'T' + str(jdate_timestamp.iso.split()[1])
    return iso_date


def jdate_to_unix(jdate):
    """ Takes a date in Julian format and converts it to a UNIX
    time stamp.

    Raises:
        ValueError: If the string is malformed or the date represented by the string is invalid
                    in the JDate format.
    """
    utc_date = jdate_to_utc(jdate)

    # remove the millisecond precision
    return utc_to_unix(utc_date[:-1])
