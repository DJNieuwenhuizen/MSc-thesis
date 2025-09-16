# *************************************************************************************************
#
#  main_MEX-TGO_measurement_planning.py
#
# *************************************************************************************************
#  
#  Description: 
#  |  This model is used to determine MEX-TGO UHF BSR measurement opportunities in the near-
#     equatorial region of Mars (for shallow subsurface water (ice) research).
#     |  As input it takes a UTC time range and it outputs a folder including:
#               - a 'run.log' file with the script's logging information
#               - a 'midpoints.txt' file with the names of the midpoint data files
#               - a 'meas_ops.txt' file with the names of the measurement opportunity data files
#               - a .csv file describing the midpoints and measurement opportunity data for each
#                 individual opportunity
#               - three .png plots of the ground tracks for measurement opportunities, midpoints
#                 and both combined
#
# **********************************************************************************************
#
#  Author information:
#  |  Dominique Julianne Nieuwenhuizen
#  |  Delft University of Technology; Faculty of Aerospace Engineering
#
#  MSc thesis: 
#  | 'Probing shallow subsurface water on Mars through bi-static radar measurements
#  |  at UHF wavelengths'
#
#  Available at: 
#  |  https://github.com/DJNieuwenhuizen/MSc-thesis
#  |  https://resolver.tudelft.nl/uuid:fb735003-5c8a-4c37-963f-7546ca0358e8
#
# **********************************************************************************************

""" Import Python packages """

import glob
import logging
import numpy as np
import os
from pathlib import Path
import spiceypy as spice
import time
from tqdm import tqdm

from my_functions import plot_spoints
from my_parameters import Mars_param

# *************************************************************************************************

""" Set UTC timeframe over which to look for measurement opportunities """

# Format 'YYYY MON DD HH:MM:SS'
utc0_single = '2025 JUL 17 00:00:00'
utcf_single = '2025 JUL 17 23:59:59'

# months_jul_aug = [['2025 JUL 01 00:00:00', '2025 AUG 01 23:59:59'],
#                   ['2025 AUG 01 00:00:00', '2025 SEP 01 23:59:59']]

time_frames = [[utc0_single, utcf_single]]
# time_frames = months_jul_aug.copy()

MTP_selections = ['MTP276/95']
BSR_selections = ['3']

# *************************************************************************************************

""" Load SPICE kernels """

MetaKernelsPath = os.path.join(os.path.dirname(os.getcwd()),
                               "thesis\code_and_simulations\kernels\mk")

main_gen_mk = os.path.join(MetaKernelsPath, 'gen_mk.tm') # General Solar System/Time/Mars kernels
main_MEX_mk = os.path.join(MetaKernelsPath, 'MEX_mk.tm') # MEX-specific kernels
main_TGO_mk = os.path.join(MetaKernelsPath, 'TGO_mk.tm') # TGO-specific kernels

spice.furnsh(main_gen_mk)
spice.furnsh(main_MEX_mk)
spice.furnsh(main_TGO_mk)

# *************************************************************************************************

""" Measurement limitations/requirements """

# Maximum MEX-TGO distance [km]
max_distance_lowestpoint = 1500  # At lowest point
max_distance_lp_margin = 2000 # Margin on lowest point vs. measurement duration
max_distance = max_distance_lowestpoint + max_distance_lp_margin # Over measurement duration

# Minimum measurement duration [minutes]
meas_duration = 10

# Latitude bound for near-equatorial region of interest [degrees]
lat_bound = 40 # True bound for the converged equiemission spoints
lat_bound_margin = 5 # Margin for the midpoint estimates
lat_bound_midpoint = lat_bound + lat_bound_margin #  Simulated bound for the midpoint estimates
lat_bound_times = 70 # Simulated bound for singular times for the converged equiemission spoints
lat_bound_times_midpoint = lat_bound_times + lat_bound_margin # Simulated bound for singular times 
                                                              # for the midpoint estimates

# Maximum equiemission angle (to minimise scattering effects due to surface roughness) [degrees]
emission_max_lowestpoint = 40 # At the lowest ("ideal") point
emission_max_lp_margin = 20 # Margin on lowest point vs. measurement duration
emission_max = emission_max_lowestpoint + emission_max_lp_margin # Over measurement duration for
                                                                 # the converged equiemission
                                                                 # spoints
midpoint_emission_margin = 5 # Margin for the midpoint estimates
midpoint_emission_max = emission_max + midpoint_emission_margin # Over measurement duration for
                                                                # the midpoint estimates

# Parameters for the equiemission spoint convergence algorithm
time_step_int = 10 # seconds; time step interval (ET times)
equiemission_lat_errorbound = np.deg2rad(1e-2) # radians; allowed deviation of the spoint
                                               # normal vector from the MEX-spoint-TGO plane
equiemission_diff_errorbound = 1e-2 # degrees; allowed emission angle difference of MEX and TGO
n_iterations_max = 1000 # Maximum number of iterations

# *************************************************************************************************

""" Mars geometry """

radii_ell = Mars_param['radii_ell'] # km; ellipsoidal Mars
radii_sph = Mars_param['radii_sph'] # km; spherical Mars

# *************************************************************************************************

""" Functions """

def log_print(*args):
    """ 
    log_print: This is a custom utility function to simultaneously print and log statements to the 
    'run.log' file saved in the directory of the inputted UTC time range.
    """
    message = " ".join(map(str, args))
    print(message) 
    logging.info(message) 
    return

def check_lineofsight(MEX_pos, TGO_pos):
    """
    check_lineofsight: This function checks whether there is a direct line of sight between the 
    Mars Express (MEX) and Trace Gas Orbiter (TGO) spacecrafts, effectively determining whether or
    not Mars is located between them.

    :param MEX_pos: Cartesian vector from Mars to MEX in the IAU_MARS frame [km], determined
        using the spice.spkpos function.
    :type MEX_pos: class 'numpy.ndarray'
    :param TGO_pos:  Cartesian vector from Mars to TGO in the IAU_MARS frame [km], determined
        using the spice.spkpos function.
    :type TGO_pos: class 'numpy.ndarray'

    :return: The 'intersect_found' parameter indicates whether or not there was an intersect found
        on the line of sight between MEX and TGO. If True, then there is no line of sight between
        MEX and TGO.
    :rtype: class 'bool'
    """

    lineofsight = TGO_pos - MEX_pos
    lineofsight_norm = spice.vnorm(lineofsight)
    lineofsight_unit = lineofsight / lineofsight_norm

    try:
        intersect_point = spice.surfpt(MEX_pos, lineofsight_unit, radii_ell[0], radii_ell[1],
                                       radii_ell[2])[0]
        MEX2intersect = intersect_point - MEX_pos
        dis_MEX2intersect = spice.vnorm(MEX2intersect)
        if dis_MEX2intersect > lineofsight_norm:
            intersect_found = False
        else:
            intersect_found = True
    except spice.utils.exceptions.SpiceyError as e:
        intersect_found = False

    return intersect_found

def find_midpoint(et):
    """
    find_midpoint: This function calculates the midpoint on the line of sight between MEX and TGO, 
    at a specific Ephemeris Time, and projects this point onto the surface of ellipsoidal Mars. 

    :param et: Ephemeris Time seconds past J2000
    :type et: class 'float'

    :return: utc: Coordinated Universal Time string in the format 'YYYY MMM DD HH:MM:SS'
    :rtype: utc: class 'str'
    :return: dis_MEX2TGO: Distance between MEX and TGO [km]
    :rtype: dis_MEX2TGO: class 'float'
    :return: midpoint: Cartesian midpoint surface coordinates [km]
    :rtype: midpoint: class 'numpy.ndarray'
    :return: lon_midpoint: Longitude of the midpoint [radians]
    :rtype: lon_midpoint: class 'float'
    :return: lat_midpoint: Latitude of the midpoint [radians]
    :rtype: lat_midpoint: class 'float'
    :return: r_midpoint2MEX: Cartesian vector from the midpoint to MEX in the IAU_MARS frame [km]
    :rtype: r_midpoint2MEX: class 'numpy.ndarray'
    :return: r_midpoint2TGO: Cartesian vector from the midpoint to TGO in the IAU_MARS frame [km]
    :rtype: r_midpoint2TGO: class 'numpy.ndarray'
    :return: r_MEX2TGO: Cartesian vector from MEX to TGO in the IAU_MARS frame [km]
    :rtype: r_MEX2TGO: class 'numpy.ndarray'
    :return: r_Mars2TGO: Cartesian vector from Mars to TGO in the IAU_MARS frame [km]
    :rtype: r_Mars2TGO: class 'numpy.ndarray'
    :return: r_Mars2MEX: Cartesian vector from Mars to MEX in the IAU_MARS frame [km]
    :rtype: r_Mars2MEX: class 'numpy.ndarray'
    :return: r_Mars2mid: Cartesian vector from Mars to the midpoint in the IAU_MARS frame [km]
    :rtype: r_Mars2mid: class 'numpy.ndarray'
    """
    utc = spice.et2utc(et, 'C', 0)
    
    r_Mars2TGO = spice.spkpos('TGO', et, 'IAU_MARS', 'LT', 'MARS')[0]
    r_Mars2MEX = spice.spkpos('MEX', et, 'IAU_MARS', 'LT', 'MARS')[0]
    r_MEX2TGO = spice.spkpos('TGO', et, 'IAU_MARS', 'LT', 'MARS')[0]
    dis_MEX2TGO = spice.vnorm(r_MEX2TGO)

    r_Mars2mid = (r_Mars2TGO+r_Mars2MEX)/2 
    lon_midpoint, lat_midpoint = spice.reclat(r_Mars2mid)[1:] 

    midpoint = spice.latsrf('ELLIPSOID', 'MARS', et, 'IAU_MARS', [[lon_midpoint, lat_midpoint]])[0]
    
    r_midpoint2TGO = r_Mars2TGO - midpoint
    r_midpoint2MEX = r_Mars2MEX - midpoint      

    return utc, dis_MEX2TGO, midpoint, lon_midpoint, lat_midpoint, r_midpoint2MEX, \
           r_midpoint2TGO, r_MEX2TGO, r_Mars2TGO, r_Mars2MEX, r_Mars2mid

def find_equiemission(et):
    """
    find_equiemission: This function calculates the point on the surface of elliptical Mars for
    which the emission angle is the same for both MEX and TGO. It was based strongly on the 
    'compute_equiemission_point.py' code by Alfredo Escalante (European Space Agency).

    :param et: Ephemeris Time seconds past J2000
    :type et: class 'float'

    :return: utc: Coordinated Universal Time string in the format 'YYYY MMM DD HH:MM:SS'
    :rtype: utc: class 'str'
    :return: dis_MEX2TGO: Distance between MEX and TGO [km]
    :rtype: dis_MEX2TGO: class 'float'
    :return: spoint: Cartesian spoint surface coordinates [km]
    :rtyp spoint: class 'numpy.ndarray'
    :return: r_spoint2TGO: Cartesian vector from the spoint to TGO in the IAU_MARS frame [km]
    :rtype: r_spoint2TGO: class 'numpy.ndarray'
    :return: r_spoint2MEX: Cartesian vector from the spoint to MEX in the IAU_MARS frame [km]
    :rtype: r_spoint2MEX: class 'numpy.ndarray'
    :return: lon: Longitude of the spoint [radians]
    :rtype: lon: class 'float'
    :return: lat: Latitude of the spoint [radians]
    :rtype: lat: class 'float'
    :return: lat_sph: Latitude of the spoint on spherical Mars [radians]
    :rtype: lat_sph: class 'float'
    :return: emission_MEX: Emission angle between the spoint and MEX [degrees]
    :rtype: emission_MEX: class 'float'
    :return: emission_TGO: Emission angle between the spoint and MEX [degrees]
    :rtype: emission_TGO: class 'float'
    :return: lon_midpoint: Longitude of the midpoint [radians]
    :rtype: lon_midpoint: class 'float'
    :return: lat_midpoint: Latitude of the midpoint [radians]
    :rtype: lat_midpoint: class 'float'
    """

    convergence = False
    utc = spice.et2utc(et, 'C', 0)

    r_MEX2TGO = spice.spkpos('TGO', et, 'IAU_MARS', 'LT', 'MEX')[0]
    r_Mars2TGO = spice.spkpos('TGO', et, 'IAU_MARS', 'LT', 'MARS')[0]
    r_Mars2MEX = spice.spkpos('MEX', et, 'IAU_MARS', 'LT', 'MARS')[0]
    dis_MEX2TGO = spice.vnorm(r_MEX2TGO)

    r_Mars2mid = (r_Mars2TGO+r_Mars2MEX)/2
    lon_midpoint_org, lat_midpoint_org = spice.reclat(r_Mars2mid)[1:]
    
    n_conv_iterations = 0

    while convergence == False and n_conv_iterations < n_iterations_max:
        n_conv_iterations += 1
        emission_step = 100

        if n_conv_iterations >= n_iterations_max/2:
            emission_step = 10
            
        lon_midpoint, lat_midpoint = spice.reclat(r_Mars2mid)[1:]
        lon = lon_midpoint
        lat = lat_midpoint

        spoint = spice.latsrf('ELLIPSOID', 'MARS', et, 'IAU_MARS', [[lon, lat]])[0]

        spoint_normal_ell = spice.surfnm(radii_ell[0], radii_ell[1], radii_ell[2], spoint)
        spoint_normal_sph = spice.surfnm(radii_sph[0], radii_sph[1], radii_sph[2], spoint)

        spoint_normal = spoint_normal_ell # Initial guess for spoint_normal taken as the normal
                                          # vector of perfect ellipsoid Mars

        # The resultant vector of a cross product is perpendicular to the two input vectors
        # (thus the plane containing them). Hence the angular separation between two cross 
        # products describes the error in the common vector.
        #
        # If the normal vector is contained in the MEX-spoint-TGO plane, this error is 0
        e_error_rad = spice.vsep(np.cross(r_Mars2TGO, spoint_normal), 
                                 np.cross(spoint_normal, r_Mars2MEX))
        e_error = spice.convrt(e_error_rad, 'RADIANS', 'DEGREES')

        lat += np.arctan2(spice.vnorm(np.cross(spoint_normal_ell, spoint_normal_sph)),
                          np.dot(spoint_normal_ell, spoint_normal_sph))
        lat_sph = lat

        n_iterations = 0
        lat_step = 10
        e_error_rad_list = []
        e_error_rad_list.append(e_error_rad)
        lat_list = []
        lat_list.append(lat)

        while e_error_rad > equiemission_lat_errorbound and n_iterations < n_iterations_max:
            n_iterations += 1        
            
            spoint = spice.latsrf('ELLIPSOID', 'MARS', et, 'IAU_MARS', [[lon, lat]])[0]
            spoint_normal_ell = spice.surfnm(3396.19, 3396.19, 3376.20, spoint)
            
            r_spoint2TGO = r_Mars2TGO - spoint
            r_spoint2MEX = r_Mars2MEX - spoint

            v1 = np.cross(r_spoint2TGO, spoint_normal_ell)
            v2 = np.cross(spoint_normal_ell, r_spoint2MEX)
            
            e_error_rad = np.arctan2(spice.vnorm(np.cross(v1, v2)), np.dot(v1, v2))
            e_error_rad_list.append(e_error_rad)
            
            if e_error_rad_list[n_iterations] > e_error_rad_list[n_iterations-1]:
                lat = lat_list[n_iterations-2]
                lat_step = 1
            else:
                lat -= (lat_step*e_error_rad) / 360
            
            lat_list.append(lat)

        # Correct the surface point along the MEX-TGO line based on emission angle difference
        #     
        emission_MEX_rad = spice.illumf('ELLIPSOID', 'MARS', 'SUN', et, 'IAU_MARS', 'LT+S', 
                                        'MEX', spoint)[4]
        emission_MEX = spice.convrt(emission_MEX_rad, 'RADIANS', 'DEGREES')
        emission_TGO_rad = spice.illumf('ELLIPSOID', 'MARS', 'SUN', et, 'IAU_MARS', 'LT+S', 
                                        'TGO', spoint)[4]
        emission_TGO = spice.convrt(emission_TGO_rad, 'RADIANS', 'DEGREES')
        emission_diff = np.abs(emission_MEX - emission_TGO)
        emission_diff_rad = np.abs(emission_MEX_rad - emission_TGO_rad)

        if emission_diff < equiemission_diff_errorbound:
            convergence = True
            
        elif emission_TGO > emission_MEX:
            r_Mars2mid += r_MEX2TGO / spice.vnorm(r_MEX2TGO) * emission_diff_rad * emission_step

        elif emission_TGO < emission_MEX:   
            r_Mars2mid -= r_MEX2TGO / spice.vnorm(r_MEX2TGO) * emission_diff_rad * emission_step

    r_spoint2TGO = r_Mars2TGO - spoint
    r_spoint2MEX = r_Mars2MEX - spoint

    return utc, dis_MEX2TGO, spoint, r_spoint2TGO, r_spoint2MEX, lon, lat, lat_sph, \
           emission_MEX, emission_TGO, lon_midpoint_org, lat_midpoint_org

def perc_removed(len0, lenf):
    """
    perc_removed: This function calculates the percentage of times / measurement opportunities
    that were removed during a certain check.

    :param len0: Original length of a list of times/measurement opportunities
    :type len0: class 'int'
    :param lenf: Final length of a list of times/measurement opportunities
    :type lenf: class 'int'
    :return: The 'perc_removed' parameter indicates the percentage of times / measurement
        opportunities removed by a certain check
    :rtype: class 'float'
    """

    if len0 != 0:
        perc_removed = ((len0 - lenf)/len0)*100
    else:
        perc_removed = 100

    return perc_removed

def remove_times(i, counter, et_values_reduced):
    """
    remove_times This function removes times from a list which do not meet a certain condition

    :param i: Index of the to be removed time within the original list
    :type i: class 'int'
    :param counter: Counter tracking the amount of times already removed from the list
    :type counter: class 'int'
    :param et_values_reduced: List of times, with already some removed, from which another will 
        be removed
    :type et_values_reduced: class numpy.ndarray

    :return: counter: Updated counter tracking the amount of times already removed from the list
    :rtype: class 'int'
    :return: et_values_reduced: Updated list of times, with another time removed
    :rtype: class 'numpy.ndarray'
    """
    
    i_adap = i - counter
    et_values_reduced = np.delete(et_values_reduced, i_adap)
    counter += 1
    
    return counter, et_values_reduced

def remove_meas_ops(i, counter, meas_ops_reduced):
    """
    remove_meas_ops This function removes measurement opportunities from a list which do not meet
        a certain condition

    :param i: Index of the to be removed measurement opportunity within the original list
    :type i: class 'int'
    :param counter: Counter tracking the amount of measurement opportunities already removed from
        the list
    :type counter: class 'int'
    :param meas_ops_reduced: List of measurement opportunties, with already some removed, from
        which another will be removed
    :type meas_ops_reduced: class numpy.ndarray

    :return: counter: Updated counter tracking the amount of measurement opportunities already
        removed from the list
    :rtype: class 'int'
    :return: meas_ops_reduced: Updated list of measurement opportunities, with another one removed
    :rtype: class 'numpy.ndarray'
    """

    i_adap = i - counter
    meas_ops_reduced.pop(i_adap)
    counter += 1
    
    return counter, meas_ops_reduced

def remove_equi_and_midpoint(i, counter, equiemission_data_reduced, midpoints_reduced):
    """
    remove_equi_and_midpoint This function removes measurement opportunities from a list of
        equiemission data and one of midpoint data which do not meet a certain condition

    :param i: Index of the to be removed measurement opportunity within the original lists
    :type i: class 'int'
    :param counter: Counter tracking the amount of measurement opportunities already removed from
        the lists
    :type counter: class 'int'
    :param equiemission_data_reduced: List of equiemission data for measurement opportunties, with
        already some removed, from which another will be removed
    :type equiemission_data_reduced: class numpy.ndarray
    :param midpoints_reduced: List of midpoint data for measurement opportunties, with already some
        removed, from which another will be removed
    :type midpoints_reduced: class numpy.ndarray
    :return: counter: Updated counter tracking the amount of measurement opportunities already
        removed from the lists
    :rtype: class 'int'
    :return: equiemission_data_reduced: Updated list of equiemission data for measurement
        opportunities, with another one removed
    :rtype: class 'numpy.ndarray'
    :return: midpoints_reduced: Updated list of midpoint data for measurement opportunities, with
        another one removed
    :rtype: class 'numpy.ndarray'
    """
    
    i_adap = i - counter
    equiemission_data_reduced.pop(i_adap)
    midpoints_reduced.pop(i_adap)
    counter += 1

    return counter, equiemission_data_reduced, midpoints_reduced

def remove_equi_and_midpoint_times(j, counter, equiemission_data_i_reduced, midpoints_i_reduced):
    """
    remove_equi_and_midpoint_times This function removes times from measurement opportunity i from
        a list of equiemission data and one of midpoint data which do not meet a certain condition

    :param j: Index of the to be removed time within the original lists
    :type j: class 'int'
    :param counter: Counter tracking the amount of times already removed from the lists
    :type counter: class 'int'
    :param equiemission_data_i_reduced: List of equiemission data for measurement opportunty i,
        with already some times removed, from which another will be removed
    :type equiemission_data_i_reduced: class numpy.ndarray
    :param midpoints_i_reduced: List of midpoint data for measurement opportunty i, with already
        some times removed, from which another will be removed
    :type midpoints_i_reduced: class numpy.ndarray

    :return: counter: Updated counter tracking the amount of times already removed from the lists
    :rtype: class 'int'
    :return: equiemission_data_i_reduced: Updated list of equiemission data for measurement
        opportunity i, with another time removed
    :rtype: class 'numpy.ndarray'
    :return: midpoints_i_reduced: Updated list of midpoint data for measurement opportunity i,
        with another time removed
    :rtype: class 'numpy.ndarray'
    """
    
    j_adap = j - counter
    equiemission_data_i_reduced.pop(j_adap)
    midpoints_i_reduced = np.delete(midpoints_i_reduced, j_adap)
    counter += 1
    
    return counter, equiemission_data_i_reduced, midpoints_i_reduced

# ************************************************************************************************

""" Main script/loop """

for [utc0, utcf] in time_frames:
    MTP_i = MTP_selections[time_frames.index([utc0, utcf])]
    BSR_i = BSR_selections[time_frames.index([utc0, utcf])]

    """ Directory, file and logging setup """
    directory_name = utc0.replace(" ", ".").replace(":", ".") + '-' \
                     + utcf.replace(" ", ".").replace(":", ".")
    data_directory_path = os.path.join(os.path.join(os.path.dirname(os.getcwd()),
                                                    "thesis\code_and_simulations\data\meas_ops\planning"),
                                                    directory_name)
    
    data_directory = Path(data_directory_path)
    data_directory.mkdir(parents=True, exist_ok=True)

    old_files = glob.glob(data_directory_path + '/*.csv') + \
                glob.glob(data_directory_path + '/*.txt') + \
                glob.glob(data_directory_path + '/*.png')
    for old_file in old_files:
        # Delete data files from previous runs
        os.remove(old_file)

    log_file_name = "run.log"
    log_file_path = os.path.join(data_directory_path, log_file_name)
   
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format="%(asctime)s -" + \
                        "%(levelname)s - %(message)s")
    logger = logging.getLogger()
    logger.handlers[0].stream.write("\n")
    
    # ********************************************************************************************

    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("")
    log_print("Script started at:", start_time_str)

    log_print("")
    log_print("Starting measurement opportunity search from: '" + utc0 + "' to '" + utcf + "'")

    # ********************************************************************************************

    et0 = spice.utc2et(utc0)
    etf = spice.utc2et(utcf)

    et_interval = int((etf - et0) / time_step_int)
    time_step_float = (etf - et0)/et_interval
    et_values = np.linspace(et0, etf, et_interval)

    # ********************************************************************************************

    log_print("")
    log_print("Check 1.1 [START]: Remove times with no line of sight between MEX and TGO")

    et_values_reduced = et_values.copy()
    len_et_values0 = len(et_values)
    counter = 0

    for i in tqdm(range(len(et_values))):
        et = et_values[i]
        r_Mars2TGO, lt_Mars2TGO = spice.spkpos('TGO', et, 'IAU_MARS', 'LT', 'MARS')
        r_Mars2MEX, lt_Mars2MEX = spice.spkpos('MEX', et, 'IAU_MARS', 'LT', 'MARS')
        
        intersect_lineofsight = check_lineofsight(r_Mars2MEX, r_Mars2TGO)
        if intersect_lineofsight:
            counter, et_values_reduced = remove_times(i, counter, et_values_reduced)
            
    et_values = et_values_reduced.copy()
    len_et_valuesf = len(et_values)

    log_print("Percentage of ET times removed with no line of sight between MEX and TGO:",
                round(perc_removed(len_et_values0, len_et_valuesf),1), "%")
    log_print("Check 1.1 [END]: Remove times with no line of sight between MEX and TGO")

    # ********************************************************************************************

    log_print("")
    log_print("Check 1.2 [START]: Remove times with MEX-TGO distance over", max_distance, "km")

    et_values_reduced = et_values.copy()
    len_et_values0 = len(et_values)
    counter = 0

    for i in tqdm(range(len(et_values))):
        et = et_values[i]
        r_MEX2TGO, lt_MEX2TGO = spice.spkpos('TGO', et, 'IAU_MARS', 'LT', 'MEX')
        dis_MEX2TGO = spice.vnorm(r_MEX2TGO)
        
        if dis_MEX2TGO > max_distance:
            counter, et_values_reduced = remove_times(i, counter, et_values_reduced)

    et_values = et_values_reduced.copy()
    len_et_valuesf = len(et_values_reduced)

    log_print("Percentage of ET times removed with MEX-TGO distance over", max_distance, "km:",
              round(perc_removed(len_et_values0, len_et_valuesf),1), "%")
    log_print("Check 1.2 [END]: Remove times with MEX-TGO distance over", max_distance, "km")

    # ********************************************************************************************

    log_print("")
    log_print("Check 1.3 [START]: Remove times with MEX-TGO angular separation (taken from the" \
              " midpoint estimate) larger than", midpoint_emission_max, "degrees")

    et_values_reduced = et_values.copy()
    len_et_values0 = len(et_values_reduced)
    counter = 0

    for i in tqdm(range(len(et_values))):
        et = et_values[i]
        utc, dis_MEX2TGO, midpoint, lon_midpoint, lat_midpoint, r_midpoint2MEX, r_midpoint2TGO, \
            r_MEX2TGO, r_Mars2TGO, r_Mars2MEX, r_Mars2mid = find_midpoint(et)
        emission_midpoint_avg = 0.5 * spice.convrt(spice.vsep(r_midpoint2TGO, r_midpoint2MEX), \
                                                   "RADIANS", "DEGREES")

        if emission_midpoint_avg > midpoint_emission_max:
            counter, et_values_reduced = remove_times(i, counter, et_values_reduced)

    et_values = et_values_reduced.copy()
    len_et_valuesf = len(et_values_reduced)

    log_print("Percentage of ET times removed with MEX-TGO angular separation (taken from the " \
              "midpoint estimate) larger than", midpoint_emission_max, "degrees:", 
              round(perc_removed(len_et_values0, len_et_valuesf),1), "%")
    log_print("Check 1.3 [END]: Remove times with MEX-TGO angular separation (taken from the " \
              "midpoint estimate) larger than", midpoint_emission_max, "degrees")

    # ********************************************************************************************

    log_print("")
    log_print("Check 1.4 [START]: Remove times with midpoint spoints outside",
              lat_bound_times_midpoint, "degrees latitude")

    et_values_reduced = et_values.copy()
    len_et_values0 = len(et_values_reduced)
    counter = 0

    for i in tqdm(range(len(et_values))):
        et = et_values[i]
        utc, dis_MEX2TGO, midpoint, lon_midpoint, lat_midpoint, r_midpoint2MEX, r_midpoint2TGO, \
            r_MEX2TGO, r_Mars2TGO, r_Mars2MEX, r_Mars2mid = find_midpoint(et)
        
        lat_midpoint_deg = spice.convrt(lat_midpoint, 'RADIANS', 'DEGREES')

        if np.abs(lat_midpoint_deg) > lat_bound_times_midpoint:
            counter, et_values_reduced = remove_times(i, counter, et_values_reduced)

    et_values = et_values_reduced.copy()
    len_et_valuesf = len(et_values_reduced)

    log_print("Percentage of ET times removed with midpoint spoints outside",
              lat_bound_times_midpoint, "degrees latitude:",
              round(perc_removed(len_et_values0, len_et_valuesf),1), "%")
    log_print("Check 1.4 [END]: Remove times with midpoint spoints outside",
              lat_bound_times_midpoint, "degrees latitude")

    # ********************************************************************************************

    log_print("")
    log_print("Splitting times into measurement opportunities [START]")

    meas_ops = []
    i_split = 0

    for i in range(0, len(et_values)-1):
        et0 = et_values[i]
        et = et_values[i+1]

        et_diff = et - et0

        if et_diff > 1.5*time_step_float:                            
            meas_op = et_values[i_split:i+1]
            i_split = i+1
            meas_ops.append(meas_op)

        if i == len(et_values)-2:                            
            meas_op = et_values[i_split:i+2]
            meas_ops.append(meas_op)

    log_print("Resultant amount of opportunities:", len(meas_ops))
    log_print("Splitting times into measurement opportunities [END]")

    # ********************************************************************************************

    log_print("")
    log_print("Check 2.1 [START]: Remove opportunities with a duration under", meas_duration,
              "minutes")

    meas_ops_reduced = meas_ops.copy()
    len_meas_ops0 = len(meas_ops)
    
    counter = 0

    for i in tqdm(range(len(meas_ops))):
        meas_op = meas_ops[i]
        meas_op_et0 = meas_op[0]
        meas_op_etf = meas_op[-1]
        meas_op_dur_s = meas_op_etf - meas_op_et0
        meas_op_dur_m = meas_op_dur_s / 60

        if meas_op_dur_m < (meas_duration  - (1/120)): # Allows for a half second deviation
            counter, meas_ops_reduced = remove_meas_ops(i, counter, meas_ops_reduced)
    
    meas_ops = meas_ops_reduced.copy()
    len_meas_opsf = len(meas_ops_reduced)
    
    log_print("Percentage of measurement opportunities removed with duration under",
              meas_duration, "minutes:", round(perc_removed(len_meas_ops0, len_meas_opsf),1),
              "%; resultant amount of opportunities:", len(meas_ops))
    log_print("Check 2.1 [END]: Remove opportunities with a duration under", meas_duration,
              "minutes")

    # ********************************************************************************************

    log_print("")
    log_print("Check 2.2 [START]: Remove opportunities with midpoint spoint estimates outside",
              lat_bound_midpoint, "degrees latitude")

    meas_ops_reduced = meas_ops.copy()
    len_meas_ops0 = len(meas_ops_reduced)

    counter = 0
    
    for i in tqdm(range(len(meas_ops))):
        meas_op = meas_ops[i]
        meas_op_bool = False

        for et in meas_op:
            utc, dis_MEX2TGO, midpoint, lon_midpoint, lat_midpoint, r_midpoint2MEX, \
                r_midpoint2TGO, r_MEX2TGO, r_Mars2TGO, r_Mars2MEX, r_Mars2mid = find_midpoint(et)
                    
            lat_midpoint_deg = spice.convrt(lat_midpoint, 'RADIANS', 'DEGREES') 

            if np.abs(lat_midpoint_deg) < lat_bound_midpoint:
                # Equatorial region of interest probed
                meas_op_bool = True

        if meas_op_bool == False:
            counter, meas_ops_reduced = remove_meas_ops(i, counter, meas_ops_reduced)

    meas_ops = meas_ops_reduced.copy()
    len_meas_opsf = len(meas_ops)

    midpoints_original = meas_ops.copy()
    midpoints_data = midpoints_original.copy()

    log_print("Percentage of measurement opportunities removed with midpoint estimates outside",
              lat_bound_midpoint, "degrees latitude:",
              round(perc_removed(len_meas_ops0, len_meas_opsf),1), "%; resultant amount of " \
              "opportunities:", len(meas_ops))
    log_print("Check 2.2 [END]: Remove opportunities with midpoint spoint estimates outside",
              lat_bound_midpoint, "degrees latitude")

    # ********************************************************************************************

    log_print("")
    log_print("Calculate equiemission points [START]")

    equiemission_data = []

    for i in range(len(meas_ops)):
        print("")
        log_print("Measurement opportunity", i+1, "of", len(meas_ops))

        meas_op = meas_ops[i]
        equiemission_data_i = []
    
        for et in tqdm(meas_op):
            utc, dis_MEX2TGO, spoint, r_spoint2TGO, r_spoint2MEX, lon, lat, lat_sph, \
                emission_MEX, emission_TGO, lon_midpoint, lat_midpoint = find_equiemission(et)
            equiemission_data_i.append([
                utc, et,
                dis_MEX2TGO,
                spoint[0], spoint[1], spoint[2],
                spice.convrt(lon, 'RADIANS', 'DEGREES'),
                spice.convrt(lat_sph, 'RADIANS', 'DEGREES'),
                spice.convrt(lat, 'RADIANS', 'DEGREES'),
                emission_MEX, emission_TGO,
                spice.vnorm(r_spoint2MEX), spice.vnorm(r_spoint2TGO),
                spice.convrt(lon_midpoint, 'RADIANS', 'DEGREES'),
                spice.convrt(lat_midpoint, 'RADIANS', 'DEGREES')
            ])

        equiemission_data.append(equiemission_data_i)        
    
    print("")
    log_print("Calculate equiemission points [END]")

    # ********************************************************************************************

    log_print("")
    log_print("Check 3.1 [START]: Remove times with equiemission angles larger than",
              emission_max, "degrees")
    
    equiemission_data_len0 = 0
    equiemission_data_lenf = 0

    for i in tqdm(range(len(equiemission_data))):
        equiemission_data_i = equiemission_data[i]
        equiemission_data_i_reduced = equiemission_data[i].copy()
        counter = 0
        equiemission_data_len0 += len(equiemission_data_i)

        midpoints_i_reduced = midpoints_data[i].copy()

        print(len(equiemission_data_i_reduced))
        print(len(midpoints_i_reduced))

        for j in range(len(equiemission_data_i)):
            emission_MEX = equiemission_data_i[j][9]
            emission_TGO = equiemission_data_i[j][10]
            emission_avg = (emission_MEX + emission_TGO)/2
            
            if emission_avg > emission_max:
                counter, equiemission_data_i_reduced, midpoints_i_reduced = \
                    remove_equi_and_midpoint_times(j, counter, equiemission_data_i_reduced,
                                                   midpoints_i_reduced)
        
        print(len(equiemission_data_i_reduced))
        print(len(midpoints_i_reduced))
        
        equiemission_data_lenf += len(equiemission_data_i_reduced)

        equiemission_data[i] = equiemission_data_i_reduced.copy()
        midpoints_data[i] = midpoints_i_reduced.copy()	

    log_print("Percentage of times removed with equiemission angles larger than", emission_max,
              "degrees:", round(perc_removed(equiemission_data_len0, equiemission_data_lenf),1),
              "%")
    log_print("Check 3.1 [END]: Remove times with equiemission angles larger than", emission_max,
              "degrees")

    # ********************************************************************************************

    log_print("")
    log_print("Check 3.2 [START]: Remove times with equiemission spoints outside",
              lat_bound_times, "degrees latitude")
    
    equiemission_data_len0 = 0
    equiemission_data_lenf = 0

    for i in tqdm(range(len(equiemission_data))):
        equiemission_data_i = equiemission_data[i]
        equiemission_data_i_reduced = equiemission_data[i].copy()
        counter = 0
        equiemission_data_len0 += len(equiemission_data_i)

        midpoints_i_reduced = midpoints_data[i].copy()

        for j in range(len(equiemission_data_i)):
            lat_spoint = equiemission_data_i[j][8]
            
            if np.abs(lat_spoint) > lat_bound_times:
                counter, equiemission_data_i_reduced, midpoints_i_reduced = \
                    remove_equi_and_midpoint_times(j, counter, equiemission_data_i_reduced,
                                                   midpoints_i_reduced)
        
        print(len(equiemission_data_i_reduced))
        print(len(midpoints_i_reduced))
        
        equiemission_data_lenf += len(equiemission_data_i_reduced)

        equiemission_data[i] = equiemission_data_i_reduced.copy()
        midpoints_data[i] = midpoints_i_reduced.copy()	

    log_print("Percentage of times removed with equiemission spoints outside", lat_bound_times,
              "degrees latitude:", round(perc_removed(equiemission_data_len0,
                                                      equiemission_data_lenf),1), "%")
    log_print("Check 3.2 [END]: Remove times with equiemission spoints outside", lat_bound_times,
              "degrees latitude")

    # ********************************************************************************************

    log_print("")
    log_print("Splitting measurement opportunities with discontinuous times [START]")

    midpoints_data_split = []
    equiemission_data_split = []

    for i in range(len(midpoints_data)):
        midpoints_data_i = midpoints_data[i]
        equiemission_data_i = equiemission_data[i]

        j_split = 0

        for j in range(0, len(midpoints_data_i)-1):
            et0 = midpoints_data_i[j]
            et = midpoints_data_i[j+1]
            et_diff = et - et0

            if et_diff > 1.5*time_step_float:       
                midpoints_data_split.append(midpoints_data_i[j_split:j+1])
                equiemission_data_split.append(equiemission_data_i[j_split:j+1])
                j_split = j+1

            if j == len(midpoints_data_i)-2:      
                midpoints_data_split.append(midpoints_data_i[j_split:j+2])
                equiemission_data_split.append(equiemission_data_i[j_split:j+2])
        
        print(len(equiemission_data_i_reduced))
        print(len(midpoints_i_reduced))

    midpoints_data = midpoints_data_split.copy()
    equiemission_data = equiemission_data_split.copy()

    log_print("Resultant amount of opportunities:", len(equiemission_data))
    log_print("Splitting measurement opportunities with discontinuous times [END]")

    # ********************************************************************************************

    log_print("")
    log_print("Check 4.1 [START]: Remove opportunities with a duration under", meas_duration,
              "minutes")

    equiemission_data_reduced = equiemission_data.copy()
    len_equi_data0 =  len(equiemission_data)
    counter = 0
    midpoints_reduced = midpoints_data.copy()

    for i in range(len(equiemission_data)):
        equiemission_data_i = equiemission_data[i]
        midpoints_data_i = midpoints_data[i]

        if len(equiemission_data_i) >= 2:
            equi_et0 = equiemission_data_i[0][1] 
            equi_etf = equiemission_data_i[-1][1] 
            equi_dur_s = equi_etf - equi_et0
            equi_dur_m = equi_dur_s / 60

            if equi_dur_m < (meas_duration - (1/120)): # Allows for a half second deviation
                counter, equiemission_data_reduced, midpoints_reduced = \
                    remove_equi_and_midpoint(i, counter, equiemission_data_reduced,
                                             midpoints_reduced)

        else:
            counter, equiemission_data_reduced, midpoints_reduced = \
                remove_equi_and_midpoint(i, counter, equiemission_data_reduced, midpoints_reduced)
    
    equiemission_data = equiemission_data_reduced.copy()
    midpoints_data = midpoints_reduced.copy()
    len_equi_dataf =  len(equiemission_data)

    log_print("Percentage of measurement opportunities removed with duration under",
              meas_duration, "minutes:", round(perc_removed(len_equi_data0, len_equi_dataf),1),
              "%; resultant amount of opportunities:", len(equiemission_data))
    log_print("Check 4.1 [END]: Remove opportunities with a duration under", meas_duration,
              "minutes")

    # ********************************************************************************************

    log_print("")
    log_print("Check 4.2 [START]: Remove opportunities with equiemission points outside",
              lat_bound, "degrees latitude")
    
    equiemission_data_reduced = equiemission_data.copy()
    midpoints_reduced = midpoints_data.copy()
    len_equi_data0 = len(equiemission_data)
    counter = 0

    for i in tqdm(range(len(equiemission_data))):
        equiemission_data_i = equiemission_data[i]
        midpoints_data_i = midpoints_data[i]
        meas_op_bool = False

        for j in range(len(equiemission_data_i)):
            lat_deg = equiemission_data_i[j][8]

            if np.abs(lat_deg) < lat_bound:
                # Equatorial region of interest probed
                meas_op_bool = True
        
        if meas_op_bool == False:
            counter, equiemission_data_reduced, midpoints_reduced = \
                remove_equi_and_midpoint(i, counter, equiemission_data_reduced, midpoints_reduced)

    equiemission_data = equiemission_data_reduced.copy()
    midpoints_data = midpoints_reduced.copy()
    len_equi_dataf = len(equiemission_data)

    log_print("Percentage of measurement opportunities removed with equiemission points outside",
              lat_bound, "degrees latitude:", round(perc_removed(len_equi_data0,
              len_equi_dataf),1), "%; resultant amount of opportunities:", len(equiemission_data))
    log_print("Check 4.2 [END]: Remove opportunities with equiemission points outside",
              lat_bound, "degrees latitude")

    # ********************************************************************************************

    log_print("")
    log_print("Check 4.3 [START]: Remove opportunities with lowest point at MEX-TGO distance over", max_distance_lowestpoint, "km")
    
    equiemission_data_reduced = equiemission_data.copy()
    midpoints_reduced = midpoints_data.copy()
    len_equi_data0 = len(equiemission_data)
    counter = 0

    for i in range(len(equiemission_data)):
        equiemission_data_i = equiemission_data[i]
        midpoints_data_i = midpoints_data[i]

        equi_dis_i = []

        for j in range(len(equiemission_data_i)):
            equi_dis_ij = equiemission_data_i[j][2]
            equi_dis_i.append(equi_dis_ij)

        lp_dis_i = min(equi_dis_i)

        if lp_dis_i > max_distance_lowestpoint:
            counter, equiemission_data_reduced, midpoints_reduced = \
                remove_equi_and_midpoint(i, counter, equiemission_data_reduced, midpoints_reduced)

    equiemission_data = equiemission_data_reduced.copy()
    midpoints_data = midpoints_reduced.copy()
    len_equi_dataf = len(equiemission_data)

    log_print("Percentage of measurement opportunities removed with lowest point at MEX-TGO " \
              "distance over", max_distance_lowestpoint, "km:",
              round(perc_removed(len_equi_data0, len_equi_dataf),1), "%; " \
              "resultant amount of opportunities:", len(equiemission_data))
    log_print("Check 4.3 [END]: Remove opportunities with lowest point at MEX-TGO distance over",
              max_distance_lowestpoint, "km")
    
    # *******************************************************************************************

    log_print("")
    log_print("Check 4.4 [START]: Remove opportunities with lowest point at equiemission angle" \
              " larger than", emission_max_lowestpoint, "degrees")
    
    equiemission_data_reduced = equiemission_data.copy()
    midpoints_reduced = midpoints_data.copy()
    len_equi_data0 = len(equiemission_data_reduced)
    counter = 0

    for i in range(len(equiemission_data)):
        equiemission_data_i = equiemission_data[i]
        midpoints_data_i = midpoints_data[i]

        equi_ang_i = []

        for j in range(len(equiemission_data_i)):
            equi_ang_ij_MEX = equiemission_data_i[j][9]
            equi_ang_ij_TGO = equiemission_data_i[j][10]
            equi_ang_ij_avg = (equi_ang_ij_MEX + equi_ang_ij_TGO)/2
            equi_ang_i.append(equi_ang_ij_avg)

        lp_ang_i = np.min(equi_ang_i)

        if lp_ang_i > emission_max_lowestpoint:
            counter, equiemission_data_reduced, midpoints_reduced = \
                remove_equi_and_midpoint(i, counter, equiemission_data_reduced, midpoints_reduced)

    equiemission_data = equiemission_data_reduced.copy()
    midpoints_data = midpoints_reduced.copy()
    len_equi_dataf = len(equiemission_data_reduced)

    log_print("Percentage of measurement opportunities removed with lowest point at equiemission" \
              "angle larger than", emission_max_lowestpoint, "degrees:", 
              round(perc_removed(len_equi_data0, len_equi_dataf),1),
              "%; resultant amount of opportunities:", len(equiemission_data))
    log_print("Check 4.4 [END]: Remove opportunities with lowest point at equiemission angle " \
              "larger than", emission_max_lowestpoint, "degrees")
    
    # ********************************************************************************************

    log_print("")
    log_print("Write resultant midpoint data to file [START]")

    midpoint_names_file_name = 'midpoints.txt'        
    midpoint_names_file_path = os.path.join(data_directory_path, midpoint_names_file_name)
    midpoint_names_file = open(midpoint_names_file_path, 'w')

    midpoint_found = False

    for i in range(len(midpoints_data)):
        midpoint_found = True

        meas_op = midpoints_data[i]

        et0_midpoint = meas_op[0]
        etf_midpoint = meas_op[-1]
        utc0_midpoint = spice.et2utc(et0_midpoint, 'C', 0)
        utcf_midpoint = spice.et2utc(etf_midpoint, 'C', 0)

        data_file_name = 'midpoint_' + utc0_midpoint.replace(" ", ".").replace(":", ".") + '-' \
                          + utcf_midpoint.replace(" ", ".").replace(":", ".") + '.csv'
        data_file_path = os.path.join(data_directory_path, data_file_name)
        data_file = open(data_file_path, 'w')

        data_file.write('# utc [-], et [s], MEX2TGO [km], midpoint_x [km], midpoint_y [km],' \
                        'midpoint_z [km], midpoint_lon [deg], midpoint_lat [deg], midpoint2MEX' \
                        '[km], midpoint2TGO [km] \n')

        for j in range(len(meas_op)):
            et = meas_op[j]

            utc, dis_MEX2TGO, midpoint, lon_midpoint, lat_midpoint, r_midpoint2MEX, \
                r_midpoint2TGO, r_MEX2TGO, r_Mars2TGO, r_Mars2MEX, r_Mars2mid = find_midpoint(et)

            data_file.write(utc + ', ' + str(et) + ', ' + str(dis_MEX2TGO)+ ', ' 
                            + str(midpoint[0]) + ', ' + str(midpoint[1]) + ', ' 
                            + str(midpoint[2]) + ', ' 
                            + str(spice.convrt(lon_midpoint, 'RADIANS', 'DEGREES')) + ', '
                            + str(spice.convrt(lat_midpoint, 'RADIANS', 'DEGREES')) + ', '
                            + str(spice.vnorm(r_midpoint2MEX)) + ', ' 
                            + str(spice.vnorm(r_midpoint2TGO)) + ', ' + '\n') 

        midpoint_names_file.write(data_file_name + " ")
        data_file.close()

    midpoint_names_file.close()

    log_print("Write resultant midpoint data to file [END]")

    # *******************************************************************************************

    log_print("")
    log_print("Write resultant equiemission data to file [START]")

    meas_op_names_file_name = 'meas_ops.txt'        
    meas_op_names_file_path = os.path.join(data_directory_path, meas_op_names_file_name)
    meas_op_names_file = open(meas_op_names_file_path, 'w')

    meas_op_found = False

    for i in range(len(equiemission_data)):
        meas_op_found = True
        equiemission_data_i = equiemission_data[i]

        et0_equi_data_i = equiemission_data_i[0][1]
        etf_equi_data_i = equiemission_data_i[-1][1]
        utc0_equi_data_i = spice.et2utc(et0_equi_data_i, 'C', 0)
        utcf_equi_data_i = spice.et2utc(etf_equi_data_i, 'C', 0)

        data_file_name = 'meas_op_' + utc0_equi_data_i.replace(" ", ".").replace(":", ".") \
                          + '-' + utcf_equi_data_i.replace(" ", ".").replace(":", ".") + '.csv'
        data_file_path = os.path.join(data_directory_path, data_file_name)
        data_file = open(data_file_path, 'w')

        data_file.write('# utc [-], et [s], MEX2TGO [km], spoint_x [km], spoint_y [km], ' \
                        'spoint_z [km], spoint_lon [deg], spoint_lat_sph [deg], spoint_lat ' \
                        '[deg], emission_MEX [deg], emission_TGO [deg], spoint2MEX [km], ' \
                        'spoint2TGO [km], lon_midpoint [deg], lat_midpoint [deg] \n')

        for j in range(len(equiemission_data_i)):      
            data_file.write(str(equiemission_data_i[j][0]) + ', ' \
                            + str(equiemission_data_i[j][1]) + ', ' \
                            + str(equiemission_data_i[j][2]) + ', ' \
                            + str(equiemission_data_i[j][3]) + ', ' \
                            + str(equiemission_data_i[j][4]) + ', ' \
                            + str(equiemission_data_i[j][5]) + ', ' \
                            + str(equiemission_data_i[j][6]) + ', ' \
                            + str(equiemission_data_i[j][7]) + ', ' \
                            + str(equiemission_data_i[j][8]) + ', ' \
                            + str(equiemission_data_i[j][9]) + ', ' \
                            + str(equiemission_data_i[j][10]) + ', ' \
                            + str(equiemission_data_i[j][11]) + ', ' \
                            + str(equiemission_data_i[j][12]) + ', ' \
                            + str(equiemission_data_i[j][13]) + ', ' \
                            + str(equiemission_data_i[j][14]) + '\n')

        data_file.close()
        meas_op_names_file.write(data_file_name + " ")

    meas_op_names_file.close()

    log_print("Write resultant equiemission data to file [END]")

    # *******************************************************************************************

    if midpoint_found == True and meas_op_found == True:
        plot_spoints(utc0, utcf, data_directory_path, 'MOP', MTP_i, BSR_i)
    else:
        log_print("")
        log_print("No measurement opportunities found in the selected time range:", utc0,
                  "to", utcf)

    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    run_time_s = end_time - start_time
    run_time_m = run_time_s / 60

    log_print("")
    log_print("Script ended at:", end_time_str)

    if run_time_s < 100:
        log_print("Total runtime:", round(run_time_s,2), "seconds\n")
    else:
        log_print("Total runtime:", round(run_time_m,2), "minutes\n")

# ************************************************************************************************

spice.kclear()

# ************************************************************************************************