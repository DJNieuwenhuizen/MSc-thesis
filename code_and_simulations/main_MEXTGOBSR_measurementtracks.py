# *************************************************************************************************
#
#  main_MEXTGOBSR_measurementtracks.py
#
#  Description:
#  |  Code used to determine the measurement tracks of the MEX-TGO BSR measurements
#  |  Output:
#     |  PNG plots of the ground tracks for measurement opportunities, midpoints and both combined
#               - a 'run.log' file with the script's logging information
#               - a 'midpoints.txt' file with the names of the midpoint data files
#               - a 'meas_ops.txt' file with the names of the measurement opportunity data files
#               - a .csv file describing the midpoints and measurement opportunity data for each
#                 individual opportunity
#               - three .png 
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

from cycler import cycler
import glob
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import spiceypy as spice
import time
from tqdm import tqdm

from my_functions import plot_spoints, plot_STLs, set_plot_info
from my_parameters import Mars_param, BSR_param

# *************************************************************************************************

""" Define time frames for processing measurement tracks """
""" ------------> Uncomment block for processing performed measurements """
time_frames = [[['2024 JAN 23 03:49:07', '2024 JAN 23 03:59:07'],
               ['2024 JAN 30 10:30:44', '2024 JAN 30 10:40:44'],
               ['2024 FEB 06 09:43:22', '2024 FEB 06 09:53:22'],
               ['2024 FEB 11 01:53:19', '2024 FEB 11 02:03:19']],
               [['2025 MAR 20 23:40:54', '2025 MAR 20 23:50:54'],
               ['2025 APR 5 16:56:04', '2025 APR 5 17:06:04']],
               [['2025 JUL 17 10:38:31', '2025 JUL 17 10:48:31']],
               [['2025 AUG 15 19:39:12', '2025 AUG 15 19:58:12']]]
BSR_titles = [['BSR-1.1', 'BSR-1.2', 'BSR-1.3', 'BSR-1.4'], ['BSR-2.1', 'BSR-2.2'], ['BSR-3.1'], ['BSR-4.1']]
BSR_campaigns = ['1', '2', '3', '4']
MTP_selections = ['MTP257/76', 'MTP272/91', 'MTP276/95', 'MTP277/96']
MTP_dirnames = ['MTP257-76', 'MTP272-91', 'MTP276-95', 'MTP277-96']
usage = 'BSR'

""" ------------> Uncomment block for processing (to be) performed measurements """
# time_frames = [[['2024 JAN 23 03:49:07', '2024 JAN 23 03:59:07'],
#                ['2024 JAN 30 10:30:44', '2024 JAN 30 10:40:44'],
#                ['2024 FEB 06 09:43:22', '2024 FEB 06 09:53:22'],
#                ['2024 FEB 11 01:53:19', '2024 FEB 11 02:03:19']],
#                [['2025 MAR 20 23:40:54', '2025 MAR 20 23:50:54'],
#                ['2025 APR 5 16:56:04', '2025 APR 5 17:06:04']],
#                [['2025 JUL 17 10:38:31', '2025 JUL 17 10:48:31']],
#                [['2025 AUG 15 19:39:12', '2025 AUG 15 19:58:12']],
#                [['2025 SEP 22 22:56:40', '2025 SEP 22 23:06:40']],
#                [['2025 SEP 30 19:33:20', '2025 SEP 30 19:43:20'],
#                 ['2025 OCT 07 05:02:15', '2025 OCT 07 05:10:15'],
#                 ['2025 OCT 13 00:38:36', '2025 OCT 13 00:57:36'],
#                 ['2025 OCT 15 15:30:52', '2025 OCT 15 15:40:52'],
#                 ['2025 OCT 16 05:21:53', '2025 OCT 16 05:41:53']],
#                [['2025 NOV 01 12:33:54', '2025 NOV 01 12:43:54'],
#                 ['2025 NOV 04 03:21:58', '2025 NOV 04 03:31:58'],
#                 ['2025 NOV 09 09:10:58', '2025 NOV 09 09:20:58'],
#                 ['2025 NOV 15 04:49:32', '2025 NOV 15 04:59:32']],
#                [['2025 NOV 25 16:19:48', '2025 NOV 25 16:29:48'],
#                 ['2025 NOV 30 22:04:33', '2025 NOV 30 22:14:33'],
#                 ['2025 DEC 11 23:25:54', '2025 DEC 11 23:35:54'],
#                 ['2025 DEC 14 14:12:14', '2025 DEC 14 14:22:14']]]
# BSR_titles = [['BSR-1.1', 'BSR-1.2', 'BSR-1.3', 'BSR-1.4'], ['BSR-2.1', 'BSR-2.2'], ['BSR-3.1'], ['BSR-4.1'], ['BSR-5.1'], ['BSR-6.1', 'BSR-6.2', 'BSR-6.3', 'BSR-6.4', 'BSR-6.5'], ['BSR-7.1', 'BSR-7.2', 'BSR-7.3', 'BSR-7.4'], ['BSR-8.1', 'BSR-8.2', 'BSR-8.3', 'BSR-8.4']]
# BSR_campaigns = ['1', '2', '3', '4', '5', '6', '7', '8']
# MTP_selections = ['MTP257/76', 'MTP272/91', 'MTP276/95', 'MTP277/96', 'MTP278/97', 'MTP279/98', 'MTP280/99', 'MTP281/100']
# MTP_dirnames = ['MTP257-76', 'MTP272-91', 'MTP276-95', 'MTP277-96', 'MTP278-97', 'MTP279-98', 'MTP280-99', 'MTP281-100']
# usage = 'BSR'

""" ------------> Uncomment block for processing requested measurements (July) """
# time_frames = [[['2025 JUL 11 15:15:07', '2025 JUL 11 15:25:07'],
#                ['2025 JUL 12 04:59:27', '2025 JUL 12 05:09:27'],
#                ['2025 JUL 12 05:09:27', '2025 JUL 12 05:19:27'],
#                ['2025 JUL 14 19:53:08', '2025 JUL 14 20:03:08'],
#                ['2025 JUL 14 20:03:08', '2025 JUL 14 20:13:08'],
#                ['2025 JUL 17 10:37:50', '2025 JUL 17 10:47:50'],
#                ['2025 JUL 20 01:49:31', '2025 JUL 20 01:59:31'],
#                ['2025 JUL 27 22:22:06', '2025 JUL 27 22:32:06']]]
# BSR_titles = [['meas_op-3.1', 'meas_op-3.2', 'meas_op-3.3', 'meas_op-3.4', 'meas_op-3.5', 'meas_op-3.6', 'meas_op-3.7', 'meas_op-3.8']]
# BSR_campaigns = ['3']
# MTP_selections = ['MTP276/95']
# MTP_dirnames = ['MTP276-95']
# usage = 'MOP'

""" ------------> Uncomment block for processing requested measurements (August) """
# time_frames = [[['2025 AUG 02 04:09:47', '2025 AUG 02 04:19:47'],
#               ['2025 AUG 02 17:50:32', '2025 AUG 02 18:00:32'],
#               ['2025 AUG 05 08:49:07', '2025 AUG 05 08:59:07'],
#               ['2025 AUG 07 09:51:03', '2025 AUG 07 10:01:03'],
#               ['2025 AUG 16 10:14:16', '2025 AUG 16 10:24:16'],
#               ['2025 AUG 18 11:16:13', '2025 AUG 18 11:26:13'],
#               ['2025 AUG 28 01:42:36', '2025 AUG 28 01:52:36']]]
# BSR_titles = [['meas_op-4.1', 'meas_op-4.2', 'meas_op-4.3', 'meas_op-4.4', 'meas_op-4.5', 'meas_op-4.6', 'meas_op-4.7']]
# BSR_campaigns = ['4']
# MTP_selections = ['MTP276/95 and MTP277/96']
# MTP_dirnames = ['MTP276-95andMTP277-96']
# usage = 'MOP'

# *************************************************************************************************

""" Load SPICE kernels """

MetaKernelsPath = os.path.join(os.path.dirname(os.getcwd()),
                               "thesis\\code_and_simulations\\kernels\\mk")

main_gen_mk = os.path.join(MetaKernelsPath, 'gen_mk.tm') # General Solar System/Time/Mars kernels
main_MEX_mk = os.path.join(MetaKernelsPath, 'MEX_mk.tm') # MEX-specific kernels
main_TGO_mk = os.path.join(MetaKernelsPath, 'TGO_mk.tm') # TGO-specific kernels

spice.furnsh(main_gen_mk)
spice.furnsh(main_MEX_mk)
spice.furnsh(main_TGO_mk)

# *************************************************************************************************

""" Measurement limitations/requirements & Mars geometry """

if usage == 'BSR':
    time_step_int = 1 # seconds; time step interval (ET times)
elif usage == 'MOP':
    time_step_int = 10 # seconds; time step interval (ET times)

equiemission_lat_errorbound = np.deg2rad(1e-2) # radians; allowed deviation of the spoint
                                               # normal vector from the MEX-spoint-TGO plane
equiemission_diff_errorbound = 1e-2 # degrees; allowed emission angle difference of MEX and TGO
n_iterations_max = 1000 # Maximum number of iterations

radii_ell = Mars_param['radii_ell'] # km; ellipsoidal Mars
radii_sph = Mars_param['radii_sph'] # km; spherical Mars

# *************************************************************************************************

""" Functions """

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

def plot_all_tracks():
    """ 
    plot_all_tracks: This function plots the ground tracks of all MEX-TGO BSR measurements 
        in a single figure
    """

    rcParams['font.size'] = 14
    code_and_sim_path = os.path.join(os.path.dirname(os.getcwd()),
                                "thesis\\code_and_simulations")
    directory_path = os.path.join(code_and_sim_path, "data\\meas_ops\\processing")
    img_MarsBG_Path = os.path.join(code_and_sim_path,
                                "data\\HRSC-MOLA_Blended-DEM\\RecBG_dem_Mars_1500mp.png")
    img_MarsBackground = plt.imread(img_MarsBG_Path)

    for plot in ['_all', '_midpoints', '_meas_ops']:
        if plot == '_all':
            rcParams['axes.prop_cycle'] = cycler(color=plt.cm.tab20.colors)
        else:
            rcParams['axes.prop_cycle'] = cycler(color=plt.cm.tab10.colors)

        plot_type = plot

        fig_MarsSpoints = plt.figure(figsize=(12, 10))
        fig_MarsSpoints.set_size_inches(19.2, 10.8)  # Nominal full screen: 1920x1080
        plt.imshow(img_MarsBackground, extent=[-180, 180, -90, 90], aspect='equal')
        
        plot_STLs()

        for BSR_campaign in BSR_campaigns:
            BSR_campaign_id = BSR_campaigns.index(BSR_campaign)
            MTP_i = MTP_selections[BSR_campaign_id]

            # ********************************************************************************************

            """ Directory and file setup """

            MTP_i_dirname = MTP_dirnames[BSR_campaign_id]
            data_directory_path = os.path.join(directory_path, MTP_i_dirname)
                
            midpoints_txt_path = os.path.join(data_directory_path, 'midpoints.txt')
            midpoints_txt = open(midpoints_txt_path, 'r')
            midpoints_lines = midpoints_txt.readlines()
            midpoints_txt.close()
            midpoints_list = midpoints_lines[0].split()

            meas_ops_txt_path = os.path.join(data_directory_path, 'meas_ops.txt')
            meas_ops_txt = open(meas_ops_txt_path, 'r')
            meas_ops_lines = meas_ops_txt.readlines()
            meas_ops_txt.close()
            meas_ops_list = meas_ops_lines[0].split()  

            # ********************************************************************************************

            for BSR_i in BSR_titles[BSR_campaign_id]:
                BSR_i_id = BSR_titles[BSR_campaign_id].index(BSR_i)

                # ********************************************************************************************

                utc0 = time_frames[BSR_campaign_id][BSR_i_id][0]
                utcf = time_frames[BSR_campaign_id][BSR_i_id][1]

                # ********************************************************************************************
                marker_size = 10

                if plot_type == '_all' or plot_type == '_meas_ops':
                    i = meas_ops_list[BSR_i_id]
                    label_i = f"{BSR_i}: {utc0} - {utcf[12:]}"
                    meas_op_path = os.path.join(os.path.join(directory_path, MTP_i_dirname), i)
                    meas_op_lon = pd.read_csv(meas_op_path, usecols=[6])
                    meas_op_lat = pd.read_csv(meas_op_path, usecols=[8])
                    plt.scatter(meas_op_lon, meas_op_lat, label=label_i, s=marker_size)

                if plot_type == '_all' or plot_type == '_midpoints':
                    i = midpoints_list[BSR_i_id]
                    if plot_type == '_all':
                        label_i = f"{BSR_i} (midpoint track)"
                    if plot_type == '_midpoints':
                        label_i = f"midpoint-{BSR_i[4:]}: {utc0} - {utcf}"
                    midpoint_path = os.path.join(os.path.join(directory_path, MTP_i_dirname), i)
                    midpoint_lon = pd.read_csv(midpoint_path, usecols=[6])
                    midpoint_lat = pd.read_csv(midpoint_path, usecols=[7])
                    plt.scatter(midpoint_lon, midpoint_lat, label=label_i, s=marker_size)
                    if plot_type == '_all' and BSR_i != BSR_titles[-1][-1]:
                        plt.plot(np.nan, np.nan, '-', color='none', label=' ')
                
        ax_MarsSpoints = plt.gca()

        title = f'MEX-TGO BSR tracks (Campaigns {BSR_campaigns[0]}-{BSR_campaigns[-1]}: {MTP_selections})'

        fig_MarsSpoints.text(s=title, x=0.43, y=0.72+0.09, fontsize=18, ha='center', va='center', fontweight='bold')
        fig_MarsSpoints.text(s='(Background: HRSC_MOLA Blended DEM at 1500mp)', x=0.43, y=0.695+0.09, fontsize=12, ha='center', va='center')

        set_plot_info(ax_MarsSpoints)

        fig_name = 'spoints' + plot + '.png'
        fig_path = os.path.join(directory_path, fig_name)
        fig_MarsSpoints.savefig(fig_path, bbox_inches='tight', pad_inches=0.05, dpi=150)
        plt.close(fig_MarsSpoints)

    return

def plot_all_tracks_singlecampaign(BSR_campaign_id):
    """
    plot_all_tracks_singlecampaign: This function plots the ground tracks of all MEX-TGO BSR 
        measurements of a single campaign in a single figure
        
    :param BSR_campaign_id: ID of the BSR campaign to be processed (0, 1, 2, ...)
    :type BSR_campaign_id: class 'int'
    """

    rcParams['font.size'] = 14
    code_and_sim_path = os.path.join(os.path.dirname(os.getcwd()),
                                "thesis\\code_and_simulations")
    if usage == 'MOP':
        rcParams['axes.prop_cycle'] = cycler(color=plt.cm.tab20.colors)
        directory_path = os.path.join(code_and_sim_path, "data\\meas_ops\\planning")
    if usage == 'BSR':
        directory_path = os.path.join(code_and_sim_path, "data\\meas_ops\\processing")
    img_MarsBG_Path = os.path.join(code_and_sim_path,
                                "data\\HRSC-MOLA_Blended-DEM\\RecBG_dem_Mars_1500mp.png")
    img_MarsBackground = plt.imread(img_MarsBG_Path)

    for plot in ['_all', '_midpoints', '_meas_ops']:
        plot_type = plot

        fig_MarsSpoints = plt.figure(figsize=(12, 10))
        fig_MarsSpoints.set_size_inches(19.2, 10.8)  # Nominal full screen: 1920x1080
        plt.imshow(img_MarsBackground, extent=[-180, 180, -90, 90], aspect='equal')
        
        plot_STLs()

        BSR_campaign = BSR_campaigns[BSR_campaign_id]
        MTP_i = MTP_selections[BSR_campaign_id]

        # ********************************************************************************************

        """ Directory and file setup """

        MTP_i_dirname = MTP_dirnames[BSR_campaign_id]
        data_directory_path = os.path.join(directory_path, MTP_i_dirname)
            
        midpoints_txt_path = os.path.join(data_directory_path, 'midpoints.txt')
        midpoints_txt = open(midpoints_txt_path, 'r')
        midpoints_lines = midpoints_txt.readlines()
        midpoints_txt.close()
        midpoints_list = midpoints_lines[0].split()

        meas_ops_txt_path = os.path.join(data_directory_path, 'meas_ops.txt')
        meas_ops_txt = open(meas_ops_txt_path, 'r')
        meas_ops_lines = meas_ops_txt.readlines()
        meas_ops_txt.close()
        meas_ops_list = meas_ops_lines[0].split()  

        # ********************************************************************************************

        for BSR_i in BSR_titles[BSR_campaign_id]:
            BSR_i_id = BSR_titles[BSR_campaign_id].index(BSR_i)

            # ********************************************************************************************

            utc0 = time_frames[BSR_campaign_id][BSR_i_id][0]
            utcf = time_frames[BSR_campaign_id][BSR_i_id][1]

            # ********************************************************************************************
            marker_size = 10

            if plot_type == '_all' or plot_type == '_meas_ops':
                i = meas_ops_list[BSR_i_id]
                label_i = f"{BSR_i}: {utc0} - {utcf[12:]}"
                meas_op_path = os.path.join(os.path.join(directory_path, MTP_i_dirname), i)
                meas_op_lon = pd.read_csv(meas_op_path, usecols=[6])
                meas_op_lat = pd.read_csv(meas_op_path, usecols=[8])
                plt.scatter(meas_op_lon, meas_op_lat, label=label_i, s=marker_size)

            if plot_type == '_all' or plot_type == '_midpoints':
                i = midpoints_list[BSR_i_id]
                if plot_type == '_all':
                    label_i = f"{BSR_i} (midpoint track)"
                if plot_type == '_midpoints':
                    label_i = f"midpoint-{BSR_i[4:]}: {utc0} - {utcf}"
                midpoint_path = os.path.join(os.path.join(directory_path, MTP_i_dirname), i)
                midpoint_lon = pd.read_csv(midpoint_path, usecols=[6])
                midpoint_lat = pd.read_csv(midpoint_path, usecols=[7])
                plt.scatter(midpoint_lon, midpoint_lat, label=label_i, s=marker_size)
                if plot_type == '_all' and BSR_i != BSR_titles[BSR_campaign_id][-1]:
                    plt.plot(np.nan, np.nan, '-', color='none', label=' ')
                
        ax_MarsSpoints = plt.gca()

        if usage == 'MOP':
            title = f'Ground tracks for MEX-TGO BSR (meas_ops-{BSR_campaign}.x; {MTP_selections[BSR_campaign_id]})'
        if usage == 'BSR':
            title = f'Ground tracks for MEX-TGO BSR (BSR-{BSR_campaign}.x; {MTP_selections[BSR_campaign_id]})'

        fig_MarsSpoints.text(s=title, x=0.42, y=0.72+0.09, fontsize=18, ha='center', va='center', fontweight='bold')
        fig_MarsSpoints.text(s='(Background: HRSC_MOLA Blended DEM at 1500mp)', x=0.42, y=0.695+0.09, fontsize=12, ha='center', va='center')

        set_plot_info(ax_MarsSpoints)

        fig_name = 'spoints' + plot + '.png'
        fig_path = os.path.join(os.path.join(directory_path, MTP_dirnames[BSR_campaign_id]), fig_name)
        fig_MarsSpoints.savefig(fig_path, bbox_inches='tight', pad_inches=0.05, dpi=150)
        plt.close(fig_MarsSpoints)

    return

# ************************************************************************************************

""" Main script/loop """

start_time = time.time()
start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("\nScript started at:", start_time_str)

# plot_all_tracks()

# for singcamp in range(len(BSR_campaigns)):
#     plot_all_tracks_singlecampaign(singcamp)

for BSR_campaign in BSR_campaigns:
    BSR_campaign_id = BSR_campaigns.index(BSR_campaign)

    MTP_i = MTP_selections[BSR_campaign_id]
    print("\nRunning for BSR campaign:", BSR_campaign, "in", MTP_i)

    # ********************************************************************************************

    """ Directory and file setup """

    MTP_i_dirname = MTP_dirnames[BSR_campaign_id]

    if usage == 'BSR':
        data_directory_path = os.path.join(os.path.join(os.path.dirname(os.getcwd()),
                                                        "thesis\code_and_simulations\data\meas_ops\processing"),
                                                        MTP_i_dirname)
    elif usage == 'MOP':
        data_directory_path = os.path.join(os.path.join(os.path.dirname(os.getcwd()),
                                                        "thesis\code_and_simulations\data\meas_ops\planning"),
                                                        MTP_i_dirname)
        
    data_directory = Path(data_directory_path)
    data_directory.mkdir(parents=True, exist_ok=True)

    old_files = glob.glob(data_directory_path + '/*.csv') + \
                glob.glob(data_directory_path + '/*.txt') + \
                glob.glob(data_directory_path + '/*.png')
    for old_file in old_files:
        # Delete data files from previous runs
        os.remove(old_file)

    midpoint_names_file_name = 'midpoints.txt'        
    midpoint_names_file_path = os.path.join(data_directory_path, midpoint_names_file_name)
    midpoint_names_file = open(midpoint_names_file_path, 'w')

    meas_op_names_file_name = 'meas_ops.txt'        
    meas_op_names_file_path = os.path.join(data_directory_path, meas_op_names_file_name)
    meas_op_names_file = open(meas_op_names_file_path, 'w')

    # ********************************************************************************************

    for BSR_i in BSR_titles[BSR_campaign_id]:
        BSR_i_id = BSR_titles[BSR_campaign_id].index(BSR_i)
        print("\nRunning for:", BSR_i)

        # ********************************************************************************************

        utc0 = time_frames[BSR_campaign_id][BSR_i_id][0]
        utcf = time_frames[BSR_campaign_id][BSR_i_id][1]

        et0 = spice.utc2et(utc0)
        etf = spice.utc2et(utcf)
        et_interval = int(round(etf - et0) / time_step_int)
        time_step_float = (etf - et0)/et_interval
        meas_op = np.linspace(et0, etf, et_interval)
        midpoints_data = meas_op.copy()

        # ********************************************************************************************

        print("Calculate equiemission points [START]")

        equiemission_data = []
    
        for et in tqdm(meas_op):
            utc, dis_MEX2TGO, spoint, r_spoint2TGO, r_spoint2MEX, lon, lat, lat_sph, \
                emission_MEX, emission_TGO, lon_midpoint, lat_midpoint = find_equiemission(et)
            equiemission_data.append([utc, et,
                                    dis_MEX2TGO,
                                    spoint[0], spoint[1], spoint[2],
                                    spice.convrt(lon, 'RADIANS', 'DEGREES'),
                                    spice.convrt(lat_sph, 'RADIANS', 'DEGREES'),
                                    spice.convrt(lat, 'RADIANS', 'DEGREES'),
                                    emission_MEX, emission_TGO,
                                    spice.vnorm(r_spoint2MEX), spice.vnorm(r_spoint2TGO),
                                    spice.convrt(lon_midpoint, 'RADIANS', 'DEGREES'),
                                    spice.convrt(lat_midpoint, 'RADIANS', 'DEGREES')])
    
        print("Calculate equiemission points [END]")
   
        # ********************************************************************************************

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

        data_file.close()
        midpoint_names_file.write(data_file_name + " ")

        # *******************************************************************************************

        for i in range(len(equiemission_data)):
            et0_equi_data_i = equiemission_data[0][1]
            etf_equi_data_i = equiemission_data[-1][1]
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

            for j in range(len(equiemission_data)):      
                data_file.write(str(equiemission_data[j][0]) + ', ' \
                                + str(equiemission_data[j][1]) + ', ' \
                                + str(equiemission_data[j][2]) + ', ' \
                                + str(equiemission_data[j][3]) + ', ' \
                                + str(equiemission_data[j][4]) + ', ' \
                                + str(equiemission_data[j][5]) + ', ' \
                                + str(equiemission_data[j][6]) + ', ' \
                                + str(equiemission_data[j][7]) + ', ' \
                                + str(equiemission_data[j][8]) + ', ' \
                                + str(equiemission_data[j][9]) + ', ' \
                                + str(equiemission_data[j][10]) + ', ' \
                                + str(equiemission_data[j][11]) + ', ' \
                                + str(equiemission_data[j][12]) + ', ' \
                                + str(equiemission_data[j][13]) + ', ' \
                                + str(equiemission_data[j][14]) + '\n')

        data_file.close()
        meas_op_names_file.write(data_file_name + " ")

    # ********************************************************************************************

    midpoint_names_file.close()
    meas_op_names_file.close()

    # *******************************************************************************************

    plot_spoints(utc0, utcf, data_directory_path, usage, MTP_i, BSR_campaign)

# ************************************************************************************************

end_time = time.time()
end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

run_time_s = end_time - start_time
run_time_m = run_time_s / 60

print("Script ended at:", end_time_str)

if run_time_s < 100:
    print("Total runtime:", round(run_time_s,2), "seconds\n")
else:
    print("Total runtime:", round(run_time_m,2), "minutes\n")

# ************************************************************************************************

spice.kclear()

# ************************************************************************************************
