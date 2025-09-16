# **********************************************************************************************
#
#  my_functions.py
#
# **********************************************************************************************
#
#  Description: 
#  |  Functions used for the MEX-TGO BSR measurement planning and analysis
#     |  MEX and TGO antenna patterns
#     |  Creating visuals
#     |  Mapping ground tracks
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
import distinctipy
from matplotlib import rcParams
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from scipy.optimize import curve_fit

# **********************************************************************************************

""" Functions for the MEX and TGO antenna patterns """

def gaussian(x, A, B, C):
    """
    gaussian: Calculate Gaussian function for curve fitting
    """
    return A * np.exp(-B * x**2) + C

def calc_gain_gaussian(SC_offbs_data_angles, SC_offbs_data_gains):
    """
    calc_gain_gaussian: Calculate the Gaussian parameters for antenna pattern fitting

    :param SC_offbs_data_angles: Off-boresight angles in degrees
    :type SC_offbs_data_angles: class 'list'
    :param SC_offbs_data_gains: Off-boresight gains in dB
    :type SC_offbs_data_gains: class 'list'
    """

    gaussian_parameters, _ = curve_fit(gaussian, SC_offbs_data_angles, SC_offbs_data_gains,
                                       p0=[np.max(SC_offbs_data_gains), 0.001,
                                           np.min(SC_offbs_data_gains)], maxfev=5000)
    return gaussian_parameters

# **********************************************************************************************

""" Functions for creating visuals """

def create_gif(img_dir_path, img_name_end, frame_duration_ms, gif_path):
    """
    create_gif Create a GIF from a series of images

    :param img_dir_path: Path to the directory containing the images
    :type img_dir_path: class 'str'
    :param img_name_end: The ending of the image filenames to be included
    :type img_name_end: class 'str'
    :param frame_duration_ms: Duration of each frame in milliseconds
    :type frame_duration_ms: class 'int'
    :param gif_path: Path to save the generated GIF
    :type gif_path: class 'str'
    """

    print(f"Generating GIF for images in {img_dir_path}, ending with '{img_name_end}'...")

    img_files = sorted([
        f for f in os.listdir(img_dir_path)
        if f.endswith((f"{img_name_end}.png"))
    ])

    imgs = []
    for f in img_files:
        try:
            img = Image.open(os.path.join(img_dir_path, f))
            img.load()  # Force loading to catch truncated images
            imgs.append(img)
        except Exception as e:
            print(f"Warning: Skipping file '{f}' due to error: {e}")

    if not imgs:
        print("No valid images found to create GIF.")
        return

    if os.path.exists(gif_path):
        os.remove(gif_path)

    imgs[0].save(gif_path, save_all=True,
        append_images=imgs[1:],
        duration=frame_duration_ms,
        loop=0
    )

    print(f"Generated GIF saved to {gif_path}")

    return

def plot_antenna_pattern(SC_offbs_data_angles, SC_offbs_data_gains, only_positive, SC_name, SC_color):
    """
    plot_antenna_pattern Plot the antenna pattern of a spacecraft

    :param SC_offbs_data_angles: Off-boresight angles in degrees
    :type SC_offbs_data_angles: class 'list'
    :param SC_offbs_data_gains: Off-boresight gains in dB
    :type SC_offbs_data_gains: class 'list'
    :param only_positive: Boolean for deciding whether to only plot positive angles
    :type only_positive: class 'bool'
    :param SC_name: Name of the spacecraft
    :type SC_name: class 'str'
    :param SC_color: Color for the fitted plot, data points will be a darker version of this color
    :type SC_color: class 'str'
    """
    
    plot_angles = np.arange(-90, 91, 1)
    plot_xticks = np.arange(-100, 100, 10)

    if only_positive:
        plot_angles = plot_angles[plot_angles >= 0]
        plot_xticks = plot_xticks[plot_xticks >= 0]
        data_idx = [i for i, a in enumerate(SC_offbs_data_angles) if a >= 0]
    else:
        data_idx = range(len(SC_offbs_data_angles))

    gaussian_params = calc_gain_gaussian(SC_offbs_data_angles, SC_offbs_data_gains)
    plot_gains = gaussian(plot_angles, *gaussian_params)

    SC_rgb_color = mcolors.to_rgb(SC_color)
    SC_rgb_darker_color = tuple(np.clip(np.array(SC_rgb_color) * 0.6, 0, 1))

    plt.plot(plot_angles, plot_gains, color=SC_rgb_color, label='Gaussian fit')
    plt.scatter(np.array(SC_offbs_data_angles)[data_idx], np.array(SC_offbs_data_gains)[data_idx],
                color=SC_rgb_darker_color, label='Data points')

    plt.subplots_adjust(left=0.1, right=0.99, top=0.94, bottom=0.1)
    plt.grid()
    plt.title(f"{SC_name} antenna gain pattern")
    plt.xlabel('Off boresight angle [deg]')
    plt.ylabel('Antenna gain [dB]')
    plt.xticks() 
    plt.legend()
    plt.show()

    return

# **********************************************************************************************

""" Functions for mapping ground tracks """

def plot_STLs():
    """
    plot_STLs: Plot the Scientific Target Locations on Mars, with the possibility to include 
        past and future landing sites
    """
    marker_size = 200
    marker_size_LS = 300

    # Latitudes are defined North positive; -90 to 90 degrees
    latitudes = {'Viking1': 22.24,
                'Viking2': 47.64,
                'Pathfinder': 19.13,
                'Spirit': -14.57,
                'Opportunity': -1.95,
                'Phoenix': 68.22,
                'Curiosity': -4.59,
                'InSight': 4.50,
                'Tianwen1': 25.07,
                'Perseverance': 18.44,
                'ExoMars2028': 18.28}
    latitudes_craters = {'S1094bCrater': 35.1,
                         'MROCrater': 43.28}

    contour_latitudes = {'VallesMarineris': [0, 0, 5, 5, -5, -5, 10, 10, 5, -10, -15, -15, -20, -20, -10, -10, 0],
                        'ChrysePlanitia': [45, 35, 20, 10, 10, 15, 35, 40, 45, 45],
                        'XantheTerra': [0, -5, -5, 10, 15, 0],
                        'UtopiaPlanitia': [60, 60, 50, 35, 25, 15, 10, 15, 25, 35, 50, 60],
                        'ElysiumPlanitia': [-15, -20, -20, 0, 10, 15, 10, 10, 25, 30, 35, 25, -15],
                        'MFF': [-5, -10, -10, -5, 0, 5, 5, 0, -5]}

    # Longitudes are defined East positive; -180 to 180 degrees
    longitudes = {'Viking1': -48.01,
                'Viking2': 134.29,
                'Pathfinder': -33.22,
                'Spirit': 175.47,
                'Opportunity': -5.53,
                'Phoenix': -125.75,
                'Curiosity': 137.44,
                'InSight': 135.62,
                'Tianwen1': 109.93,
                'Perseverance': 77.45,
                'ExoMars2028': -24.63}
    
    longitudes_craters = {'S1094bCrater': -170.20,
                          'MROCrater': 164.22}

    contour_longitudes = {'VallesMarineris': [-100, -85, -80, -70, -60, -50, -45, -30, -25, -25, -30, -40, -40, -60, -85, -100, -100],
                        'ChrysePlanitia': [-45, -55, -55, -45, -25, -25, -10, -10, -15, -45],
                        'XantheTerra': [-65, -60, -50, -45, -50, -65],
                        'UtopiaPlanitia': [125, 90, 75, 75, 90, 100, 120, 135, 135, 145, 145, 125],
                        'ElysiumPlanitia': [180, 175, 160, 120, 120, 135, 145, 150, 160, 160, 165, 180, 180],
                        'MFF': [-155, -160, -165, -170, -170, -165, -160, -155, -155]}

    # plt.scatter(longitudes['Viking1'], latitudes['Viking1'], label='Viking 1 (1975)', marker='*', s=marker_size_LS, edgecolors='black')
    # plt.scatter(longitudes['Viking2'], latitudes['Viking2'], label='Viking 2 (1976)', marker='*', s=marker_size_LS, edgecolors='black')
    # plt.scatter(longitudes['Pathfinder'], latitudes['Pathfinder'], label='Pathfinder (1996)', marker='*', s=marker_size_LS, edgecolors='black')
    # plt.scatter(longitudes['Spirit'], latitudes['Spirit'], label='Spirit (2003)', marker='*', s=marker_size_LS, edgecolors='black')
    # plt.scatter(longitudes['Opportunity'], latitudes['Opportunity'], label='Opportunity (2003)', marker='*', s=marker_size_LS, edgecolors='black')
    # plt.scatter(longitudes['Phoenix'], latitudes['Phoenix'], label='Phoenix (2007)', marker='*', s=marker_size_LS, edgecolors='black')
    # plt.scatter(longitudes['Curiosity'], latitudes['Curiosity'], label='Curiosity (2011)', marker='*', s=marker_size_LS, edgecolors='black')
    # plt.scatter(longitudes['InSight'], latitudes['InSight'], label='InSight (2018)', marker='*', s=marker_size_LS, edgecolors='black')
    # plt.scatter(longitudes['Tianwen1'], latitudes['Tianwen1'], label='Tianwen-1 (2020)', marker='*', s=marker_size_LS, edgecolors='black')
    # plt.scatter(longitudes['Perseverance'], latitudes['Perseverance'], label='Perseverance (2020)', marker='*', s=marker_size_LS, edgecolors='black')
    # plt.scatter(longitudes['ExoMars2028'], latitudes['ExoMars2028'], label='ExoMars 2018 (Future)', marker='*', s=marker_size_LS, edgecolors='black')

    """ Plot calibration STL locations """
    plt.scatter(longitudes_craters['S1094bCrater'], latitudes_craters['S1094bCrater'], color ='royalblue', marker='$\odot$', s=marker_size, alpha=0.7)
    plt.annotate('S1094b Crater \n (STL-C-M1)', (longitudes_craters['S1094bCrater'], latitudes_craters['S1094bCrater']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='royalblue', fontweight='bold', alpha=0.7)
    plt.scatter(longitudes_craters['MROCrater'], latitudes_craters['MROCrater'], color ='orange', marker='$\odot$', s=marker_size, alpha=0.7)
    plt.annotate('MRO Impact Crater \n (STL-C-B1)', (longitudes_craters['MROCrater'], latitudes_craters['MROCrater']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='orange', fontweight='bold', alpha=0.7)

    """ Plot backup scientific STL locations """
    plt.plot(contour_longitudes['MFF'], contour_latitudes['MFF'], color='orange', linestyle='-', alpha=0.7)
    plt.annotate('Medusae Fossae \n Formation \n (STL-S-B1)', (-162.5, 5), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color='orange', fontweight='bold', alpha=0.7)
    plt.plot(contour_longitudes['ChrysePlanitia'], contour_latitudes['ChrysePlanitia'], color='orange', linestyle='-', alpha=0.7)
    plt.annotate('Chryse Planitia \n (STL-S-B2)', (-30, 45), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color='orange', fontweight='bold', alpha=0.7)
    plt.plot(contour_longitudes['XantheTerra'], contour_latitudes['XantheTerra'], color='orange', linestyle='-', alpha=0.7)
    plt.annotate('Xanthe Terra \n (STL-S-B3)', (-60, 5), textcoords="offset points", xytext=(0,5), ha='right', fontsize=8, color='orange', fontweight='bold', alpha=0.7)

    """ Plot main scientific STL locations """
    plt.plot(contour_longitudes['UtopiaPlanitia'], contour_latitudes['UtopiaPlanitia'], color='royalblue', linestyle='-', alpha=0.7)
    plt.annotate('Utopia Planitia \n (STL-S-M1)', (107.5, 60), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color='royalblue', fontweight='bold', alpha=0.7)
    plt.plot(contour_longitudes['ElysiumPlanitia'], contour_latitudes['ElysiumPlanitia'], color='royalblue', linestyle='-', alpha=0.7)
    plt.annotate('Elysium Planitia \n (STL-S-M2)', (140, -10), textcoords="offset points", xytext=(0,-15), ha='right', fontsize=8, color='royalblue', fontweight='bold', alpha=0.7)
    plt.plot(contour_longitudes['VallesMarineris'], contour_latitudes['VallesMarineris'], color='royalblue', linestyle='-', alpha=0.7)
    plt.annotate('Valles Marineris \n (STL-S-M3)', (-70, -15), textcoords="offset points", xytext=(0,-17.5), ha='right', fontsize=8, color='royalblue', fontweight='bold', alpha=0.7)

    """ Plot boundaries of the near-equatorial band of interest """
    plt.hlines(-40, -180, 180, colors='red', linestyles='--', linewidth=1)
    plt.hlines(40, -180, 180, colors='red', linestyles='--', linewidth=1)
    return

def plot_tracks(directory_path, plot_type, midpoints_list, meas_ops_list, colours, naming, MTP, BSR_i):
    """
    plot_tracks: Plot tracks of the specular points and midpoints from CSV files

    :param directory_path: Path to the directory containing the CSV files
    :type directory_path: class 'str'
    :param plot_type: Desired plot type: '_all' (both specular and midpoints), '_midpoints' 
        (only midpoints) or '_meas_ops' (only specular points)
    :type plot_type: class 'str'
    :param midpoints_list: List of midpoint CSV file names
    :type midpoints_list: class 'list'
    :param meas_ops_list: List of measurement opportunity, i.e. specular point, CSV file names
    :type meas_ops_list: class 'list'
    :param colours: List of colors for plotting the different tracks   
    :type colours: class 'list'
    :param naming: Designator for the title, either 'BSR' for (to be) performed measurements
        or 'MOP' for proposals
    :type naming: class 'str'
    :param MTP: Title of Medium Term Plan 
    :type MTP: class 'str'
    :param BSR_i: Title of BSR compaign
    :type BSR_i: class 'str'
    """
    marker_size = 10
    counter = 0

    col_counter = len(meas_ops_list)

    if plot_type == '_all' or plot_type == '_midpoints':
        for i in midpoints_list:
            counter += 1
            label_i = i.replace("-", " - ").replace(".csv", "").replace("midpoint_", "midpoint-" + BSR_i + "." + str(counter) + ": ")
            midpoint_path = os.path.join(directory_path, i)
            midpoint_lon = pd.read_csv(midpoint_path, usecols=[6])
            midpoint_lat = pd.read_csv(midpoint_path, usecols=[7])
            plt.scatter(midpoint_lon, midpoint_lat, label=label_i, s=marker_size, color=colours[col_counter])
            col_counter += 1
    
    col_counter = 0
    
    if plot_type == '_all' or plot_type == '_meas_ops':
        MOP_counter = 1
        for i in meas_ops_list:
            if naming == 'BSR':
                label_i = i.replace("-", " - ").replace(".csv", "").replace("meas_op_", "BSR-" + BSR_i + "." + str(MOP_counter) + ": ")
            if naming == 'MOP':
                label_i = i.replace("-", " - ").replace(".csv", "").replace("meas_op_", "meas_op-" + BSR_i + "." + str(MOP_counter) + ": ")
            MOP_counter += 1
            meas_op_path = os.path.join(directory_path, i)
            meas_op_lon = pd.read_csv(meas_op_path, usecols=[6])
            meas_op_lat = pd.read_csv(meas_op_path, usecols=[8])
            plt.scatter(meas_op_lon, meas_op_lat, label=label_i, s=marker_size, color=colours[col_counter])
            col_counter += 1
    return

def set_plot_info(ax_MarsSpoints):
    """
    set_plot_info: Setting plot information, such as ticks and axes labels

    :param ax_MarsSpoints: Axes object of the plot
    :type ax_MarsSpoints: class 'matplotlib.axes._axes.Axes'
    """
    lon_major_ticks = np.arange(-180, 181, 30)
    lon_minor_ticks = np.arange(-180, 181, 5)
    lat_major_ticks = np.arange(-90, 91, 30)
    lat_minor_ticks = np.arange(-90, 91, 5)

    ax_MarsSpoints.set_xticks(lon_major_ticks)
    ax_MarsSpoints.set_xticks(lon_minor_ticks, minor=True)
    ax_MarsSpoints.set_yticks(lat_major_ticks)
    ax_MarsSpoints.set_yticks(lat_minor_ticks, minor=True)

    ax_MarsSpoints.tick_params(axis='both', which='major', labelsize=10)

    ax_MarsSpoints.grid(which='minor', alpha=0.2, color='black', linestyle='-', linewidth=0.3)
    ax_MarsSpoints.grid(which='major', alpha=0.7, color='black', linestyle='-', linewidth=0.3)

    ax_MarsSpoints.set_axisbelow(True)

    box_MarsSpoints = ax_MarsSpoints.get_position()
    ax_MarsSpoints.set_position([box_MarsSpoints.x0, box_MarsSpoints.y0, box_MarsSpoints.width * 0.8, box_MarsSpoints.height])
    ax_MarsSpoints.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_MarsSpoints.set_xlabel('Longitude [$\degree$E]')
    ax_MarsSpoints.set_ylabel('Latitude [$\degree$N]')
    return

def plot_spoints(utc0, utcf, directory_name, naming, MTP, BSR_i):
    """
    plot_spoints: Plot the midpoints and specular point tracks on a Mars background

    :param utc0: Start time of the BSR campaign in UTC format 'YYYY-MM-DD HH:MM:SS'
    :type utc0: class 'str'
    :param utcf: End time of the BSR campaign in UTC format 'YYYY-MM-DD HH:MM:SS'
    :type utcf: class 'str'
    :param directory_name: Name of the directory containing the CSV files
    :type directory_name: class 'str'
    :param naming: Designator for the title, either 'BSR' for (to be) performed measurements
        or 'MOP' for proposals
    :type naming: class 'str'
    :param MTP: Title of Medium Term Plan 
    :type MTP: class 'str'
    :param BSR_i: Title of BSR compaign
    :type BSR_i: class 'str'
    """
    
    rcParams['font.size'] = 14
    rcParams['axes.prop_cycle'] = cycler(color=plt.cm.tab20.colors)
    code_and_sim_path = os.path.join(os.path.dirname(os.getcwd()),
                                "thesis\\code_and_simulations")
    img_MarsBG_Path = os.path.join(code_and_sim_path,
                                "data\\HRSC-MOLA_Blended-DEM\\RecBG_dem_Mars_1500mp.png")
    img_MarsBackground = plt.imread(img_MarsBG_Path)

    """ Plot spoints from equiemission csv files """
    directory_path = os.path.join(os.path.dirname(os.getcwd()), directory_name)

    """ Create final plot on Mars background """
    midpoints_txt_path = os.path.join(directory_path, 'midpoints.txt')
    midpoints_txt = open(midpoints_txt_path, 'r')
    midpoints_lines = midpoints_txt.readlines()
    midpoints_txt.close()
    midpoints_list = midpoints_lines[0].split()
    N_midpoints = len(midpoints_list)

    meas_ops_txt_path = os.path.join(directory_path, 'meas_ops.txt')
    meas_ops_txt = open(meas_ops_txt_path, 'r')
    meas_ops_lines = meas_ops_txt.readlines()
    meas_ops_txt.close()
    meas_ops_list = meas_ops_lines[0].split()    
    N_meas_ops = len(meas_ops_list)

    N_colours = N_midpoints + N_meas_ops
    colours = distinctipy.get_colors(N_colours)

    for plot in ['_all', '_midpoints', '_meas_ops']:
        fig_MarsSpoints = plt.figure(figsize=(12, 10))
        fig_MarsSpoints.set_size_inches(19.2, 10.8)  # Nominal full screen: 1920x1080
        plt.imshow(img_MarsBackground, extent=[-180, 180, -90, 90], aspect='equal')
        
        plot_STLs()
        plot_tracks(directory_path, plot, midpoints_list, meas_ops_list, colours, naming, MTP, BSR_i)

        ax_MarsSpoints = plt.gca()
        if naming == 'BSR':
            title = 'Ground tracks for MEX-TGO BSR (BSR-' + BSR_i + '.x; ' + MTP + ')'
        if naming == 'MOP':
            title = 'Ground tracks for MEX-TGO BSR (meas_ops-' + BSR_i + '.x; ' + MTP + ')'
        fig_MarsSpoints.text(s=title, x=0.42, y=0.72+0.09, fontsize=18, ha='center', va='center', fontweight='bold')
        fig_MarsSpoints.text(s='(Background: HRSC_MOLA Blended DEM at 1500mp)', x=0.42, y=0.695+0.09, fontsize=12, ha='center', va='center')

        set_plot_info(ax_MarsSpoints)

        fig_name = 'spoints' + plot + '.png'
        fig_path = os.path.join(directory_path, fig_name)
        fig_MarsSpoints.savefig(fig_path, bbox_inches='tight', pad_inches=0.05, dpi=150)
        plt.close(fig_MarsSpoints)

    return

# *********************************************************************************************