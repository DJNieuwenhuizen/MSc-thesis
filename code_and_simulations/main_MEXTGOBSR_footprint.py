# **********************************************************************************************
#
#  main_MEXTGOBSR_footprint.py
#
# **********************************************************************************************
#
#  Description: 
#  |  Code used to calculate mutual signal footprints between MEX and TGO
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
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import spiceypy as spice
import time
from tqdm import tqdm

from my_functions import create_gif
from my_parameters import Mars_param

# **********************************************************************************************

""" Data file(s) to be analyzed """

planning_files = []
planning_path = os.path.join(os.path.dirname(os.getcwd()), "thesis\code_and_simulations\\analysis")
planning_txt_path = os.path.join(planning_path, 'spoint_tracks\\meas_ops.txt')
planning_txt = open(planning_txt_path, 'r')
planning_txt_lines = planning_txt.readlines()
planning_txt.close()
planning_txt_list = planning_txt_lines[0].split()
for i in range(len(planning_txt_list)):
    planning_dir_path = os.path.join(planning_path, 'spoint_tracks\\meas_ops')
    planning_file_path = os.path.join(planning_dir_path, planning_txt_list[i])
    planning_files.append(planning_file_path)

BSR_titles = ['BSR-1.1', 'BSR-1.2', 'BSR-1.3', 'BSR-1.4', 'BSR-2.1', 'BSR-2.2', 'BSR-3.1', 'BSR-4.1']

results_path = os.path.join(planning_path, "results")

# **********************************************************************************************

""" Load SPICE kernels """
MetaKernelsPath = os.path.join(os.path.dirname(os.getcwd()),
                               "thesis\\code_and_simulations\\kernels\\mk")

main_gen_mk = os.path.join(MetaKernelsPath, 'gen_mk.tm') # General Solar System/Time/Mars kernels
main_MEX_mk = os.path.join(MetaKernelsPath, 'MEX_mk.tm') # MEX-specific kernels
main_TGO_mk = os.path.join(MetaKernelsPath, 'TGO_mk.tm') # TGO-specific kernels

spice.furnsh(main_gen_mk)
spice.furnsh(main_MEX_mk)
spice.furnsh(main_TGO_mk)

# **********************************************************************************************

""" Mars geometry and local variables """

radii_ell = Mars_param['radii_ell'] # km; ellipsoidal Mars radii

# **********************************************************************************************

""" Functions """

def illuminated_area(r_sat, r_satbs):
    """
    illuminated_area: This function calculates the illuminated area as seen from a S/C
        on a planetary body (approximated as an ellipsoid)

    :param r_sat: Cartesian position vector of the S/C in IAU_Mars
    :type r_sat: class 'numpy.ndarray'
    :param r_satbs: Cartesian vector from the S/C to its boresight, i.e. nadir point, 
        on the planetary body
    :type r_satbs: class 'numpy.ndarray'

    :return: intpoint_lon: List of longitudes [deg] of the intersection points which bound
        the illuminated area / footprint
    :rtype: class 'list'
    :return: intpoint_lat: List of latitudes [deg] of the intersection points which bound
        the illuminated area / footprint
    :rtype: class 'list'
    :return: intpoint_angle: List of off-boresight angles [deg] of the intersection points
        which bound the illuminated area / footprint
    :rtype: class 'list'
    """
    intpoint_lon = []
    intpoint_lat = []
    intpoint_angle = []

    limb = spice.edlimb(radii_ell[0], radii_ell[1], radii_ell[2], r_sat)

    center = limb.center
    semi_major = limb.semi_major
    semi_minor = limb.semi_minor

    limb = spice.cgv2el(center, semi_major, semi_minor)
    
    steps = np.linspace(-np.pi, np.pi, 360)

    for t in steps:
        x = center + np.cos(t) * semi_major + np.sin(t) * semi_minor
        x_lon, x_lat = spice.reclat(x)[1:]
        intpoint_lon.append(spice.convrt(x_lon, 'RADIANS', 'DEGREES'))
        intpoint_lat.append(spice.convrt(x_lat, 'RADIANS', 'DEGREES'))

        r_satx = x - r_sat
        x_angle = spice.vsep(r_satbs, r_satx)
        x_angle_deg = spice.convrt(x_angle, 'RADIANS', 'DEGREES')
        intpoint_angle.append(x_angle_deg)
    
    return intpoint_lon, intpoint_lat, intpoint_angle

# **********************************************************************************************

""" Main script/loop """

for i in range(len(planning_files)):
    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    print("\nProcessing file:", planning_txt_list[i])
    print("Script started at:", start_time_str)
    
    dir_path = os.path.join(results_path, BSR_titles[i])
    os.makedirs(dir_path, exist_ok=True)
    subdir_path = os.path.join(dir_path, 'footprints')
    os.makedirs(subdir_path, exist_ok=True)

    # ******************************************************************************************

    """ Readout data """

    planning_file_path = planning_files[i]    
    planning_dataframe = pd.read_csv(planning_file_path, header=0)
    planning_dataframe.columns = ['utc'] + [col.split()[0].strip() for col in planning_dataframe.columns[1:]]
    planning_data = {col: planning_dataframe[col].values for col in planning_dataframe.columns}
    et_steps = planning_data['et'][:]

    spoint_track = [[],[]]
    midpoint_track = [[],[]]
    MEXbs_track = [[],[]]
    TGObs_track = [[],[]]
    MEX_intpoint_track = []
    TGO_intpoint_track = []
    r_MEX_track = []
    r_TGO_track = []

    # ******************************************************************************************

    """ Setup new data files """

    utc0 = planning_data['utc'][0]
    utcf = planning_data['utc'][-1]

    for j in range(len(et_steps)):

        utc = planning_data['utc'][j]
        et = planning_data['et'][j]
        spoint_lon_deg = planning_data['spoint_lon'][j]
        spoint_lat_deg = planning_data['spoint_lat'][j]
        lon_midpoint = planning_data['lon_midpoint'][j]
        lat_midpoint = planning_data['lat_midpoint'][j]

        # **************************************************************************************
        
        """ Setup new data files """

        footprint_file_name = 'footprint_' + utc.replace(" ", ".").replace(":", ".") + '.csv'
        footprint_file_dirpath = os.path.join(subdir_path, "data")
        os.makedirs(footprint_file_dirpath, exist_ok=True)
        footprint_file_path = os.path.join(footprint_file_dirpath, footprint_file_name)
        footprint_file = open(footprint_file_path, 'w')

        visuals_file_dirpath = os.path.join(subdir_path, "visuals")
        os.makedirs(visuals_file_dirpath, exist_ok=True)

        footprint_file.write(f"# MEX_intpoint_lon [deg], MEX_intpoint_lat [deg], MEX_intpoint_angle [deg], TGO_intpoint_lon [deg], TGO_intpoint_lat [deg], TGO_intpoint_angle [deg] \n")

        # ***************************************************************************************

        """ Measurement geometry """
       
        spoint_lon = spice.convrt(spoint_lon_deg, 'DEGREES', 'RADIANS') # radians
        spoint_lat = spice.convrt(spoint_lat_deg, 'DEGREES', 'RADIANS') # radians
        r_spoint = spice.latsrf('ELLIPSOID', 'MARS', et, 'IAU_MARS', [[spoint_lon, spoint_lat]])[0]
        r_MEX = spice.spkpos('MEX', et, 'IAU_MARS', 'LT+S', 'MARS')[0] # m; MEX position
        r_TGO = spice.spkpos('TGO', et, 'IAU_MARS', 'LT+S', 'MARS')[0] # m; TGO position

        r_spointMEX = r_MEX - r_spoint # m; vector from specular point to MEX
        r_spointTGO = r_TGO - r_spoint # m; vector from specular point to TGO

        r_MEXspoint = r_spoint - r_MEX # m; vector from MEX to the specular point
        r_TGOspoint = r_spoint - r_TGO # m; vector from TGO to the specular point
        R_T = spice.vnorm(r_spointMEX) # m; distance from MEX to the specular point
        R_R = spice.vnorm(r_spointTGO) # m; distance from TGO to the specular point

        MEXbs_lon, MEXbs_lat = spice.reclat(r_MEX)[1:] # radians; MEX boresight in conventional attitude: nadir-pointing
        MEXbs_lon_deg = spice.convrt(MEXbs_lon, 'RADIANS', 'DEGREES') # degrees
        MEXbs_lat_deg = spice.convrt(MEXbs_lat, 'RADIANS', 'DEGREES') # degrees
        MEX_r_bs = spice.latsrf('ELLIPSOID', 'MARS', et, 'IAU_MARS', [[MEXbs_lon, MEXbs_lat]])[0]
        r_MEXbs = MEX_r_bs - r_MEX # m; vector from MEX to the boresight point

        TGObs_lon, TGObs_lat = spice.reclat(r_TGO)[1:] # radians; TGO boresight in conventional attitude: nadir-pointing
        TGObs_lon_deg = spice.convrt(TGObs_lon, 'RADIANS', 'DEGREES') # degrees
        TGObs_lat_deg = spice.convrt(TGObs_lat, 'RADIANS', 'DEGREES') # degrees
        TGO_r_bs = spice.latsrf('ELLIPSOID', 'MARS', et, 'IAU_MARS', [[TGObs_lon, TGObs_lat]])[0]
        r_TGObs = TGO_r_bs - r_TGO # m; vector from TGO to the boresight point

        r_MEX_track.append(r_MEX)
        r_TGO_track.append(r_TGO)
        spoint_track[0].append(spoint_lon_deg)
        spoint_track[1].append(spoint_lat_deg)
        midpoint_track[0].append(lon_midpoint)
        midpoint_track[1].append(lat_midpoint)
        MEXbs_track[0].append(MEXbs_lon_deg)
        MEXbs_track[1].append(MEXbs_lat_deg)
        TGObs_track[0].append(TGObs_lon_deg)
        TGObs_track[1].append(TGObs_lat_deg)

        # ***************************************************************************************
        
        """ Calculate antenna footprints """

        MEX_intpoint_lon, MEX_intpoint_lat, MEX_intpoint_angle = illuminated_area(r_MEX, r_MEXbs)
        MEX_intpoint_track.append([MEX_intpoint_lon, MEX_intpoint_lat])

        TGO_intpoint_lon, TGO_intpoint_lat, TGO_intpoint_angle = illuminated_area(r_TGO, r_TGObs)
        TGO_intpoint_track.append([TGO_intpoint_lon, TGO_intpoint_lat])

        # ***************************************************************************************

        """ Save data to file """
        for k in range(len(MEX_intpoint_lon)):
            footprint_file.write(f"{MEX_intpoint_lon[k]}, {MEX_intpoint_lat[k]}, {MEX_intpoint_angle[k]}, {TGO_intpoint_lon[k]}, {TGO_intpoint_lat[k]}, {TGO_intpoint_angle[k]} \n")
        footprint_file.close()
        
        # ***************************************************************************************
                                    
    """ Plot measurement geometry """

    marker_size = 1
    marker_size_j = 150

    rcParams['font.size'] = 14
    rcParams['axes.prop_cycle'] = cycler(color=plt.cm.tab20.colors)
    code_and_sim_path = os.path.join(os.path.dirname(os.getcwd()),
                                "thesis\\code_and_simulations")
    img_MarsBG_Path = os.path.join(code_and_sim_path,
                                   "data\\HRSC-MOLA_Blended-DEM\\RecBG_dem_Mars_1500mp.png")
    img_MarsBackground = plt.imread(img_MarsBG_Path)    
    print("\nPlotting resultant footprint tracks...")

    for j in tqdm(range(len(MEX_intpoint_track))):
        fig_MarsSpoints = plt.figure(figsize=(12, 10))
        fig_MarsSpoints.set_size_inches(19.2, 10.8)  # Nominal full screen: 1920x1080
        plt.imshow(img_MarsBackground, extent=[-180, 180, -90, 90], aspect='equal')
        
        # plt.scatter(midpoint_track[0], midpoint_track[1], label='midpoint track', s=marker_size, color='lightpink')
        plt.scatter(MEXbs_track[0], MEXbs_track[1], label='MEX_boresight track', s=marker_size, color='red')
        plt.scatter(TGObs_track[0], TGObs_track[1], label='TGO_boresight track', s=marker_size, color='cyan')
        plt.scatter(spoint_track[0], spoint_track[1], label='spoint track', s=marker_size, color='fuchsia')

        # plt.scatter(midpoint_track[0][j], midpoint_track[1][j], label='midpoint', marker='*', s=marker_size_j, color='lightpink', edgecolors='black')
        plt.scatter(MEXbs_track[0][j], MEXbs_track[1][j], label='MEX_boresight', marker='*', s=marker_size_j, color='red', edgecolors='black')
        plt.scatter(TGObs_track[0][j], TGObs_track[1][j], label='TGO_boresight', marker='*', s=marker_size_j, color='cyan', edgecolors='black')
        plt.scatter(spoint_track[0][j], spoint_track[1][j], label='spoint', marker='*', s=marker_size_j, color='fuchsia', edgecolors='black')

        MEX_k_split = 0
        for k in range(len(MEX_intpoint_track[j][0])-1):
            MEX_lon_diff = np.abs(MEX_intpoint_track[j][0][k+1] - MEX_intpoint_track[j][0][k])
            MEX_lat_diff = np.abs(MEX_intpoint_track[j][1][k+1] -  MEX_intpoint_track[j][1][k])

            if MEX_lon_diff > 180 or MEX_lat_diff > 90:
                plt.plot(MEX_intpoint_track[j][0][MEX_k_split:k+1], MEX_intpoint_track[j][1][MEX_k_split:k+1], color='red', alpha=0.5)
                MEX_k_split = k + 1

        plt.plot(MEX_intpoint_track[j][0][MEX_k_split:k+1], MEX_intpoint_track[j][1][MEX_k_split:k+1], color='red', alpha=0.5)
    
        TGO_k_split = 0
        for k in range(len(TGO_intpoint_track[j][0])-1):
            TGO_lon_diff = np.abs(TGO_intpoint_track[j][0][k+1] - TGO_intpoint_track[j][0][k])
            TGO_lat_diff = np.abs(TGO_intpoint_track[j][1][k+1] -  TGO_intpoint_track[j][1][k])

            if TGO_lon_diff > 180 or TGO_lat_diff > 90:
                plt.plot(TGO_intpoint_track[j][0][TGO_k_split:k+1], TGO_intpoint_track[j][1][TGO_k_split:k+1], color='cyan', alpha=0.5)
                TGO_k_split = k + 1

        plt.plot(TGO_intpoint_track[j][0][TGO_k_split:k+1], TGO_intpoint_track[j][1][TGO_k_split:k+1], color='cyan', alpha=0.5)

        ax_MarsSpoints = plt.gca()

        title = 'Mars spoint and boresight tracks for the MEX-TGO BSR measurements'
        fig_MarsSpoints.text(s=title, x=0.42, y=0.72+0.09, fontsize=18, ha='center', va='center')
        fig_MarsSpoints.text(s='(Background: HRSC_MOLA Blended DEM at 1500mp)', x=0.42, y=0.695+0.09, fontsize=12, ha='center', va='center')

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

        meas_name = (planning_txt_list[i].replace('meas_op_', '')).replace('.csv', '')
        utc = planning_data['utc'][j]
        utc_stripped = (utc.replace(' ', '.')).replace(':', '.')
        fig_name = utc_stripped + '_2D.png'
        fig_path = os.path.join(visuals_file_dirpath, fig_name)
        fig_MarsSpoints.savefig(fig_path, bbox_inches='tight', pad_inches=0.05, dpi=150)

        plt.close(fig_MarsSpoints)

        # ***************************************************************************************
        
        MEX_intpoint_track_cart = []
        TGO_intpoint_track_cart = []

        for k in range(len(MEX_intpoint_track[j][0])):
            MEX_intpoint_lon_rad = spice.convrt(MEX_intpoint_track[j][0][k], 'DEGREES', 'RADIANS')
            MEX_intpoint_lat_rad = spice.convrt(MEX_intpoint_track[j][1][k], 'DEGREES', 'RADIANS')
            MEX_intpoint_cart = spice.latsrf('ELLIPSOID', 'MARS', planning_data['et'][j], 'IAU_MARS', [[MEX_intpoint_lon_rad, MEX_intpoint_lat_rad]])[0]
            MEX_intpoint_track_cart.append(MEX_intpoint_cart)

        for k in range(len(TGO_intpoint_track[j][0])):
            TGO_intpoint_lon_rad = spice.convrt(TGO_intpoint_track[j][0][k], 'DEGREES', 'RADIANS')
            TGO_intpoint_lat_rad = spice.convrt(TGO_intpoint_track[j][1][k], 'DEGREES', 'RADIANS')
            TGO_intpoint_cart = spice.latsrf('ELLIPSOID', 'MARS', planning_data['et'][j], 'IAU_MARS', [[TGO_intpoint_lon_rad, TGO_intpoint_lat_rad]])[0]
            TGO_intpoint_track_cart.append(TGO_intpoint_cart)

        MEX_intpoint_track_cart = np.asarray(MEX_intpoint_track_cart)
        TGO_intpoint_track_cart = np.asarray(TGO_intpoint_track_cart)

        plot_lim_xplus = np.max([np.max([vec[0] for vec in r_MEX_track]), np.max([vec[0] for vec in r_TGO_track]), radii_ell[0]])
        plot_lim_xminus = np.min([np.min([vec[0] for vec in r_MEX_track]), np.min([vec[0] for vec in r_TGO_track]), -radii_ell[0]])
        plot_lim_yplus = np.max([np.max([vec[1] for vec in r_MEX_track]), np.max([vec[1] for vec in r_TGO_track]), radii_ell[1]])
        plot_lim_yminus = np.min([np.min([vec[1] for vec in r_MEX_track]), np.min([vec[1] for vec in r_TGO_track]), -radii_ell[1]])
        plot_lim_zplus = np.max([np.max([vec[2] for vec in r_MEX_track]), np.max([vec[2] for vec in r_TGO_track]), radii_ell[2]])
        plot_lim_zminus = np.min([np.min([vec[2] for vec in r_MEX_track]), np.min([vec[2] for vec in r_TGO_track]), -radii_ell[2]])

        aspect_x = plot_lim_xplus - plot_lim_xminus
        aspect_y = plot_lim_yplus - plot_lim_yminus
        aspect_z = plot_lim_zplus - plot_lim_zminus
        max_aspect = max(aspect_x, aspect_y, aspect_z)
        aspect_x /= max_aspect
        aspect_y /= max_aspect
        aspect_z /= max_aspect

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        u, v = np.meshgrid(u, v)
        x = radii_ell[0] * np.cos(u) * np.sin(v)
        y = radii_ell[1] * np.sin(u) * np.sin(v)
        z = radii_ell[2] * np.cos(v)

        fig_MarsSpoints_3D = plt.figure()
        ax_MarsSpoints_3D = fig_MarsSpoints_3D.add_subplot(111, projection='3d')

        ax_MarsSpoints_3D.plot_surface(x, y, z, color='orange', alpha=0.7, zorder=1)
        ax_MarsSpoints_3D.set_xlabel('x')
        ax_MarsSpoints_3D.set_ylabel('y')
        ax_MarsSpoints_3D.set_zlabel('z')

        ax_MarsSpoints_3D.set_xlim([plot_lim_xminus, plot_lim_xplus])
        ax_MarsSpoints_3D.set_ylim([plot_lim_yminus, plot_lim_yplus])
        ax_MarsSpoints_3D.set_zlim([plot_lim_zminus, plot_lim_zplus])

        ax_MarsSpoints_3D.set_box_aspect([aspect_x, aspect_y, aspect_z])

        ax_MarsSpoints_3D.tick_params(axis='z', pad=10)

        ax_MarsSpoints_3D.plot(MEX_intpoint_track_cart[:, 0], MEX_intpoint_track_cart[:, 1], MEX_intpoint_track_cart[:, 2], color='red', label='MEX footprint', linewidth=2, zorder=10)
        ax_MarsSpoints_3D.plot(TGO_intpoint_track_cart[:, 0], TGO_intpoint_track_cart[:, 1], TGO_intpoint_track_cart[:, 2], color='cyan', label='TGO footprint', linewidth=2, zorder=10)
        
        et = planning_data['et'][j]
        MEX_bs = spice.latsrf('ELLIPSOID', 'MARS', et, 'IAU_MARS', [[np.radians(MEXbs_track[0][j]), np.radians(MEXbs_track[1][j])]])[0]
        TGO_bs = spice.latsrf('ELLIPSOID', 'MARS', et, 'IAU_MARS', [[np.radians(TGObs_track[0][j]), np.radians(TGObs_track[1][j])]])[0]
        spoint = spice.latsrf('ELLIPSOID', 'MARS', et, 'IAU_MARS', [[np.radians(spoint_track[0][j]), np.radians(spoint_track[1][j])]])[0]

        ax_MarsSpoints_3D.plot([MEX_bs[0]], [MEX_bs[1]], [MEX_bs[2]], marker='*', color='red', markeredgecolor='black', markersize=7, linestyle='None', label='MEX boresight', zorder=100)
        ax_MarsSpoints_3D.plot(r_MEX_track[j][0], r_MEX_track[j][1], r_MEX_track[j][2], marker='o', color='red', markeredgecolor='black', markersize=5, linestyle='None', label='MEX position', zorder=101)
        ax_MarsSpoints_3D.plot([TGO_bs[0]], [TGO_bs[1]], [TGO_bs[2]], marker='*', color='cyan', markeredgecolor='black', markersize=7, linestyle='None', label='TGO boresight', zorder=102)
        ax_MarsSpoints_3D.plot(r_TGO_track[j][0], r_TGO_track[j][1], r_TGO_track[j][2], marker='o', color='cyan', markeredgecolor='black', markersize=5, linestyle='None', label='TGO position', zorder=103)
        ax_MarsSpoints_3D.plot([spoint[0]], [spoint[1]], [spoint[2]], marker='*', color='fuchsia', markeredgecolor='black', markersize=8, linestyle='None', label='Specular point', zorder=104)
        ax_MarsSpoints_3D.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fig_name = utc_stripped + '_3D.png'
        fig_path = os.path.join(visuals_file_dirpath, fig_name)
        # plt.show()
        fig_MarsSpoints_3D.savefig(fig_path, bbox_inches='tight', pad_inches=0.05, dpi=150)
        plt.close(fig_MarsSpoints_3D)

    # *******************************************************************************************

    gif_path_2D = os.path.join(subdir_path, 'animation_2D.gif')
    create_gif(visuals_file_dirpath, '2D', frame_duration_ms=30, gif_path=gif_path_2D)

    gif_path_3D = os.path.join(subdir_path, 'animation_3D.gif')
    create_gif(visuals_file_dirpath, '3D', frame_duration_ms=30, gif_path=gif_path_3D)

# ***********************************************************************************************

end_time = time.time()
end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

run_time_s = end_time - start_time
run_time_m = run_time_s / 60

print("Script ended at:", end_time_str)

if run_time_s < 100:
    print("Total runtime:", round(run_time_s,2), "seconds\n")
else:
    print("Total runtime:", round(run_time_m,2), "minutes\n")

# ***********************************************************************************************

spice.kclear()

# ***********************************************************************************************