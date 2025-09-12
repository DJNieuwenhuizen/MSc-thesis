# **********************************************************************************************
#
#  infer_slope_MOLA.py
#
# **********************************************************************************************
#
#  Description: 
#  |  Code used to calculate slopes from MOLA MEGDR data
#  |  Resolution options: 4 or 16 pixels per degree
#  |  Output: 
#     | CSV file '{n_pixels}pixel_slopes.csv' including:
#        | Center longitude, center latitude, slope, RMS slope and local surface normal
#     | PNG plots of MOLA MEGDR topography, and derived (RMS) slopes
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

import os
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import spiceypy as spice
import sympy as smp

from my_parameters import Mars_param

# **********************************************************************************************

""" Load SPICE kernels """

MetaKernelsPath = os.path.join(os.path.dirname(os.getcwd()), "thesis\\code_and_simulations\\kernels\\mk")
main_gen_mk = os.path.join(MetaKernelsPath, 'gen_mk.tm') # General Solar System/Time/Mars kernels
spice.furnsh(main_gen_mk)

# **********************************************************************************************

""" Setup MOLA MEGDR data directories and file names """

n_pixels_list = [4, 16]
subdir_names = ["meg004", "meg016"]
# megdr_file_names = ["megc90n000cb.img", "megc90n000eb.img"] # Uncomment to use for plotting counts
megdr_file_names = ["megt90n000cb.img", "megt90n000eb.img"]

megdr_dir_path = os.path.join(os.path.dirname(os.getcwd()), "MSc-thesis\\code_and_simulations\\data\\MOLA_MEGDR")
analysis_dir_path = os.path.join(os.path.dirname(os.getcwd()), "MSc-thesis\\code_and_simulations\\analysis\\MOLA")
os.makedirs(analysis_dir_path, exist_ok=True)

radii_ell = Mars_param['radii_ell'] # km; ellipsoidal Mars radii

# **********************************************************************************************

""" Readout MOLA MEGDR data """

n_pixels = 4 # Options: 4 or 16
area_discretisation = 1/n_pixels

print("\nData resolution:", n_pixels, "pixels per degree")

if n_pixels in n_pixels_list:
    n_pixels_index = n_pixels_list.index(n_pixels)
    subdir_name = subdir_names[n_pixels_index]

    lon_0, lon_f = -180, 180
    lat_0, lat_f = -90, 90

    megdr_file_name = megdr_file_names[n_pixels_index]
    megdr_file_path = os.path.join(megdr_dir_path, subdir_name, megdr_file_name)
    megdr_file = open(megdr_file_path, 'rb')

    n_latrows = np.abs(lat_f-lat_0)*n_pixels
    n_loncols = np.abs(lon_f-lon_0)*n_pixels

    # System reads in little-endian, but the file uses big-endian (>)
    # i2 is a 2-byte signed integer; hence 16bit signed integer
    megdr_dtype = '>i2'  # big-endian 2-byte signed integer

    megdr_data = np.fromfile(megdr_file, dtype=megdr_dtype, count=n_latrows * n_loncols)
    megdr_file.close()

    megdr_data = megdr_data.reshape((n_latrows, n_loncols))
    megdr_data = np.flipud(megdr_data)  # Flip along the latitude axis

    # Shift MEGDR data longitudes from [0, 360] to [-180, 180]
    mid_lon_index = n_loncols // 2
    megdr_data = np.hstack((megdr_data[:, mid_lon_index:], megdr_data[:, :mid_lon_index]))

    lon_data = np.arange(lon_0 + 1/(2*n_pixels), lon_f + 1/(2*n_pixels), 1/n_pixels)
    lat_data = np.arange(lat_0 + 1/(2*n_pixels), lat_f + 1/(2*n_pixels), 1/n_pixels)

    plot_lon_data, plot_lat_data = np.meshgrid(lon_data, lat_data)

    slope_file_name = f"{n_pixels}pixel_slopes.csv"
    slope_file_path = os.path.join(analysis_dir_path, slope_file_name)
    slope_file = open(slope_file_path, 'w')

    slope_file.write(f"# lon_i [deg], lat_i [deg], slope [deg], slope_RMS [deg], n_hat [-] \n")

    # ******************************************************************************************

    """ Calculate slopes """    

    slopes = np.zeros_like(megdr_data, dtype=float)
    slopes_RMS = np.zeros_like(megdr_data, dtype=float)
    n_hats = np.zeros_like(megdr_data, dtype=float)
    et = 0.00 # Not used for ELLIPSOID model, but required for spice.latsrf function

    pixel_size = 1 / n_pixels  # degrees per pixel
    pixel_size_rad = spice.convrt(pixel_size, 'DEGREES', 'RADIANS')  # radians per pixel

    a = radii_ell[0] * 1000 # semi-major axis in metres
    b = radii_ell[2] * 1000 # semi-minor axis in metres
    ecc = np.sqrt(a**2 - b**2)/(a)

    lat_data_rad = spice.convrt(lat_data, 'DEGREES', 'RADIANS')
    lon_data_rad = spice.convrt(lon_data, 'DEGREES', 'RADIANS')

    print("\nIntegrating over latitudes to calculate delta_y...")

    theta = smp.symbols('theta', real=True)
    func = smp.sqrt(1 - ecc**2 * (smp.sin(theta))**2)
    func_ints = [float(smp.integrate(func, (theta, lat_i - 0.5*pixel_size_rad, lat_i + 0.5*pixel_size_rad))) 
                 for lat_i in tqdm(lat_data_rad)]
    delta_y_arr = a * np.array(func_ints)

    x_i_arr = np.zeros(lat_data.shape)
    delta_x_arr = np.zeros(lat_data.shape)

    for i, lat_i in enumerate(lat_data_rad):
        # Use lon=0 for latsrf, since radius only depends on latitude for ellipsoid
        cart_i_km = spice.latsrf('ELLIPSOID', 'MARS', et, 'IAU_MARS', [[0, lat_i]])[0]
        r_i_km = spice.vnorm(cart_i_km)
        x_i_km = r_i_km * np.cos(lat_i)
        x_i = x_i_km * 1000
        x_i_arr[i] = x_i
        delta_x_arr[i] = np.abs((2 * np.pi * x_i)/(360 * n_pixels)) # metres

    print("\nWriting slopes to file...")

    for i_y in tqdm(range(len(lat_data_rad))): 
        # Assume planar 3x3 grid around each point for slope calculation
        # delta_x and delta_y are determined by the grid resolution and should be converted to metres
        # conversion is dependent on location due to cylindrical projection used for MOLA MEGDR data
        lat_i_deg = lat_data[i_y]
        lat_i = lat_data_rad[i_y]

        delta_y = delta_y_arr[i_y]  # metres
        delta_x_original = delta_x_arr[i_y]  # metres

        delta_x_multiplier = int(np.round(delta_y / delta_x_original))
        if delta_x_multiplier % 2 == 0:
            # Ensure delta_x_multiplier is odd for symmetry in slope calculation
            delta_x_multiplier += 1 if (delta_y / delta_x_original) > delta_x_multiplier else -1
        delta_x = delta_x_multiplier * delta_x_original

        delta_x_step = delta_x_multiplier

        for i_x in range(len(lon_data_rad)):  
            lon_i_deg = lon_data[i_x]
            lon_i = lon_data_rad[i_x]

            z_i = megdr_data[i_y, i_x]

            if i_y + 1 == len(lat_data):
                z_N = megdr_data[i_y, i_x-int(len(lat_data)/2)]
            else:
                z_N = megdr_data[i_y + 1, i_x]

            if i_x + delta_x_step >= len(lon_data):
                z_E = megdr_data[i_y, delta_x_step-1]
            else:
                z_E = megdr_data[i_y, i_x + delta_x_step]

            if i_y == 0:
                z_S = megdr_data[i_y, i_x-int(len(lat_data)/2)]
            else: 
                z_S = megdr_data[i_y - 1, i_x]

            z_W = megdr_data[i_y, i_x - delta_x_step]

            x_comp = (z_E - z_W)/(2*delta_x)
            y_comp = (z_N - z_S)/(2*delta_y)

            slope = np.sqrt(x_comp**2 + y_comp**2)
            slope_rad = np.arctan(slope)
            slope_deg = np.degrees(slope_rad)
            slopes[i_y, i_x] = slope_deg

            RMS_Ncomp = np.arctan((np.abs(z_i - z_N))/delta_y)
            RMS_Ecomp = np.arctan((np.abs(z_i - z_E))/delta_x)
            RMS_Scomp = np.arctan((np.abs(z_i - z_S))/delta_y)
            RMS_Wcomp = np.arctan((np.abs(z_i - z_W))/delta_x)
            slope_RMS_rad = np.sqrt((RMS_Ncomp**2 + RMS_Ecomp**2 + RMS_Scomp**2 + RMS_Wcomp**2)/4)
            slope_RMS_deg = np.degrees(slope_RMS_rad)
            slopes_RMS[i_y, i_x] = slope_RMS_deg
            
            """ Calculate n_hat """    
            
            r_i = spice.latsrf('ELLIPSOID', 'MARS', et, 'IAU_MARS', [[lon_i, lat_i]])[0]
            n_hat_ell = spice.surfnm(radii_ell[0], radii_ell[1], radii_ell[2], r_i)

            # the omega_matrix is defined as the fixed axis around which the rotation is performed
            # local plane is defined by local_east and local_north (perpendicular to n_hat_ell);
            # axis of rotation is the slope direction in this plane
            local_east = np.array([-np.sin(lon_i), np.cos(lon_i), 0]) # tangent to constant latitude, points to increasing longitude
            local_north = np.array([-np.cos(lon_i) * np.sin(lat_i), -np.sin(lon_i) * np.sin(lat_i), np.cos(lat_i)]) # tangent to constant longitude, points to increasing latitude

            slope_vec = x_comp * local_east + y_comp * local_north
            slope_vec_norm = spice.vnorm(slope_vec)

            if slope_vec_norm != 0:
                slope_vec_unit = slope_vec / slope_vec_norm
                
                matrix_I = np.eye(3)
                matrix_omega = np.array([[0, -slope_vec_unit[2], slope_vec_unit[1]], 
                                        [slope_vec_unit[2], 0, -slope_vec_unit[0]],
                                        [-slope_vec_unit[1], slope_vec_unit[0], 0]])
                rot_matrix = matrix_I + np.sin(slope_rad) * matrix_omega + (1 - np.cos(slope_rad)) * np.dot(matrix_omega, matrix_omega)

                n_hat = np.dot(rot_matrix, n_hat_ell)
                n_hat_diff = np.degrees(spice.vsep(n_hat, n_hat_ell))

            else:
                slope_vec_unit = np.zeros_like(slope_vec)
                n_hat = n_hat_ell
                n_hat_diff = np.degrees(spice.vsep(n_hat, n_hat_ell))
            
            if slope_deg > 80.0:
                print("\nWarning: Slope exceeds 80 degrees at lon:", lon_i_deg, "lat:", lat_i_deg, "slope:", slope_deg)
                print("delta_x_original:", delta_x_original, "delta_x:", delta_x, "delta_y:", delta_y)

            slope_file.write(f"{lon_i_deg}, {lat_i_deg}, {slope_deg}, {slope_RMS_deg}, {n_hat}\n")

    slope_file.close()

    # ******************************************************************************************

    """ Plot MOLA MEGDR data """

    print("\nExpected elevation range: -8206 to 21181 meters") # from 'Roughness and near-surface density of Mars from SHARADradar echoes', Campbell et. al 2013
    print("Elevation range:", np.min(megdr_data), "to", np.max(megdr_data), "meters\n") 

    megdr_fig_2D = plt.figure(figsize=(12, 6))
    megdr_img_2D = plt.imshow(megdr_data, extent=[lon_0, lon_f, lat_0, lat_f], origin='lower', cmap='terrain', aspect='auto')
    plt.xlabel('Longitude [$\degree$E]')
    plt.ylabel('Latitude [$\degree$N]')
    title_2D = f'MOLA MEGDR Topography (at {n_pixels} pixel/degree resolution)'
    megdr_fig_2D.text(s=title_2D, x=0.43, y=0.94, fontsize=14, ha='center', va='center', fontweight='bold')
    megdr_fig_2D.text(s=f'Data retrieved from {megdr_file_name}', x=0.43, y=0.905, fontsize=12, ha='center', va='center')  
    cbar = plt.colorbar(megdr_img_2D, label='Elevation [m]')
    plt.grid(True, color='black', alpha=0.1)
    fig_name_2D = f"{n_pixels}pixel_MEGDR_topography_2D.png"
    fig_path_2D = os.path.join(analysis_dir_path, fig_name_2D)
    megdr_fig_2D.savefig(fig_path_2D, bbox_inches='tight', pad_inches=0.05, dpi=150)
    # plt.show()
    plt.close(megdr_fig_2D)

    # megdr_fig_2D_counts = plt.figure(figsize=(12, 6))
    # megdr_img_2D_counts = plt.imshow(megdr_data, extent=[lon_0, lon_f, lat_0, lat_f], origin='lower', cmap='terrain', aspect='auto')
    # plt.xlabel('Longitude [$\degree$E]')
    # plt.ylabel('Latitude [$\degree$N]')
    # title_2D_counts = f'MOLA MEGDR Counts (at {n_pixels} pixel/degree resolution)'
    # megdr_fig_2D_counts.text(s=title_2D_counts, x=0.43, y=0.94, fontsize=14, ha='center', va='center', fontweight='bold')
    # megdr_fig_2D_counts.text(s=f'Data retrieved from {megdr_file_name}', x=0.43, y=0.905, fontsize=12, ha='center', va='center')  
    # cbar = plt.colorbar(megdr_img_2D_counts, label='Counts [-]')
    # plt.grid(True, color='black', alpha=0.1)
    # fig_name_2D_counts = str(n_pixels) + "pixel_MEGDR_counts_2D.png"
    # fig_path_2D_counts = os.path.join(analysis_dir_path, fig_name_2D_counts)
    # megdr_fig_2D_counts.savefig(fig_path_2D_counts, bbox_inches='tight', pad_inches=0.05, dpi=150)
    # # plt.show()
    # plt.close(megdr_fig_2D_counts)

    norm = LogNorm(vmin=max(slopes[slopes > 0].min(), 1e-2), vmax=slopes.max())
    custom_ticks = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    slopes_fig_2D = plt.figure(figsize=(12, 6))
    slopes_ax_2D = slopes_fig_2D.add_subplot(111)
    slopes_img_2D = slopes_ax_2D.imshow(slopes, extent=[lon_0, lon_f, lat_0, lat_f], origin='lower', cmap='terrain', aspect='auto', norm=norm)
    slopes_ax_2D.set_xlabel('Longitude [$\degree$E]')
    slopes_ax_2D.set_ylabel('Latitude [$\degree$N]')
    title_2D = f'MOLA MEGDR Slopes (at {n_pixels} pixel/degree resolution)'
    slopes_fig_2D.text(s=title_2D, x=0.43, y=0.94, fontsize=14, ha='center', va='center', fontweight='bold')
    slopes_fig_2D.text(s=f'At {round(2*max(delta_x_arr)*10**-3, 1)} km resulting spatial resolution; Data retrieved from {megdr_file_name}', x=0.43, y=0.905, fontsize=12, ha='center', va='center') 
    cbar = plt.colorbar(slopes_img_2D, label='Slope [deg]', ticks=custom_ticks)
    cbar.ax.set_yticklabels([str(tick) for tick in custom_ticks])
    plt.grid(True, color='black', alpha=0.1)
    fig_name_2D = f"{n_pixels}pixel_MEGDR_slopes_2D.png"
    fig_path_2D = os.path.join(analysis_dir_path, fig_name_2D)
    slopes_fig_2D.savefig(fig_path_2D, bbox_inches='tight', pad_inches=0.05, dpi=150)
    # plt.show()
    plt.close(slopes_fig_2D)

    norm = LogNorm(vmin=max(slopes_RMS[slopes_RMS > 0].min(), 1e-2), vmax=slopes_RMS.max())
    custom_ticks = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    slopes_RMS_fig_2D = plt.figure(figsize=(12, 6))
    slopes_RMS_ax_2D = slopes_RMS_fig_2D.add_subplot(111)
    slopes_RMS_img_2D = slopes_RMS_ax_2D.imshow(slopes_RMS, extent=[lon_0, lon_f, lat_0, lat_f], origin='lower', cmap='terrain', aspect='auto', norm=norm)
    slopes_RMS_ax_2D.set_xlabel('Longitude [$\degree$E]')
    slopes_RMS_ax_2D.set_ylabel('Latitude [$\degree$N]')
    title_2D = f'MOLA MEGDR RMS Slopes (at {n_pixels} pixel/degree resolution)'
    slopes_RMS_fig_2D.text(s=title_2D, x=0.43, y=0.94, fontsize=14, ha='center', va='center', fontweight='bold')
    slopes_RMS_fig_2D.text(s=f'At {round(max(delta_x_arr)*10**-3, 1)} km resulting spatial resolution; Data retrieved from {megdr_file_name}', x=0.43, y=0.905, fontsize=12, ha='center', va='center') 
    cbar = plt.colorbar(slopes_RMS_img_2D, label='Slope [deg]', ticks=custom_ticks)
    cbar.ax.set_yticklabels([str(tick) for tick in custom_ticks])
    plt.grid(True, color='black', alpha=0.1)
    fig_name_2D = f"{n_pixels}pixel_MEGDR_slopes_RMS_2D.png"
    fig_path_2D = os.path.join(analysis_dir_path, fig_name_2D)
    slopes_RMS_fig_2D.savefig(fig_path_2D, bbox_inches='tight', pad_inches=0.05, dpi=150)
    # plt.show()
    plt.close(slopes_RMS_fig_2D)

# **********************************************************************************************

spice.kclear()