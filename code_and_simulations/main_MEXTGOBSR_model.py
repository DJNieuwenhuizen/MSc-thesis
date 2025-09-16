# **********************************************************************************************
#
#  main_MEXTGOBSR_model.py
#
# **********************************************************************************************
#
#  Description: 
#  |  Main model for processing and simulating the MEX-TGO BSR measurements
#     |  Prior to running the model, the slope and footprints must be generated using:
#             - infer_slope_MOLA.py
#             - main_MEXTGOBSR_footprint.py
#     |  Outcomes are stored in the respective BSR-x.x folders, including:
#             - Power distribution over the gridpoints inside the footprint per time step
#             - High-level BSR data products over time
#  |  Main model ran with multiprocessing at mp.cpu_count() - 2 for 4 pixel/degree resolution
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
from functools import partial
import matplotlib
from matplotlib import rcParams
from matplotlib.colors import BoundaryNorm
from matplotlib.path import Path
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import root_scalar
import spiceypy as spice
import sympy as smp
import time
from tqdm import tqdm

from my_parameters import Mars_param, BSR_param
from my_functions import create_gif, gaussian, calc_gain_gaussian 

# **********************************************************************************************

""" Functions """

# Speed up code simplifications to be able to handle 16 pixel data:
def mars_latlon_to_cartesian(lon, lat, a, c):
    """
    mars_latlon_to_cartesian: Convert planetocentric to Cartesion coordinates, without needing
        to call upon the SPICE information system 

    :param lon: Planetocentric longitude in radians
    :type lon: class 'float'
    :param lat: Planetocentric latitude in radians
    :type lat: class 'float'
    :param a: Semi-major axis of Mars in km
    :type a: class 'float'
    :param c: Semi-minor axis of Mars in km
    :type c: class 'float'

    :return: Cartesian coordinates [x,y,z] in km
    :rtype: class 'np.ndarray'
    """
    e2 = 1 - (c**2 / a**2)
    lat_d = np.arctan(np.tan(lat) / (1 - e2))
    cos_lat = np.cos(lat_d)
    sin_lat = np.sin(lat_d)
    denom = np.sqrt((a * cos_lat)**2 + (c * sin_lat)**2)
    x = (a**2 * cos_lat * np.cos(lon)) / denom
    y = (a**2 * cos_lat * np.sin(lon)) / denom
    z = (c**2 * sin_lat) / denom

    return np.stack([x, y, z], axis=-1)

def vsep_vec(v1, v2):
    """
    vsep_vec: Calculate the angular separation between two vectors, without needing
        to call upon the SPICE information system 

    :param v1: First vector
    :type v1: class 'np.ndarray'
    :param v2: Second vector
    :type v2: class 'np.ndarray'

    :return: Angular separation in radians
    :rtype: class 'float'
    """

    v1n = v1 / np.linalg.norm(v1, axis=1)[:, None]
    v2n = v2 / np.linalg.norm(v2, axis=1)[:, None]
    
    return np.arccos(np.clip(np.sum(v1n * v2n, axis=1), -1.0, 1.0))

# Normal functions:
def readout_measop(measop_path):
    """
    readout_measop: Read out the measurement opportunity data from a CSV file

    :param measop_path: Path to the CSV file
    :type measop_path: class 'str'

    :return: Dictionary containing the measurement opportunity data
    :rtype: class 'dict'
    """

    measop_dataframe = pd.read_csv(measop_path, header=0)
    measop_dataframe.columns = ['utc'] + [col.split()[0].strip() for col in measop_dataframe.columns[1:]]
    measop_data = {col: measop_dataframe[col].values for col in measop_dataframe.columns}
    
    return measop_data

def calc_grid_surfacearea(grid_S, area_discretization, n_lat_options, n_lon_options):
    """
    calc_grid_surfacearea: Set up the grid of surface areas per gridpoint

    :param grid_S: Empty grid to be filled with surface areas
    :type grid_S: class 'np.ndarray'
    :param area_discretization: Area discretization in degrees per pixel
    :type area_discretization: class 'float'
    :param n_lat_options: Amount of latitudinal grid cells
    :type n_lat_options: class 'int'
    :param n_lon_options: Amount of longitudinal grid cells
    :type n_lon_options: class 'int'

    :return: grid_S: Grid of surface areas per gridpoint
    :rtype: class 'np.ndarray'
    """
    lat_options_borders = np.linspace(0, np.pi, n_lat_options+1) # [lat_min, lat_max] runs from 0 (S) to pi (N) for the integration
    surfacearea_file_path = os.path.join(analysis_path, f"surfaceareas/S_{area_discretization}deg.csv")

    if os.path.exists(surfacearea_file_path):
        print(f"\nReading out surface areas with an accuracy of {area_discretization} degrees in latitudinal direction...")
        surfacearea_dataframe = pd.read_csv(surfacearea_file_path, header=0)
        surfacearea_dataframe.columns = ['lat_index'] + [col.split()[0].strip() for col in surfacearea_dataframe.columns[1:]]
        surfacearea_data = {col: surfacearea_dataframe[col].values for col in surfacearea_dataframe.columns}

        for i in tqdm(range(len(lat_options_borders)-1)):
            lat_index = surfacearea_data['lat_index'][i]
            lat_min = surfacearea_data['lat_min'][i]
            lat_max = surfacearea_data['lat_max'][i]
            S_latcirc = surfacearea_data['S_latcirc'][i] # km^2

            if lat_index == i: 
                S_latlon_gridpoint = (S_latcirc / n_lon_options) # km^2
                grid_S[i,:] = S_latlon_gridpoint

            else:
                print("MAYDAY: Lat index mismatch! Expected", i, "but got", lat_index)

    else:
        print(f"\nCalculating surface area with an accuracy of {area_discretization} degrees in latitudinal direction...")

        os.makedirs(os.path.dirname(surfacearea_file_path), exist_ok=True)
        surfacearea_file = open(surfacearea_file_path, 'w')
        surfacearea_file.write("# lat_index, lat_min [rad], lat_max [rad], S_latcirc [km^2]\n")

        for i in tqdm(range(len(lat_options_borders)-1)):
            lat_min = lat_options_borders[i]
            lat_max = lat_options_borders[i+1]

            v = smp.symbols('v', real=True)
            func = smp.sqrt(radii_ell[0]**2 + radii_ell[2]**2 + (radii_ell[0]**2 - radii_ell[2]**2) * smp.cos(2*v)) * smp.sin(v)
            func_int = smp.integrate(func, (v, lat_min, lat_max))
            S_latcirc = float(((2 * np.pi * radii_ell[0]) / np.sqrt(2)) * func_int) # km^2

            surfacearea_file.write(f"{i},{lat_min},{lat_max},{S_latcirc}\n")

            S_latlon_gridpoint = (S_latcirc / n_lon_options) # km^2
            grid_S[i,:] = S_latlon_gridpoint

        surfacearea_file.close()
    
    return grid_S

def interpolate_grid(lon, lat, grid_values, area_discretization, method):
    """
    interpolate_grid: Interpolate a filled in grid to a different resolution

    :param lon: Longitudinal values of the gridpoints
    :type lon: class 'np.ndarray'
    :param lat: Latitudinal values of the gridpoints
    :type lat: class 'np.ndarray'
    :param grid_values: Original grid to be interpolated
    :type grid_values: class 'np.ndarray'
    :param area_discretization: Area discretization in degrees per pixel
    :type area_discretization: class 'float'
    :param method: Interpolation method to be used, must be available in 
        'scipy.interpolate.griddata', i.e. {'linear', 'nearest', 'cubic'}
    :type method: class 'str'

    :return: grid_new: Interpolated grid
    :rtype: class 'np.ndarray'
    """

    lon_flat = lon.flatten()
    lat_flat = lat.flatten()

    lon_newsize = int(360 // area_discretization)
    lat_newsize = int(180 // area_discretization)

    lon_new = np.linspace(np.min(lon_flat), np.max(lon_flat), lon_newsize)
    lat_new = np.linspace(np.min(lat_flat), np.max(lat_flat), lat_newsize)

    lon_grid_new, lat_grid_new = np.meshgrid(lon_new, lat_new)

    grid_new = griddata((lon_flat, lat_flat), grid_values.flatten(), (lon_grid_new, lat_grid_new),  method=method)

    return grid_new

def maxwell_garnet(derived_h2o_perc, original_area_discretization, area_discretization, data_lat, data_lon, instrument_name, polar_host, file_name, data_dir_path):
    """
    maxwell_garnet: Application of the Maxwell-Garnet method to H2O concentration data

    :param derived_h2o_perc: Grid of derived H2O concentrations in wt%
    :type derived_h2o_perc: class 'np.ndarray'
    :param original_area_discretization: Original area discretization in degrees/pixel
    :type original_area_discretization: class 'float'
    :param area_discretization: Desired area discretization in degrees/pixel
    :type area_discretization: class 'float'
    :param data_lat: Latitudinal coordinates of the original data points
    :type data_lat: class 'np.ndarray'
    :param data_lon: Longitudinal coordinates of the original data points
    :type data_lon: class 'np.ndarray'
    :param instrument_name: Instrument name from which the H2O data originates, i.e. GRS or FREND
    :type instrument_name: class 'str'
    :param polar_host: Used polar host material, i.e. 'co2' (CO2 ice) or 'rego' (Regolith)
    :type polar_host: class 'str'
    :param file_name: Name of the H2O data file
    :type file_name: class 'str'
    :param data_dir_path: Directory path of the data
    :type data_dir_path: class 'str'

    :return: grid_dc: Grid of the derived permittivities
    :rtype: class 'np.ndarray'
    """
    v_ice = derived_h2o_perc / 100

    dc_h2o = 3.1
    dc_regolith = 4.0

    if polar_host == 'co2':
        dc_co2 = 2.1
    elif polar_host == 'rego':
        dc_co2 = dc_regolith

    dc_eff_regolith = dc_regolith * ((dc_regolith + (((1 + 2*v_ice)/3)*(dc_h2o - dc_regolith))) / (dc_regolith + (((1 - v_ice)/3)*(dc_h2o - dc_regolith))))   
    dc_eff_co2 = dc_co2 * ((dc_co2 + (((1 + 2*v_ice)/3)*(dc_h2o - dc_co2))) / (dc_co2 + (((1 - v_ice)/3)*(dc_h2o - dc_co2))))

    original_n_lat_options = int(180//original_area_discretization)
    original_n_lon_options = int(360//original_area_discretization)  

    original_lat_deg_options = np.linspace(-90 + original_area_discretization/2, 90 - original_area_discretization/2, original_n_lat_options)  # centers of latitudinal grid cells
    original_lon_deg_options = np.linspace(-180 + original_area_discretization/2, 180 - original_area_discretization/2, original_n_lon_options)  # centers of longitudinal grid cells

    original_grid_h2oconc = np.zeros((original_n_lat_options, original_n_lon_options))
    original_grid_dc = np.zeros((original_n_lat_options, original_n_lon_options))
    grid_lat_for_dc = np.zeros((original_n_lat_options, original_n_lon_options))

    for idx, (lat, lon) in enumerate(zip(data_lat, data_lon)):
        i = np.where(original_lat_deg_options == lat)[0][0]
        j = np.where(original_lon_deg_options == lon)[0][0]
        if lat > 60 or lat < -60:
            original_grid_dc[i, j] = dc_eff_co2[idx]
        else:
            original_grid_dc[i, j] = dc_eff_regolith[idx]
        original_grid_h2oconc[i, j] = derived_h2o_perc[idx]
        grid_lat_for_dc[i, j] = lat

    if area_discretization == original_area_discretization:
        grid_dc = original_grid_dc

    else:
        grid_h2oconc = interpolate_grid(data_lon, data_lat, derived_h2o_perc, area_discretization, method='linear')
        grid_v_ice = grid_h2oconc / 100
        # Use the latitude grid for the interpolated grid, not the original data_lat
        lat_newsize = int(180 // area_discretization)
        lon_newsize = int(360 // area_discretization)
        lat_grid, lon_grid = np.meshgrid(
            np.linspace(-90 + area_discretization/2, 90 - area_discretization/2, lat_newsize),
            np.linspace(-180 + area_discretization/2, 180 - area_discretization/2, lon_newsize),
            indexing='ij'
        )
        grid_dc = np.where(
            (np.abs(lat_grid) <= 60),
            dc_regolith * ((dc_regolith + (((1 + 2*grid_v_ice)/3)*(dc_h2o - dc_regolith))) / (dc_regolith + (((1 - grid_v_ice)/3)*(dc_h2o - dc_regolith)))),
            dc_co2 * ((dc_co2 + (((1 + 2*grid_v_ice)/3)*(dc_h2o - dc_co2))) / (dc_co2 + (((1 - grid_v_ice)/3)*(dc_h2o - dc_co2))))
        )
        print(f"\nInterpolated {instrument_name} dc model from an accuracy of {original_area_discretization} to {area_discretization} degrees in longitudinal/latitudinal direction")

        lon_0, lon_f = -180, 180
        lat_0, lat_f = -90, 90
        h2o_mid = 10.0

        if instrument_name == 'GRS':
            if file_name == "h2o_sr_5x5.tab":
                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(111)
                ax.imshow(img_MarsBackground_Greyscale, extent=[lon_0, lon_f, lat_0, lat_f])
                GRS_h2o_boundaries = np.concatenate([np.linspace(1, h2o_mid, 128, endpoint=False), np.linspace(h2o_mid, 70, 128, endpoint=True)])
                GRS_h2o_norm = BoundaryNorm(GRS_h2o_boundaries, ncolors=plt.get_cmap('gist_rainbow').N)           
                GRS_h2o_img = ax.imshow(np.flipud(original_grid_h2oconc),
                                                extent=[lon_0, lon_f, lat_0, lat_f],
                                                cmap='gist_rainbow',
                                                alpha=0.7,
                                                norm=GRS_h2o_norm)
                ax.set_xlabel('Longitude [$\degree$E]')
                ax.set_ylabel('Latitude [$\degree$N]')
                GRS_h2o_title = r'GRS H$\mathbf{_2}$O Concentration (Boxcar smoothed at 5x5 resolution)'
                fig.text(s=GRS_h2o_title, x=0.47, y=0.98, fontsize=14, ha='center', va='center', fontweight='bold')
                fig.text(s=f'Data retrieved from {file_name}', x=0.47, y=0.945, fontsize=12, ha='center', va='center')  
                cbar_h2o = plt.colorbar(GRS_h2o_img,
                                        label='H$_2$O concentration [wt%]',
                                        fraction=0.0235,
                                        pad=0.035)
                cbar_h2o.set_ticks([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70])
                cbar_h2o.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                cbar_h2o.ax.axhline(h2o_mid, color='black', linewidth=0.5)
                plt.grid(True, color='black', alpha=0.1)
                plt.tight_layout(rect=[-0.01, -0.025, 1.01, 0.975])
                fig_name = f"{instrument_name}_h2o.png"
                fig_path = os.path.join(data_dir_path, fig_name)
                plt.savefig(fig_path)
                # plt.show()
                plt.close(fig)

                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(111)
                ax.imshow(img_MarsBackground_Greyscale, extent=[lon_0, lon_f, lat_0, lat_f])
                GRS_dc_boundaries = np.concatenate([np.linspace(2.1, 3.1, 128, endpoint=False), np.linspace(3.1, 4.0, 128, endpoint=True)])
                GRS_dc_norm = BoundaryNorm(GRS_dc_boundaries, ncolors=plt.get_cmap('gist_rainbow_r').N)            
                GRS_dc_img = ax.imshow(np.flipud(original_grid_dc),
                                                extent=[lon_0, lon_f, lat_0, lat_f],
                                                cmap='gist_rainbow_r',
                                                alpha=0.7,
                                                norm=GRS_dc_norm)
                ax.set_xlabel('Longitude [$\degree$E]')
                ax.set_ylabel('Latitude [$\degree$N]')
                GRS_dc_title = r'Inferred permittivity from GRS H$\mathbf{_2}$O Concentration (Boxcar smoothed at 5x5 resolution)'
                fig.text(s=GRS_dc_title, x=0.47, y=0.98, fontsize=14, ha='center', va='center', fontweight='bold')
                if polar_host == 'rego':
                    fig.text(s=f'Data retrieved from {file_name}; Regolith as polar host material', x=0.47, y=0.945, fontsize=12, ha='center', va='center') 
                elif polar_host == 'co2':
                    fig.text(s=f'Data retrieved from {file_name}; CO$_2$ ice as polar host material', x=0.47, y=0.945, fontsize=12, ha='center', va='center')  
                cbar_dc = plt.colorbar(GRS_dc_img,
                                        boundaries=GRS_dc_boundaries,
                                        label='Inferred permittivity [-]',
                                        fraction=0.0235,
                                        pad=0.035)
                cbar_dc.set_ticks([2.1,2.3,2.5,2.7,2.9,3.1,3.25,3.4,3.55,3.7,3.85,4.0])
                cbar_dc.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                cbar_dc.ax.axhline(dc_h2o, color='black', linewidth=0.5)
                plt.grid(True, color='black', alpha=0.1)
                plt.tight_layout(rect=[-0.01, -0.025, 1.01, 0.975])
                fig_name = f"{instrument_name}_dc.png"
                fig_path = os.path.join(data_dir_path, fig_name)
                plt.savefig(fig_path)
                # plt.show()
                plt.close(fig)
            
            elif file_name == "h2o_sr_5x5_adap.tab":
                GRS_contour_latitudes_top = [50, 50, 46, 46, 40, 40, 46, 46, 50, 50, 55, 55, 60, 60, 55, 55, 50, 50, 46, 46, 50, 50]
                GRS_contour_longitudes_top = [-180, -175, -175, -150, -150, -135, -135, -90, -90, -75, -75, -55, -55, -25, -25, 120, 120, 135, 135, 165, 165, 180]
                GRS_contour_latitudes_bottom = [-55, -55, -60, -60, -55, -55, -50, -50, -45, -45, -50, -50, -55, -55]
                GRS_contour_longitudes_bottom = [-180, -5, -5, 25, 25, 50, 50, 85, 85, 125, 125, 155, 155, 180]

                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(111)
                ax.imshow(img_MarsBackground_Greyscale, extent=[lon_0, lon_f, lat_0, lat_f])
                GRS_h2o_boundaries = np.concatenate([np.linspace(1, h2o_mid, 128, endpoint=False), np.linspace(h2o_mid, 70, 128)])
                GRS_h2o_norm = BoundaryNorm(GRS_h2o_boundaries, ncolors=plt.get_cmap('gist_rainbow').N)
                GRS_h2o_img = ax.imshow(np.flipud(original_grid_h2oconc),
                                        extent=[lon_0, lon_f, lat_0, lat_f],
                                        cmap='gist_rainbow',
                                        alpha=0.7,
                                        norm=GRS_h2o_norm)
                plt.plot(GRS_contour_longitudes_top, GRS_contour_latitudes_top, color='black', linestyle='-', alpha=0.7)
                plt.plot(GRS_contour_longitudes_bottom, GRS_contour_latitudes_bottom, color='black', linestyle='-', alpha=0.7)
                ax.set_xlabel('Longitude [$\degree$E]')
                ax.set_ylabel('Latitude [$\degree$N]')
                GRS_h2o_title = r'GRS H$\mathbf{_2}$O Concentration (Boxcar smoothed at 5x5 resolution)'
                fig.text(s=GRS_h2o_title, x=0.47, y=0.98, fontsize=14, ha='center', va='center', fontweight='bold')
                fig.text(s=f'Data retrieved from {file_name} and partially interpreted', x=0.47, y=0.945, fontsize=12, ha='center', va='center')  
                cbar_h2o = plt.colorbar(GRS_h2o_img,
                                        # boundaries=GRS_dc_boundaries,
                                        label='H$_2$O concentration [wt%]',
                                        fraction=0.0235,
                                        pad=0.035)
                cbar_h2o.set_ticks([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70])
                cbar_h2o.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                mid_line_location = cbar_h2o.ax.transData.transform((0, h2o_mid))[1]
                cbar_h2o.ax.axhline(h2o_mid, color='black', linewidth=0.5)
                plt.grid(True, color='black', alpha=0.1)
                plt.tight_layout(rect=[-0.01, -0.025, 1.01, 0.975])
                fig_name = f"{instrument_name}_h2o_adap.png"
                fig_path = os.path.join(data_dir_path, fig_name)
                plt.savefig(fig_path)
                # plt.show()
                plt.close(fig)

                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(111)
                ax.imshow(img_MarsBackground_Greyscale, extent=[lon_0, lon_f, lat_0, lat_f])
                GRS_dc_boundaries = np.concatenate([np.linspace(2.1, 3.1, 128, endpoint=False), np.linspace(3.1, 4.0, 128, endpoint=True)])
                GRS_dc_norm = BoundaryNorm(GRS_dc_boundaries, ncolors=plt.get_cmap('gist_rainbow_r').N)            
                GRS_dc_img = ax.imshow(np.flipud(original_grid_dc),
                                                extent=[lon_0, lon_f, lat_0, lat_f],
                                                cmap='gist_rainbow_r',
                                                alpha=0.7,
                                                norm=GRS_dc_norm)
                plt.plot(GRS_contour_longitudes_top, GRS_contour_latitudes_top, color='black', linestyle='-', alpha=0.7)
                plt.plot(GRS_contour_longitudes_bottom, GRS_contour_latitudes_bottom, color='black', linestyle='-', alpha=0.7)
                ax.set_xlabel('Longitude [$\degree$E]')
                ax.set_ylabel('Latitude [$\degree$N]')
                GRS_dc_title = r'Inferred permittivity from GRS H$\mathbf{_2}$O Concentration (Boxcar smoothed at 5x5 resolution)'
                fig.text(s=GRS_dc_title, x=0.47, y=0.98, fontsize=14, ha='center', va='center', fontweight='bold')
                if polar_host == 'rego':
                    fig.text(s=f'Data retrieved from {file_name} and partially interpreted; Regolith as polar host material', x=0.47, y=0.945, fontsize=12, ha='center', va='center') 
                elif polar_host == 'co2':
                    fig.text(s=f'Data retrieved from {file_name} and partially interpreted; CO$_2$ ice as polar host material', x=0.47, y=0.945, fontsize=12, ha='center', va='center')  
                cbar_dc = plt.colorbar(GRS_dc_img,
                                        boundaries=GRS_dc_boundaries,
                                        label='Inferred permittivity [-]',
                                        fraction=0.0235,
                                        pad=0.035)
                cbar_dc.set_ticks([2.1,2.3,2.5,2.7,2.9,3.1,3.25,3.4,3.55,3.7,3.85,4.0])
                cbar_dc.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                cbar_dc.ax.axhline(dc_h2o, color='black', linewidth=0.5)
                plt.grid(True, color='black', alpha=0.1)
                plt.tight_layout(rect=[-0.01, -0.025, 1.01, 0.975])
                fig_name = f"{instrument_name}_dc_adap_{polar_host}.png"
                fig_path = os.path.join(data_dir_path, fig_name)
                plt.savefig(fig_path)
                # plt.show()
                plt.close(fig)

        elif instrument_name == 'FREND':
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            ax.imshow(img_MarsBackground_Greyscale, extent=[lon_0, lon_f, lat_0, lat_f])
            FREND_h2o_boundaries = np.concatenate([np.linspace(1, h2o_mid, 128, endpoint=False), np.linspace(h2o_mid, 70, 128)])
            FREND_h2o_norm = BoundaryNorm(FREND_h2o_boundaries, ncolors=plt.get_cmap('gist_rainbow').N)
            FREND_h2o_img = ax.imshow(np.flipud(original_grid_h2oconc),
                                    extent=[lon_0, lon_f, lat_0, lat_f],
                                    cmap='gist_rainbow',
                                    alpha=0.7,
                                    norm=FREND_h2o_norm)
            ax.set_xlabel('Longitude [$\degree$E]')
            ax.set_ylabel('Latitude [$\degree$N]')
            FREND_h2o_title = r'FREND WEH Level (12$\degree$ FWHM Gaussian smoothed at 1x1 resolution)'
            fig.text(s=FREND_h2o_title, x=0.47, y=0.98, fontsize=14, ha='center', va='center', fontweight='bold')
            fig.text(s=f'Data retrieved from {file_name}', x=0.47, y=0.945, fontsize=12, ha='center', va='center')  
            cbar_h2o = plt.colorbar(FREND_h2o_img,
                                    label='WEH Level [wt%]',
                                    fraction=0.0235,
                                    pad=0.035)
            cbar_h2o.set_ticks([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70])
            cbar_h2o.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            mid_line_location = cbar_h2o.ax.transData.transform((0, h2o_mid))[1]
            cbar_h2o.ax.axhline(h2o_mid, color='black', linewidth=0.5)
            plt.grid(True, color='black', alpha=0.1)
            plt.tight_layout(rect=[-0.01, -0.025, 1.01, 0.975])
            fig_name = f"{instrument_name}_h2o.png"
            fig_path = os.path.join(data_dir_path, fig_name)
            plt.savefig(fig_path)
            # plt.show()
            plt.close(fig)

            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            ax.imshow(img_MarsBackground_Greyscale, extent=[lon_0, lon_f, lat_0, lat_f])
            FREND_dc_boundaries = np.concatenate([np.linspace(2.1, 3.1, 128, endpoint=False), np.linspace(3.1, 4.0, 128, endpoint=True)])
            FREND_dc_norm = BoundaryNorm(FREND_dc_boundaries, ncolors=plt.get_cmap('gist_rainbow_r').N)            
            FREND_dc_img = ax.imshow(np.flipud(original_grid_dc),
                                            extent=[lon_0, lon_f, lat_0, lat_f],
                                            cmap='gist_rainbow_r',
                                            alpha=0.7,
                                            norm=FREND_dc_norm)
            ax.set_xlabel('Longitude [$\degree$E]')
            ax.set_ylabel('Latitude [$\degree$N]')
            FREND_dc_title = r'Inferred permittivity from FREND WEH Level (12$\degree$ FWHM Gaussian smoothed at 1x1 resolution)'
            fig.text(s=FREND_dc_title, x=0.47, y=0.98, fontsize=14, ha='center', va='center', fontweight='bold')
            fig.text(s=f'Data retrieved from {file_name}', x=0.47, y=0.945, fontsize=12, ha='center', va='center')  
            cbar_dc = plt.colorbar(FREND_dc_img,
                                    boundaries=FREND_dc_boundaries,
                                    label='Inferred permittivity [-]',
                                    fraction=0.0235,
                                    pad=0.035)
            cbar_dc.set_ticks([2.1,2.3,2.5,2.7,2.9,3.1,3.25,3.4,3.55,3.7,3.85,4.0])
            cbar_dc.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            cbar_dc.ax.axhline(dc_h2o, color='black', linewidth=0.5)
            plt.grid(True, color='black', alpha=0.1)
            plt.tight_layout(rect=[-0.01, -0.025, 1.01, 0.975])
            fig_name = f"{instrument_name}_dc.png"
            fig_path = os.path.join(data_dir_path, fig_name)
            plt.savefig(fig_path)
            # plt.show()
            plt.close(fig)

    return grid_dc

def calc_grid_GRSdc(grid_GRSdc, area_discretization, polar_host):
    """
    calc_grid_GRSdc Calculate the grid of inferred permittivities from GRS H2O concentration data

    :param grid_GRSdc: Original empty grid to be filled with inferred permittivities
    :type grid_GRSdc: class 'np.ndarray'
    :param area_discretization: Area discretization in degrees per pixel
    :type area_discretization: class 'float'
    :param polar_host: Used polar host material, i.e. 'co2' (CO2 ice) or 'rego' (Regolith)
    :type polar_host: class 'str'

    :return: grid_GRSdc: Grid of the derived permittivities for GRS
    :rtype: class 'np.ndarray'
    """
    original_GRS_area_discretization = 5 # Accuracy of the GRS data in 5x5 bins    

    grs_dir_path = os.path.join(os.path.dirname(os.getcwd()), "thesis\\code_and_simulations\\data\\GRS")
    grs_file_name = "h2o_sr_5x5_adap.tab"
    grs_file_path = os.path.join(grs_dir_path, grs_file_name)

    column_names = ["lat", "lon", "h2o_conc", "h2o_sigma", "h2o_sigma_wCFS"]
    data = np.genfromtxt(grs_file_path, delimiter=None, names=column_names, dtype=None, encoding=None)

    grs_data = {col: data[col] for col in data.dtype.names}
    grs_data["lon"] = np.where(grs_data["lon"] > 180, grs_data["lon"] - 360, grs_data["lon"])

    h2o_conc = grs_data["h2o_conc"]     
    mask_nondata = (h2o_conc == 9999.999)
    h2o_conc[mask_nondata] = np.nan 

    grid_GRSdc = maxwell_garnet(h2o_conc, original_GRS_area_discretization, area_discretization, grs_data["lat"], grs_data["lon"], instrument_name='GRS', polar_host=polar_host, file_name=grs_file_name, data_dir_path=grs_dir_path)

    return grid_GRSdc

def calc_grid_FRENDdc(grid_FRENDdc, area_discretization, polar_host):
    """
    calc_grid_FRENDdc Calculate the grid of inferred permittivities from FREND H2O concentration data

    :param grid_FRENDdc: Original empty grid to be filled with inferred permittivities
    :type grid_FRENDdc: class 'np.ndarray'
    :param area_discretization: Area discretization in degrees per pixel
    :type area_discretization: class 'float'
    :param polar_host: Used polar host material, i.e. 'co2' (CO2 ice) or 'rego' (Regolith)
    :type polar_host: class 'str'

    :return: grid_FRENDdc: Grid of the derived permittivities for FREND
    :rtype: class 'np.ndarray'
    """
    original_FREND_area_discretization = 1 # Accuracy of the FREND data in 1x1 bins    

    frend_dir_path = os.path.join(os.path.dirname(os.getcwd()), "thesis\\code_and_simulations\\data\\FREND")
    frend_file_name = "FREND_WEH_MAP_5years.tab"
    frend_file_path = os.path.join(frend_dir_path, frend_file_name)

    column_names = ["lon", "lat", "weh_level", "weh_level_min", "weh_level_max"]
    data = np.genfromtxt(frend_file_path, delimiter=None, names=column_names, dtype=None, encoding=None)

    frend_data = {col: data[col] for col in data.dtype.names}

    grid_FRENDdc = maxwell_garnet(frend_data["weh_level"], original_FREND_area_discretization, area_discretization, frend_data["lat"], frend_data["lon"], instrument_name='FREND', polar_host=polar_host, file_name=frend_file_name, data_dir_path=frend_dir_path)

    return grid_FRENDdc

def calc_grid_slopes(area_discretization, slopes_file_path):
    """
    calc_grid_slopes _summary_

    :param area_discretization: Area discretization in degrees per pixel
    :type area_discretization: class 'float'
    :param slopes_file_path: Path to slope file data
    :type slopes_file_path: class 'str'

    :return: grid_slopes: Grid containing absolute slope values in desired resolution
    :rtype: class 'np.ndarray'
    :return: grid_slopes_RMS: Grid containing RMS slope values in desired resolution
    :rtype: class 'np.ndarray'
    :return: grid_n_hat: Grid containing local surface normal vectors in desired resolution
    :rtype: class 'np.ndarray'
    """

    print(f"\nReading slope and normal vector data of {int(1/area_discretization)} pixel resolution...")

    slopes_dataframe = pd.read_csv(slopes_file_path, header=0)
    slopes_dataframe.columns = ['lon_i'] + [col.split()[0].strip() for col in slopes_dataframe.columns[1:]]
    slopes_data = {col: slopes_dataframe[col].values for col in slopes_dataframe.columns}

    unique_slope_lats = np.unique(slopes_data['lat_i'])  # degrees
    unique_slope_lons = np.unique(slopes_data['lon_i'])  # degrees
    slope_latlon = np.zeros((len(unique_slope_lats), len(unique_slope_lons), 2))
    grid_slopes = np.zeros((len(unique_slope_lats), len(unique_slope_lons)))
    grid_slopes_RMS = np.zeros((len(unique_slope_lats), len(unique_slope_lons)))
    grid_n_hat = np.zeros((len(unique_slope_lats), len(unique_slope_lons), 3))

    lat_indices = {lat: id for id, lat in enumerate(unique_slope_lats)}
    lon_indices = {lon: id for id, lon in enumerate(unique_slope_lons)}

    slopes_data_lat = slopes_data['lat_i'] # degrees
    slopes_data_lon = slopes_data['lon_i'] # degrees
    slopes_data_slope = slopes_data['slope'] # degrees 
    slopes_data_slope_RMS = slopes_data['slope_RMS'] # degrees 
    slopes_data_n_hat = slopes_data['n_hat']
    
    for i in tqdm(range(len(slopes_data_lat))):
        slopes_data_lat_i = slopes_data_lat[i]
        slopes_data_lon_i = slopes_data_lon[i]
        slopes_data_slope_i = slopes_data_slope[i]
        slopes_data_slope_RMS_i = slopes_data_slope_RMS[i]
        n_hat_str_i = slopes_data_n_hat[i]
        n_hat_i = np.fromstring(n_hat_str_i.strip(' []'), sep=' ')

        id_lat = lat_indices[slopes_data_lat_i]
        id_lon = lon_indices[slopes_data_lon_i]

        slope_latlon[id_lat, id_lon, 0] = slopes_data_lat_i
        slope_latlon[id_lat, id_lon, 1] = slopes_data_lon_i
        grid_slopes[id_lat, id_lon] = slopes_data_slope_i
        grid_slopes_RMS[id_lat, id_lon] = slopes_data_slope_RMS_i
        grid_n_hat[id_lat, id_lon] = n_hat_i

    grid_slopes = np.deg2rad(grid_slopes)
    grid_slopes_RMS = np.deg2rad(grid_slopes_RMS)

    return grid_slopes, grid_slopes_RMS, grid_n_hat

def global_grids(area_discretization, standard_grid, lat_deg_options, lon_deg_options, polar_host):
    """
    global_grids: This functions fills the different global grids (which are constant over time)

    :param area_discretization: Area discretization in degrees per pixel
    :type area_discretization: class 'float'
    :param standard_grid: Standard empty grid to be filled with different parameters
    :type standard_grid: class 'np.ndarray'
    :param lat_deg_options: Latitudinal centers of the grid cells in degrees
    :type lat_deg_options: class 'np.ndarray'
    :param lon_deg_options: Longitudinal centers of the grid cells in degrees
    :type lon_deg_options: class 'np.ndarray'
    :param polar_host: Selected polar host material, i.e. 'co2' (CO2 ice) or 'rego' (Regolith)
    :type polar_host: class 'str'

    :return: grid_latlon: Grid containing latitudinal and longitudinal coordinates of each grid cell
    :rtype: class 'np.ndarray'
    :return: grid_S: Grid containing surface area of each grid cell
    :rtype: class 'np.ndarray'
    :return: grid_GRSdc: Grid containing inferred permittivities from GRS data for each grid cell
    :rtype: class 'np.ndarray'
    :return: grid_FRENDdc: Grid containing inferred permittivities from FREND data for each grid cell
    :rtype: class 'np.ndarray'
    :return: grid_slopes: Grid containing absolute slopes for each grid cell
    :rtype: class 'np.ndarray'
    :return: grid_slopes_RMS: Grid containing RMS slopes for each grid cell
    :rtype: class 'np.ndarray'
    :return: grid_n_hat: Grid containing local surface normal vectors for each grid cell
    :rtype: class 'np.ndarray'
    """

    grid_latlon = np.zeros((n_lat_options, n_lon_options, 2))
    grid_S = standard_grid.copy()
    grid_GRSdc = standard_grid.copy()
    grid_FRENDdc = standard_grid.copy()

    lat_grid, lon_grid = np.meshgrid(lat_deg_options, lon_deg_options, indexing='ij')
    grid_latlon[:, :, 0] = lat_grid
    grid_latlon[:, :, 1] = lon_grid

    grid_S = calc_grid_surfacearea(grid_S, area_discretization, n_lat_options, n_lon_options)
    S_total = np.sum(grid_S) # Approx. 144 400 000 000 000 m^2 or 144 400 000 km^2
    print(f"Total Mars surface area: {S_total} km^2")

    grid_GRSdc = calc_grid_GRSdc(grid_GRSdc, area_discretization, polar_host)
    grid_FRENDdc = calc_grid_FRENDdc(grid_FRENDdc, area_discretization, polar_host)

    grid_slopes, grid_slopes_RMS, grid_n_hat = calc_grid_slopes(area_discretization, slopes_file_path)

    return grid_latlon, grid_S, grid_GRSdc, grid_FRENDdc, grid_slopes, grid_slopes_RMS, grid_n_hat

def find_footprint(grid_footprint, lat_deg_options, lon_deg_options, intpoint_lat, intpoint_lon, bs_lat):
    """
    find_footprint: This functions finds the path of the footrpint, using the intersection points

    :param grid_footprint: Original empty grid to be used for the footprint
    :type grid_footprint: class 'np.ndarray'
    :param lat_deg_options: Latitudinal center options of the grid cells in degrees
    :type lat_deg_options: class 'np.ndarray'
    :param lon_deg_options: Longitudinal center options of the grid cells in degrees
    :type lon_deg_options: class 'np.ndarray'
    :param intpoint_lat: Latitudes of the intersection points bounding the footprint
    :type intpoint_lat: class 'np.ndarray'
    :param intpoint_lon: Longitudes of the intersection points bounding the footprint
    :type intpoint_lon: class 'np.ndarray'
    :param bs_lat: Latitude of the relevent S/C boresight on Mars' surface
    :type bs_lat: class 'float'

    :return: grid_footprint: Grid defining the footprint coverage (1 inside footprint,
        0 outside footprint)
    :rtype: class 'np.ndarray'
    :return: intpoint_lat: Latitudes of the intersection points bounding the footprint (modified
        to close the footprint path)
    :rtype: class 'np.ndarray'
    :return: intpoint_lon: Longitudes of the intersection points bounding the footprint (modified
        to close the footprint path)
    :rtype: class 'np.ndarray'
    """

    bs_lat_deg = spice.convrt(bs_lat, 'RADIANS', 'DEGREES')

    point_NE = [90, 180]
    point_SE = [-90, 180]
    point_SW = [-90, -180]
    point_NW = [90, -180]

    intpoint_lat_poly = intpoint_lat.copy()
    intpoint_lon_poly = intpoint_lon.copy()

    intpoint_lon_lb = -120
    intpoint_lon_ub = 120

    lon_grid, lat_grid = np.meshgrid(lon_deg_options, lat_deg_options)
    lonlat_points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))

    if np.min(intpoint_lon) <= intpoint_lon_lb and np.max(intpoint_lon) >= intpoint_lon_ub:
        transitions = []
        for i_trans in range(len(intpoint_lon)-1):
            intpoint_lon_i = intpoint_lon[i_trans]
            intpoint_lon_inext = intpoint_lon[i_trans+1]
            if (intpoint_lon_i <= intpoint_lon_lb and intpoint_lon_inext >= intpoint_lon_ub) or (intpoint_lon_i >= intpoint_lon_ub and intpoint_lon_inext <= intpoint_lon_lb):
                transitions.append(i_trans)

        if len(transitions) == 1:
            # Footprint spans entire longitudinal range, polygon requires additional points to close the footprint
            if bs_lat_deg >= 0: # np.mean(intpoint_lat):
                intpoint_lon_min_id = np.where(intpoint_lon_poly == np.min(intpoint_lon))[0][0]
                next_id = (intpoint_lon_min_id + 1) % len(intpoint_lon_poly)

                while intpoint_lon_poly[next_id] < lon_deg_options[0]:
                    intpoint_lon_min_id = next_id
                    next_id = (intpoint_lon_min_id + 1) % len(intpoint_lon_poly)
                    if intpoint_lon_min_id == len(intpoint_lon_poly) - 1:
                        intpoint_lon_min_id = 0

                point_W = [intpoint_lat_poly[intpoint_lon_min_id], -180]
                point_E = [intpoint_lat_poly[(intpoint_lon_min_id+1) % len(intpoint_lat_poly)], 180]

                intpoint_lat_poly = np.insert(intpoint_lat_poly, intpoint_lon_min_id+1, point_W[0])
                intpoint_lon_poly = np.insert(intpoint_lon_poly, intpoint_lon_min_id+1, point_W[1])
                intpoint_lat_poly = np.insert(intpoint_lat_poly, intpoint_lon_min_id+2, point_NW[0])
                intpoint_lon_poly = np.insert(intpoint_lon_poly, intpoint_lon_min_id+2, point_NW[1])
                intpoint_lat_poly = np.insert(intpoint_lat_poly, intpoint_lon_min_id+3, point_NE[0])
                intpoint_lon_poly = np.insert(intpoint_lon_poly, intpoint_lon_min_id+3, point_NE[1])
                intpoint_lat_poly = np.insert(intpoint_lat_poly, intpoint_lon_min_id+4, point_E[0])
                intpoint_lon_poly = np.insert(intpoint_lon_poly, intpoint_lon_min_id+4, point_E[1])
                
                intpoint_lat = np.insert(intpoint_lat, intpoint_lon_min_id+1, point_W[0])
                intpoint_lon = np.insert(intpoint_lon, intpoint_lon_min_id+1, point_W[1])
                intpoint_lat = np.insert(intpoint_lat, intpoint_lon_min_id+2, point_E[0])
                intpoint_lon = np.insert(intpoint_lon, intpoint_lon_min_id+2, point_E[1])

            elif bs_lat_deg < 0: # np.mean(intpoint_lat):
                intpoint_lon_min_id = np.where(intpoint_lon_poly == np.min(intpoint_lon))[0][0]
                next_id = (intpoint_lon_min_id + 1) % len(intpoint_lon_poly)

                while intpoint_lon_poly[next_id] < lon_deg_options[0]:
                    intpoint_lon_min_id = next_id
                    next_id = (intpoint_lon_min_id + 1) % len(intpoint_lon_poly)
                    if intpoint_lon_min_id == len(intpoint_lon_poly) - 1:
                        intpoint_lon_min_id = 0

                point_W = [intpoint_lat_poly[intpoint_lon_min_id], -180]
                point_E = [intpoint_lat_poly[(intpoint_lon_min_id+1) % len(intpoint_lat_poly)], 180]

                intpoint_lat_poly = np.insert(intpoint_lat_poly, intpoint_lon_min_id, point_E[0])
                intpoint_lon_poly = np.insert(intpoint_lon_poly, intpoint_lon_min_id, point_E[1])
                intpoint_lat_poly = np.insert(intpoint_lat_poly, intpoint_lon_min_id+1, point_SE[0])
                intpoint_lon_poly = np.insert(intpoint_lon_poly, intpoint_lon_min_id+1, point_SE[1])
                intpoint_lat_poly = np.insert(intpoint_lat_poly, intpoint_lon_min_id+2, point_SW[0])
                intpoint_lon_poly = np.insert(intpoint_lon_poly, intpoint_lon_min_id+2, point_SW[1])
                intpoint_lat_poly = np.insert(intpoint_lat_poly, intpoint_lon_min_id+3, point_W[0])
                intpoint_lon_poly = np.insert(intpoint_lon_poly, intpoint_lon_min_id+3, point_W[1])

            footprint_polygon = np.column_stack((intpoint_lon_poly, intpoint_lat_poly))

            path = Path(footprint_polygon)
            mask = path.contains_points(np.column_stack((lonlat_points[:,1], lonlat_points[:,0]))).astype(np.float32)
            grid_footprint[:,:] = mask.reshape(lat_grid.shape).astype(int)

        elif len(transitions) == 2:
            # Footprint crosses min/max longitued (but not one of the poles), polygon requires additional points to close the footprint

            lat_avgs = []
            for i_trans in transitions:
                intpoint_lat_i = intpoint_lat[i_trans]
                intpoint_lat_inext = intpoint_lat[i_trans+1]
                lat_avgs.append((intpoint_lat_i + intpoint_lat_inext)/2)

            N_lat_avg = np.max(lat_avgs)
            N_lat_avg_idx = np.argmax(lat_avgs)
            S_lat_avg = np.min(lat_avgs)
            S_lat_avg_idx = np.argmin(lat_avgs)

            point_NE = [N_lat_avg, +180]
            point_SE = [S_lat_avg, +180]
            point_NW = [N_lat_avg, -180]
            point_SW = [S_lat_avg, -180]

            if N_lat_avg_idx < S_lat_avg_idx:
                # First transition at Northern border
                trans1 = transitions[N_lat_avg_idx]
                trans2 = transitions[S_lat_avg_idx]
                if intpoint_lon[trans1] > intpoint_lon[trans1+1]:
                    # Footprint crosses Northern border in clockwise direction
                    point1 = point_NE
                    point2 = point_NW
                    point3 = point_SW
                    point4 = point_SE
                    E_first = True

                else:
                    # Footprint crosses Northern border in counter-clockwise direction
                    point1 = point_NW
                    point2 = point_NE
                    point3 = point_SE
                    point4 = point_SW
                    E_first = False

            elif S_lat_avg_idx < N_lat_avg_idx:
                # First transition at Southern border
                trans1 = transitions[S_lat_avg_idx]
                trans2 = transitions[N_lat_avg_idx]
                if intpoint_lon[trans1] > intpoint_lon[trans1+1]:
                    # Footprint crosses Southern border in counter-clockwise direction
                    point1 = point_SE
                    point2 = point_SW
                    point3 = point_NW
                    point4 = point_NE
                    E_first = True

                else:
                    # Footprint crosses Southern border in clockwise direction
                    point1 = point_SW
                    point2 = point_SE
                    point3 = point_NE
                    point4 = point_NW
                    E_first = False

            if E_first == True:
                intpoint_lon_poly_E = np.concatenate([intpoint_lon[:trans1], [point1[1], point4[1]], intpoint_lon[trans2+1:]])
                intpoint_lat_poly_E = np.concatenate([intpoint_lat[:trans1], [point1[0], point4[0]], intpoint_lat[trans2+1:]])
                intpoint_lon_poly_W = np.concatenate([[point2[1]], intpoint_lon[trans1+1:trans2+1], [point3[1]]])
                intpoint_lat_poly_W = np.concatenate([[point2[0]], intpoint_lat[trans1+1:trans2+1], [point3[0]]])

            elif E_first == False:
                intpoint_lon_poly_E = np.concatenate([[point2[1]], intpoint_lon[trans1+1:trans2+1], [point3[1]]])
                intpoint_lat_poly_E = np.concatenate([[point2[0]], intpoint_lat[trans1+1:trans2+1], [point3[0]]])
                intpoint_lon_poly_W = np.concatenate([intpoint_lon[:trans1], [point1[1], point4[1]], intpoint_lon[trans2+1:]])
                intpoint_lat_poly_W = np.concatenate([intpoint_lat[:trans1], [point1[0], point4[0]], intpoint_lat[trans2+1:]])

            intpoint_lon = np.insert(intpoint_lon, trans1, point1[1])
            intpoint_lon = np.insert(intpoint_lon, trans1+1, point2[1])
            intpoint_lon = np.insert(intpoint_lon, trans2+2, point3[1])
            intpoint_lon = np.insert(intpoint_lon, trans2+3, point4[1])

            intpoint_lat = np.insert(intpoint_lat, trans1, point1[0])
            intpoint_lat = np.insert(intpoint_lat, trans1+1, point2[0])
            intpoint_lat = np.insert(intpoint_lat, trans2+2, point3[0])
            intpoint_lat = np.insert(intpoint_lat, trans2+3, point4[0])

            footprint_polygon_E = np.column_stack((intpoint_lon_poly_E, intpoint_lat_poly_E))
            path_E = Path(footprint_polygon_E)
            footprint_polygon_W = np.column_stack((intpoint_lon_poly_W, intpoint_lat_poly_W))
            path_W = Path(footprint_polygon_W)

            mask_E = path_E.contains_points(np.column_stack((lonlat_points[:,1], lonlat_points[:,0]))).astype(np.float32)
            mask_W = path_W.contains_points(np.column_stack((lonlat_points[:,1], lonlat_points[:,0]))).astype(np.float32)
            mask = np.logical_or(mask_E, mask_W)
            grid_footprint[:,:] = mask.reshape(lat_grid.shape).astype(int)

    else: 
        footprint_polygon = np.column_stack((intpoint_lon_poly, intpoint_lat_poly))

        path = Path(footprint_polygon)
        mask = path.contains_points(np.column_stack((lonlat_points[:,1], lonlat_points[:,0]))).astype(np.float32)
        grid_footprint[:,:] = mask.reshape(lat_grid.shape).astype(int)

    return grid_footprint, intpoint_lat, intpoint_lon

def plot_split_footprint(SC_inpoint_lon, SC_intpoint_lat, SC_color):
    """
    plot_split_footprint: This function plots the footprint bounds, split if crossing the map edges

    :param SC_inpoint_lon: Longitudes of the intersection points bounding the footprint
    :type SC_inpoint_lon: class 'np.ndarray'
    :param SC_intpoint_lat: Latitudes of the intersection points bounding the footprint
    :type SC_intpoint_lat: class 'np.ndarray'
    :param SC_color: Color to use for the footprint plot ('red' for MEX, 'cyan' for TGO)
    :type SC_color: class 'str'
    """

    k_split = 0
    for k in range(len(SC_inpoint_lon)-1):
        lon_diff = np.abs(SC_inpoint_lon[k+1] - SC_inpoint_lon[k])
        lat_diff = np.abs(SC_intpoint_lat[k+1] -  SC_intpoint_lat[k])

        if lon_diff > 180 or lat_diff > 90:
            plt.plot(SC_inpoint_lon[k_split:k+1], SC_intpoint_lat[k_split:k+1], color=SC_color, alpha=0.5)
            k_split = k + 1

    plt.plot(SC_inpoint_lon[k_split:k+1], SC_intpoint_lat[k_split:k+1], color=SC_color, alpha=0.5)

    return

def plot_map_axes(ax):
    """
    plot_map_axes: This function sets map axes with ticks and grid

    :param ax: Axes of the Mars plot
    :type ax: class 'matplotlib.axes._subplots.AxesSubplot'
    """
    lon_major_ticks = np.arange(-180, 181, 30)
    lon_minor_ticks = np.arange(-180, 181, 5)
    lat_major_ticks = np.arange(-90, 91, 30)
    lat_minor_ticks = np.arange(-90, 91, 5)

    ax.set_xticks(lon_major_ticks)
    ax.set_xticks(lon_minor_ticks, minor=True)
    ax.set_yticks(lat_major_ticks)
    ax.set_yticks(lat_minor_ticks, minor=True)

    ax.tick_params(axis='both', which='major', labelsize=10)

    ax.grid(which='minor', alpha=0.2, color='black', linestyle='-', linewidth=0.3)
    ax.grid(which='major', alpha=0.7, color='black', linestyle='-', linewidth=0.3)

    ax.set_axisbelow(True)
    return

def readout_IQ(BSR_title, IQ_title, target_length, AGC_baseline):
    """
    readout_IQ: This function reads out the BSR IQ files and extracts the AGC data

    :param BSR_title: Title of the BSR measurement under consideration
    :type BSR_title: class 'str'
    :param IQ_title: Title of the IQ file corresponding to the BSR measurement under consideration
    :type IQ_title: class 'str'
    :param target_length: Target length to which the AGC data should be averaged, corresponding to the amount of modelling time steps
    :type target_length: class 'int'
    :param AGC_baseline: Baseline noise floor value for the AGC data, set to 124 for MEX-TGO BSR
    :type AGC_baseline: class 'int'

    :return: agc_data_NTC32: AGC data extracted from the IQ file as 8 bit unsigned integers
    :rtype: class 'np.ndarray'
    :return: P_R_dB: Converted AGC data to received power values in dB
    :rtype: class 'np.ndarray'
    :return: P_R_W: Converted AGC data to received power values in Watts
    :rtype: class 'np.ndarray'
    """

    IQ_dir_path = os.path.join(os.path.dirname(os.getcwd()), "thesis\\code_and_simulations\\data\\BSR")
    file_path = os.path.join(IQ_dir_path, IQ_title)
    
    print(f"\tProcessing {BSR_title} data from {IQ_title}...")

    with open(file_path, 'rb') as f:
        data = f.read()

    # First, process as 4-byte words to find where time coded records end
    # Read the data as 32-bit unsigned integers (little-endian)
    words_all = np.frombuffer(data, dtype='>u4')

    words_TC96 = np.array([], dtype='>u4') # Open Sample Time Coded Record (96 bits)

    for i in np.arange(0, len(words_all), 3):
        bit_string_32b1 = f"{words_all[i]:032b}"
        bit_string_32b2 = f"{words_all[i+1]:032b}"
        bit_string_32b3 = f"{words_all[i+2]:032b}"
        bit_string_96b = bit_string_32b1 + bit_string_32b2 + bit_string_32b3

        if bit_string_96b.startswith('1100') or bit_string_96b.startswith('1110'):
            words_TC96 = np.append(words_TC96, words_all[i:i+3])
        else:
            break

    words_NTC32 = words_all[i:] # Open Sample Non-Time Coded Record (32 bits)

    agc_bits_NTC32 = words_NTC32 >> 4 # Moves AGC bits from bits 20:28, i.e. 4:11 from the right, to bits 16:24, i.e. 0:8 from the right
    agc_data_NTC32 = agc_bits_NTC32 & 255 # Mask to keep only bits 20:28, i.e. 4:11 from the right; 255 = 11111111, i.e. 8 bits

    dB_step = 1 # dB step size for each AGC value

    if min(agc_data_NTC32) < AGC_baseline:
        agc_data_NTC32_atbaseline = np.where(agc_data_NTC32 < AGC_baseline, AGC_baseline, agc_data_NTC32) # Set all values below baseline to baseline
    else:
        agc_data_NTC32_atbaseline = agc_data_NTC32.copy()

    P_R_dB = P_noisefloor_dB + (agc_data_NTC32_atbaseline - AGC_baseline) * dB_step # Convert AGC values to dB
    P_R_W = 1.0 * (10.0**(P_R_dB / 10))/1000 # Convert dB to Watts

    print(f"\tAGC range: {min(agc_data_NTC32)} to {max(agc_data_NTC32)}")

    agc_data_NTC32_arr = np.asarray(agc_data_NTC32).astype(float)
    P_R_dB_arr = np.asarray(P_R_dB)
    P_R_W_arr = np.asarray(P_R_W)

    window_size = int(np.ceil(len(agc_data_NTC32_arr) / target_length))
    trimmed_length = window_size * target_length

    agc_data_NTC32_arr_trimmed = np.pad(agc_data_NTC32_arr, (0, trimmed_length - len(agc_data_NTC32_arr)), mode='constant', constant_values=np.nan)
    agc_data_NTC32_arr_reshaped = agc_data_NTC32_arr_trimmed.reshape(target_length, window_size)
    agc_data_NTC32 = np.nanmean(agc_data_NTC32_arr_reshaped, axis=1)

    P_R_dB_arr_trimmed = np.pad(P_R_dB_arr, (0, trimmed_length - len(P_R_dB_arr)), mode='constant', constant_values=np.nan)
    P_R_dB_arr_reshaped = P_R_dB_arr_trimmed.reshape(target_length, window_size)
    P_R_dB = np.nanmean(P_R_dB_arr_reshaped, axis=1)
    
    P_R_W_arr_trimmed = np.pad(P_R_W_arr, (0, trimmed_length - len(P_R_W_arr)), mode='constant', constant_values=np.nan)
    P_R_W_arr_reshaped = P_R_W_arr_trimmed.reshape(target_length, window_size)
    P_R_W = np.nanmean(P_R_W_arr_reshaped, axis=1)

    return agc_data_NTC32, P_R_dB, P_R_W

def multiproc_filewriter(queue_manager, BSR_file_path_csv): 
    """
    multiproc_filewriter: This function writes the output data to a CSV file, using a queue manager in multi-/parallel processing

    :param queue_manager: Queue manager to handle incoming data from different processes
    :type queue_manager: class 'multiprocessing.Manager().Queue()'
    :param BSR_file_path_csv: Path to the output CSV file
    :type BSR_file_path_csv: class 'str'
    """   

    # with open(BSR_file_path_csv, 'a') as BSR_file: # Used for running the model in chunks, as was required for completing the 16 pix/deg resolution run
    with open(BSR_file_path_csv, 'w') as BSR_file: 
        BSR_file.write(f"# utc [-], et [s], R_T_spoint [km], R_R_spoint [km], R_MEXspointTGO [km], r_MEXTGO [km], spoint_lon [deg], spoint_lat [deg], midpoint_lon [deg], midpoint_lat [deg], emis_MEX [deg], emis_TGO [deg], emis_avg [deg], S_mutualfootprint [km^2], rho_data [-], rho_modelGRS [-], rho_modelFREND [-], dc_data [-], dc_modelGRS [-], dc_modelFREND [-], AGC_data [-], P_R_data [W], P_R_modelGRS [W], P_R_modelFREND [W], P_R_modelGRS_maxcont [W], P_R_modelFREND_maxcont [W], P_R_modelGRS_mincont [W], P_R_modelFREND_mincont [W], P_R_modelGRS_spoint [W], P_R_modelFREND_spoint [W], P_R_freespace [W], gamma_spoint [deg], C_spoint [-], topography_spoint [-], gamma_GRS_maxcont [deg], C_GRS_maxcont [-], topography_GRS_maxcont [-], gamma_FREND_maxcont [deg], C_FREND_maxcont [-], topography_FREND_maxcont [-], rho_modelGRS_spoint [-], dc_modelGRS_spoint [-], rho_modelGRS_maxcont [-], dc_modelGRS_maxcont [-], rho_modelFREND_spoint [-], dc_modelFREND_spoint [-], rho_modelFREND_maxcont [-], dc_modelFREND_maxcont [-], G_T_freespace [-], G_R_freespace [-], angle_MEX2TGO [deg], angle_TGO2MEX [deg], lat_GRS_maxcont [deg], lon_GRS_maxcont [deg], lat_FREND_maxcont [deg], lon_FREND_maxcont [deg], MEXbs_lon [deg], MEXbs_lat [deg], TGObs_lon [deg], TGObs_lat [deg] \n")

        while 1:
            message = queue_manager.get()
            if message == 'STOP':
                break
            
            BSR_file.write(str(message) + '\n') 
            BSR_file.flush()

    return

def multiproc_etsteps(j, manager_queue, measop_data, BSRperm_data_dir_path, standard_grid, BSRfp_data_dir_path, lat_deg_options, lon_deg_options, code_and_sim_path, img_MarsBackground, MEXbs_track, TGObs_track, area_discretization, BSRfp_visuals_dir_path, grid_S, grid_latlon, MEX_gaussian_parameters, TGO_gaussian_parameters, grid_GRSdc, grid_FRENDdc, grid_slopes, grid_slopes_RMS, grid_n_hat, P_R_data_W, AGC_data, P_T, lambda_UHF, radii_ell, BSR_title_i, MTP_title_i, n_pixels_slopes):
    """
    multiproc_etsteps: This function handles the processing of each measurement time step during multi-/parallel processing
    """
    
    # *****************************************************************************************

    MetaKernelsPath = os.path.join(code_and_sim_path, "kernels\\mk")

    main_gen_mk = os.path.join(MetaKernelsPath, 'gen_mk.tm') # General kernels
    main_MEX_mk = os.path.join(MetaKernelsPath, 'MEX_mk.tm') # MEX-specific kernels
    main_TGO_mk = os.path.join(MetaKernelsPath, 'TGO_mk.tm') # TGO-specific kernels

    spice.furnsh(main_gen_mk)
    spice.furnsh(main_MEX_mk)
    spice.furnsh(main_TGO_mk)

    # *****************************************************************************************
        
    """ Readout measop data """
    
    utc = measop_data['utc'][j]
    et = measop_data['et'][j]
    r_MEXTGO = measop_data['MEX2TGO'][j]
    spoint_lon_deg = measop_data['spoint_lon'][j]
    spoint_lat_deg = measop_data['spoint_lat'][j]
    emission_MEX = measop_data['emission_MEX'][j]
    emission_TGO = measop_data['emission_TGO'][j]
    midpoint_lon_deg = measop_data['lon_midpoint'][j]
    midpoint_lat_deg = measop_data['lat_midpoint'][j]
    
    # *****************************************************************************************

    """ Setup new data files """

    gridpoints_file_name = 'gridpoints_' + utc.replace(" ", ".").replace(":", ".") + '.csv'
    gridpoints_file_path = os.path.join(BSRperm_data_dir_path, f"{gridpoints_file_name}")
    gridpoints_file = open(gridpoints_file_path, 'w')

    gridpoints_file.write(f"# lon_gridpoint [deg], lat_gridpoint [deg], P_R_modelGRS [W], P_R_modelFREND [W] \n")

    # *****************************************************************************************
    
    """ Measurement geometry """

    r_MEX = spice.spkpos('MEX', et, 'IAU_MARS', 'LT+S', 'MARS')[0] # km; MEX position
    r_TGO = spice.spkpos('TGO', et, 'IAU_MARS', 'LT+S', 'MARS')[0] # km; TGO position
    r_MEX2TGO = r_TGO - r_MEX # km; vector from MEX to TGO
    r_TGO2MEX = r_MEX - r_TGO # km; vector from TGO to MEX
        
    spoint_lon = spice.convrt(spoint_lon_deg, 'DEGREES', 'RADIANS') # radians
    spoint_lat = spice.convrt(spoint_lat_deg, 'DEGREES', 'RADIANS') # radians
    r_spoint = spice.latsrf('ELLIPSOID', 'MARS', et, 'IAU_MARS', [[spoint_lon, spoint_lat]])[0]

    r_MEXspoint = r_spoint - r_MEX # km; vector from MEX to the specular point
    r_TGOspoint = r_spoint - r_TGO # km; vector from TGO to the specular point

    MEXbs_lon, MEXbs_lat = spice.reclat(r_MEX)[1:] # radians; MEX boresight in conventional attitude: nadir-pointing
    MEXbs_lon_deg, MEXbs_lat_deg = spice.convrt([MEXbs_lon, MEXbs_lat], 'RADIANS', 'DEGREES') # degrees; MEX boresight in conventional attitude: nadir-pointing
    MEX_r_bs = spice.latsrf('ELLIPSOID', 'MARS', et, 'IAU_MARS', [[MEXbs_lon, MEXbs_lat]])[0]
    r_MEXbs = MEX_r_bs - r_MEX # km; vector from MEX to the boresight point

    TGObs_lon, TGObs_lat = spice.reclat(r_TGO)[1:] # radians; TGO boresight in conventional attitude: nadir-pointing 
    TGObs_lon_deg, TGObs_lat_deg = spice.convrt([TGObs_lon, TGObs_lat], 'RADIANS', 'DEGREES') # degrees; TGO boresight in conventional attitude: nadir-pointing  
    TGO_r_bs = spice.latsrf('ELLIPSOID', 'MARS', et, 'IAU_MARS', [[TGObs_lon, TGObs_lat]])[0]
    r_TGObs = TGO_r_bs - r_TGO # km; vector from TGO to the boresight point

    angle_MEX2TGO = spice.vsep(r_MEXbs, r_MEX2TGO)
    angle_TGO2MEX = spice.vsep(r_TGObs, r_TGO2MEX)

    # *****************************************************************************************

    """ Calculate and plot mutual footprint """        
    
    footprint_file_name = 'footprint_' + utc.replace(" ", ".").replace(":", ".") + '.csv'
    footprint_file_path = os.path.join(BSRfp_data_dir_path, footprint_file_name)
    footprint_dataframe = pd.read_csv(footprint_file_path, header=0)
    footprint_dataframe.columns = ['MEX_intpoint_lon'] + [col.split()[0].strip() for col in footprint_dataframe.columns[1:]]
    footprint_data = {col: footprint_dataframe[col].values for col in footprint_dataframe.columns}

    MEX_intpoint_lon = footprint_data['MEX_intpoint_lon']
    MEX_intpoint_lat = footprint_data['MEX_intpoint_lat']
    TGO_intpoint_lon = footprint_data['TGO_intpoint_lon']
    TGO_intpoint_lat = footprint_data['TGO_intpoint_lat']

    grid_MEX_footprint = standard_grid.copy() # values inside footprint will be 1, outside footprint will be 0
    grid_TGO_footprint = standard_grid.copy() # values inside footprint will be 1, outside footprint will be 0

    # mutual footprint of MEX and TGO, where values inside footprint will be 1, outside footprint will be 0
    grid_MEX_footprint, MEX_intpoint_lat, MEX_intpoint_lon = find_footprint(grid_MEX_footprint, lat_deg_options, lon_deg_options, MEX_intpoint_lat, MEX_intpoint_lon, MEXbs_lat)
    grid_TGO_footprint, TGO_intpoint_lat, TGO_intpoint_lon = find_footprint(grid_TGO_footprint, lat_deg_options, lon_deg_options, TGO_intpoint_lat, TGO_intpoint_lon, TGObs_lat)
    grid_mutual_footprint = grid_MEX_footprint * grid_TGO_footprint 
    mask_mutual_footprint = (grid_mutual_footprint == 1)

    fig_footprint_grid = plt.figure(figsize=(12, 10))
    fig_footprint_grid.set_size_inches(19.2, 10.8)  # Nominal full screen: 1920x1080
    plt.imshow(img_MarsBackground, extent=[-180, 180, -90, 90], aspect='equal')

    plot_split_footprint(MEX_intpoint_lon, MEX_intpoint_lat, SC_color='red')
    plot_split_footprint(TGO_intpoint_lon, TGO_intpoint_lat, SC_color='cyan')

    ax_footprint_grid = plt.gca()

    masked_MEX_footprint = np.ma.masked_where(grid_MEX_footprint == 0, grid_MEX_footprint)
    plt.imshow(np.flipud(masked_MEX_footprint), extent=[-180, 180, -90, 90], aspect='equal', cmap='Reds_r', alpha=0.5, interpolation='none')
    masked_TGO_footprint = np.ma.masked_where(grid_TGO_footprint == 0, grid_TGO_footprint)
    plt.imshow(np.flipud(masked_TGO_footprint), extent=[-180, 180, -90, 90], aspect='equal', cmap='BrBG_r', alpha=0.5, interpolation='none')

    spoint_track_lon = measop_data['spoint_lon']
    spoint_track_lat = measop_data['spoint_lat']

    plt.scatter(MEXbs_track[0], MEXbs_track[1], label='MEX_boresight track', s=1, color='red')
    plt.scatter(TGObs_track[0], TGObs_track[1], label='TGO_boresight track', s=1, color='cyan')
    plt.scatter(spoint_track_lon, spoint_track_lat, label='spoint track', s=1, color='fuchsia')

    plt.scatter(spoint_lon_deg, spoint_lat_deg, color='fuchsia', marker='*', s=150, label='Specular point', edgecolors='black')
    plt.scatter(MEXbs_lon_deg, MEXbs_lat_deg, color='red', marker='*', s=150, label='MEX boresight', edgecolors='black')
    plt.scatter(TGObs_lon_deg, TGObs_lat_deg, color='cyan', marker='*', s=150, label='TGO boresight', edgecolors='black')

    title = f'Mutual signal footprint for {BSR_title_i} (at {n_pixels_slopes} pixel/degree resolution)'
    fig_footprint_grid.text(s=title, x=0.42, y=0.72+0.09, fontsize=18, ha='center', va='center')
    fig_footprint_grid.text(s='(Background: HRSC_MOLA Blended DEM at 1500mp)', x=0.42, y=0.695+0.09, fontsize=12, ha='center', va='center')

    plot_map_axes(ax_footprint_grid)

    box_MarsSpoints = ax_footprint_grid.get_position()
    ax_footprint_grid.set_position([box_MarsSpoints.x0, box_MarsSpoints.y0, box_MarsSpoints.width * 0.8, box_MarsSpoints.height])
    ax_footprint_grid.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_footprint_grid.set_xlabel('Longitude [$\degree$E]')
    ax_footprint_grid.set_ylabel('Latitude [$\degree$N]')

    fig_footprint_grid.text(x=0.8, y=0.725, s=f'UTC =', fontsize=12, ha='center', va='center', fontweight='bold')
    fig_footprint_grid.text(x=0.8, y=0.695, s=f'{utc[:11]}\n{utc[12:]}', fontsize=12, ha='center', va='center')
    fig_footprint_grid.text(x=0.8, y=0.635, s=f't_step =', fontsize=12, ha='center', va='center', fontweight='bold')
    fig_footprint_grid.text(x=0.8, y=0.615, s=f'{j+1}', fontsize=12, ha='center', va='center')

    utc_stripped = (utc.replace(' ', '.')).replace(':', '.')
    fig_path = os.path.join(BSRfp_visuals_dir_path, f"{utc_stripped}_footprints.png")
    fig_footprint_grid.savefig(fig_path, bbox_inches='tight', pad_inches=0.05, dpi=150)
    # plt.show()
    plt.close(fig_footprint_grid)

    # *****************************************************************************************
    
    """ Mask global grids with the mutual footprint """
    
    grid_S_mutual = grid_S[mask_mutual_footprint]
    S_mutualfootprint = np.sum(grid_S_mutual) # km^2; surface area of the mutual footprint

    grid_lat_deg = grid_latlon[:,:,0][mask_mutual_footprint]
    grid_lon_deg = grid_latlon[:,:,1][mask_mutual_footprint]
    grid_lat = np.deg2rad(grid_lat_deg)
    grid_lon = np.deg2rad(grid_lon_deg)

    grid_r_gridpoint = mars_latlon_to_cartesian(grid_lon, grid_lat, radii_ell[0], radii_ell[2])
    grid_r_gridpointMEX = r_MEX - grid_r_gridpoint
    grid_r_gridpointTGO = r_TGO - grid_r_gridpoint
    grid_r_MEXgridpoint = grid_r_gridpoint - r_MEX
    grid_r_TGOgridpoint = grid_r_gridpoint - r_TGO

    grid_R_T = np.linalg.norm(grid_r_gridpointMEX, axis=1) # km
    grid_R_R = np.linalg.norm(grid_r_gridpointTGO, axis=1) # km

    grid_MEX_offbs_rad = vsep_vec(np.repeat(r_MEXbs[None, :], len(grid_r_MEXgridpoint), axis=0), grid_r_MEXgridpoint)
    grid_MEX_offbs_deg = np.rad2deg(grid_MEX_offbs_rad)

    grid_TGO_offbs_rad = vsep_vec(np.repeat(r_TGObs[None, :], len(grid_r_TGOgridpoint), axis=0), grid_r_TGOgridpoint)
    grid_TGO_offbs_deg = np.rad2deg(grid_TGO_offbs_rad)

    grid_G_T_dB = np.array([gaussian(angle, MEX_gaussian_parameters[0], MEX_gaussian_parameters[1], MEX_gaussian_parameters[2]) for angle in grid_MEX_offbs_deg])
    grid_G_T = 10**(grid_G_T_dB/10)
    grid_G_R_dB = np.array([gaussian(angle, TGO_gaussian_parameters[0], TGO_gaussian_parameters[1], TGO_gaussian_parameters[2]) for angle in grid_TGO_offbs_deg])
    grid_G_R = 10**(grid_G_R_dB/10)

    grid_s_RMS = grid_slopes_RMS[mask_mutual_footprint] # -; RMS surface slope
    min_nonzero_RMS = np.min(grid_s_RMS[grid_s_RMS > 0])
    grid_s_RMS[grid_s_RMS == 0] = min_nonzero_RMS # Replace zeros in s with the smallest nonzero value; avoid division by zero
    grid_C = grid_s_RMS**(-2)
    
    grid_n_hat_mutual = grid_n_hat[mask_mutual_footprint]
    grid_r_gridpointMEX_norm = np.linalg.norm(grid_r_gridpointMEX, axis=1)
    grid_r_gridpointMEX_hat = grid_r_gridpointMEX / grid_r_gridpointMEX_norm.reshape(-1, 1)
    grid_r_gridpointTGO_norm = np.linalg.norm(grid_r_gridpointTGO, axis=1)
    grid_r_gridpointTGO_hat = grid_r_gridpointTGO / grid_r_gridpointTGO_norm.reshape(-1, 1)
    grid_g = grid_r_gridpointMEX_hat + grid_r_gridpointTGO_hat
    grid_g_norm = np.linalg.norm(grid_g, axis=1)
    grid_g_hat = grid_g / grid_g_norm.reshape(-1, 1)

    ng_hat_dot = np.sum(grid_n_hat_mutual * grid_g_hat, axis=1)
    grid_gamma = np.arccos(ng_hat_dot)
    grid_gamma_deg = np.rad2deg(grid_gamma)

    grid_cos_gamma = np.cos(grid_gamma)
    grid_sin_gamma = np.sin(grid_gamma)

    # *****************************************************************************************
    
    """ Bistatic radar equation , Hagfors reduced model & Retrieval of the average permittivity """
    
    theta_i_deg = (emission_MEX + emission_TGO) / 2 # degrees; incidence angle (average equal-emission angle)
    theta_i = spice.convrt(theta_i_deg, 'DEGREES', 'RADIANS')

    dc = smp.symbols('dc')
    rho_perp_eq = ((smp.cos(theta_i) - smp.sqrt(dc - (smp.sin(theta_i))**2))/((smp.cos(theta_i) + smp.sqrt(dc - (smp.sin(theta_i))**2))))**2 
    rho_par_eq = (dc*smp.cos(theta_i) - smp.sqrt(dc - (smp.sin(theta_i))**2))/(dc*smp.cos(theta_i) + smp.sqrt(dc - (smp.sin(theta_i))**2))

    """ -----------> P_R data to dc estimation """

    P_R_data = P_R_data_W[j] # W; P_R data for the current time step
    AGC_data_j = AGC_data[j] # AGC data for the current time step
    rho_data = np.nan
    dc_data = np.nan

    """ -----------> dc model (GRS) to P_R estimation """

    grid_dc_modelGRS = grid_GRSdc[mask_mutual_footprint]
    dc_modelGRS = np.mean(grid_dc_modelGRS) # average permittivity of the model (GRS)

    rho_par_func = smp.lambdify(dc, rho_par_eq, modules=['numpy'])
    rho_perp_func = smp.lambdify(dc, rho_perp_eq, modules=['numpy'])
    grid_rho_par_modelGRS = rho_par_func(grid_dc_modelGRS)
    grid_rho_perp_modelGRS = rho_perp_func(grid_dc_modelGRS)
    grid_rho_modelGRS = (grid_rho_perp_modelGRS + grid_rho_par_modelGRS) / 2
    rho_modelGRS = np.mean(grid_rho_modelGRS) # average reflectivity of the model (GRS)

    grid_sigma0 = grid_rho_modelGRS * grid_C * (((grid_cos_gamma**4) + grid_C * (grid_sin_gamma**2))**(-3/2))
    grid_P_R_modelGRS = ((P_T * grid_G_T)/(4 * np.pi * ((grid_R_T * 10**3)**2))) * grid_sigma0 * ((grid_G_R * (lambda_UHF**2))/(4 * np.pi * ((grid_R_R * 10**3)**2))) * (grid_S_mutual*10**6)
    P_R_modelGRS = np.sum(np.array(grid_P_R_modelGRS)) # sum over all grid points in the mutual footprint

    """ -----------> dc model (FREND) to P_R estimation """

    grid_dc_modelFREND = grid_FRENDdc[mask_mutual_footprint]
    dc_modelFREND = np.mean(grid_dc_modelFREND) # average permittivity of the model (FREND)

    rho_par_func = smp.lambdify(dc, rho_par_eq, modules=['numpy'])
    rho_perp_func = smp.lambdify(dc, rho_perp_eq, modules=['numpy'])
    grid_rho_par_modelFREND = rho_par_func(grid_dc_modelFREND)
    grid_rho_perp_modelFREND = rho_perp_func(grid_dc_modelFREND)
    grid_rho_modelFREND = (grid_rho_perp_modelFREND + grid_rho_par_modelFREND) / 2
    rho_modelFREND = np.mean(grid_rho_modelFREND) # average reflectivity of the model (FREND)

    grid_topography = grid_C * (((grid_cos_gamma**4) + grid_C * (grid_sin_gamma**2))**(-3/2))
    grid_sigma0 = grid_rho_modelFREND * grid_topography
    grid_P_R_modelFREND = ((P_T * grid_G_T)/(4 * np.pi * ((grid_R_T * 10**3)**2))) * grid_sigma0 * ((grid_G_R * (lambda_UHF**2))/(4 * np.pi * ((grid_R_R * 10**3)**2))) * (grid_S_mutual*10**6)
    P_R_modelFREND = np.sum(grid_P_R_modelFREND) # sum over all grid points in the mutual footprint

    spoint_idx = np.argmin((grid_lon_deg - spoint_lon_deg)**2 + (grid_lat_deg - spoint_lat_deg)**2)
    P_R_modelGRS_spoint = grid_P_R_modelGRS[spoint_idx]
    P_R_modelFREND_spoint = grid_P_R_modelFREND[spoint_idx]

    """ -----------> Model P_R estimation (direct free-space transmission) """

    G_T_freespace_dB = gaussian(np.degrees(angle_MEX2TGO), MEX_gaussian_parameters[0], MEX_gaussian_parameters[1], MEX_gaussian_parameters[2])
    G_T_freespace = 10**(G_T_freespace_dB/10)
    G_R_freespace_dB = gaussian(np.degrees(angle_TGO2MEX), TGO_gaussian_parameters[0], TGO_gaussian_parameters[1], TGO_gaussian_parameters[2])
    G_R_freespace = 10**(G_R_freespace_dB/10)
    P_R_freespace = P_T * G_T_freespace * G_R_freespace * ((lambda_UHF/(4 * np.pi * r_MEXTGO * 1e3))**2) # W; P_R estimation for direct free-space transmission (Friis' transmission equation)

    # *******************************************************************************************

    for lon_gridpoint_j, lat_gridpoint_j, P_R_modelGRS_gridpoint_j, P_R_modelFREND_gridpoint_j in zip(grid_lon_deg, grid_lat_deg, grid_P_R_modelGRS, grid_P_R_modelFREND):
        gridpoints_file.write(f"{lon_gridpoint_j}, {lat_gridpoint_j}, {P_R_modelGRS_gridpoint_j}, {P_R_modelFREND_gridpoint_j} \n")

    gridpoints_file.close()

    # *******************************************************************************************

    """ Performance at the specular point """

    R_T_spoint = spice.vnorm(r_MEXspoint)
    R_R_spoint = spice.vnorm(r_TGOspoint)
    R_MEXspointTGO = R_T_spoint + R_R_spoint

    # *******************************************************************************************

    topography_spoint = grid_topography[spoint_idx]
    gamma_spoint = grid_gamma_deg[spoint_idx]
    C_spoint = grid_C[spoint_idx]

    lat_GRS_maxcont = grid_lat_deg[grid_P_R_modelGRS.argmax()]
    lon_GRS_maxcont = grid_lon_deg[grid_P_R_modelGRS.argmax()]

    lat_FREND_maxcont = grid_lat_deg[grid_P_R_modelFREND.argmax()]
    lon_FREND_maxcont = grid_lon_deg[grid_P_R_modelFREND.argmax()]

    topography_GRS_maxcont = grid_topography[grid_P_R_modelGRS.argmax()]
    gamma_GRS_maxcont = grid_gamma_deg[grid_P_R_modelGRS.argmax()]
    C_GRS_maxcont = grid_C[grid_P_R_modelGRS.argmax()]

    topography_FREND_maxcont = grid_topography[grid_P_R_modelFREND.argmax()]
    gamma_FREND_maxcont = grid_gamma_deg[grid_P_R_modelFREND.argmax()]
    C_FREND_maxcont = grid_C[grid_P_R_modelFREND.argmax()]

    rho_modelGRS_spoint = grid_rho_modelGRS[spoint_idx]
    dc_modelGRS_spoint = grid_dc_modelGRS[spoint_idx]
    rho_modelGRS_maxcont = grid_rho_modelGRS[grid_P_R_modelGRS.argmax()]
    dc_modelGRS_maxcont = grid_dc_modelGRS[grid_P_R_modelGRS.argmax()]

    rho_modelFREND_spoint = grid_rho_modelFREND[spoint_idx]
    dc_modelFREND_spoint = grid_dc_modelFREND[spoint_idx]
    rho_modelFREND_maxcont = grid_rho_modelFREND[grid_P_R_modelFREND.argmax()]
    dc_modelFREND_maxcont = grid_dc_modelFREND[grid_P_R_modelFREND.argmax()]

    line = f"{utc}, {et}, {R_T_spoint}, {R_R_spoint}, {R_MEXspointTGO}, {r_MEXTGO}, {spoint_lon_deg}, {spoint_lat_deg}, {midpoint_lon_deg}, {midpoint_lat_deg}, {emission_MEX}, {emission_TGO}, {theta_i_deg}, {S_mutualfootprint}, {rho_data}, {rho_modelGRS}, {rho_modelFREND}, {dc_data}, {dc_modelGRS}, {dc_modelFREND}, {AGC_data_j}, {P_R_data}, {P_R_modelGRS}, {P_R_modelFREND}, {np.nanmax(grid_P_R_modelGRS)}, {np.nanmax(grid_P_R_modelFREND)}, {np.nanmin(grid_P_R_modelGRS)}, {np.nanmin(grid_P_R_modelFREND)}, {P_R_modelGRS_spoint}, {P_R_modelFREND_spoint}, {P_R_freespace}, {gamma_spoint}, {C_spoint}, {topography_spoint}, {gamma_GRS_maxcont}, {C_GRS_maxcont}, {topography_GRS_maxcont}, {gamma_FREND_maxcont}, {C_FREND_maxcont}, {topography_FREND_maxcont}, {rho_modelGRS_spoint}, {dc_modelGRS_spoint}, {rho_modelGRS_maxcont}, {dc_modelGRS_maxcont}, {rho_modelFREND_spoint}, {dc_modelFREND_spoint}, {rho_modelFREND_maxcont}, {dc_modelFREND_maxcont}, {G_T_freespace}, {G_R_freespace}, {np.degrees(angle_MEX2TGO)}, {np.degrees(angle_TGO2MEX)}, {lat_GRS_maxcont}, {lon_GRS_maxcont}, {lat_FREND_maxcont}, {lon_FREND_maxcont}, {MEXbs_lon_deg}, {MEXbs_lat_deg}, {TGObs_lon_deg}, {TGObs_lat_deg}"

    manager_queue.put(line)

    # *******************************************************************************************

    spice.kclear()

    # *******************************************************************************************

    return line

# **********************************************************************************************

""" Main script/loop """

if __name__ == '__main__':

    # ******************************************************************************************

    """ Script settings """
    
    n_pixels_slopes = 4 # Number of pixels in the slope data (Options: 4, 16 or 32)
    polar_host = 'co2' # Polar host material (Options: 'rego', 'co2')

    # ******************************************************************************************

    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("\nScript started at:", start_time_str)
    print(f"\tHost material polar regions: {polar_host}")
    print(f"\tResolution (from MOLA data): {n_pixels_slopes} pixels per degree")
    
    # ******************************************************************************************
    
    """ Setup paths for loading data and saving results """
    """ ----- Measurement tracks ----------------------------------------------------------- """

    code_and_sim_path = os.path.join(os.path.dirname(os.getcwd()),
                                    "thesis\\code_and_simulations")

    analysis_path = os.path.join(code_and_sim_path, "analysis")
    measop_txt_path = os.path.join(analysis_path, "spoint_tracks\\meas_ops.txt")

    with open(measop_txt_path, 'r') as measop_txt:
        measop_txt_lines = measop_txt.readlines()
        measop_txt_list = measop_txt_lines[0].split()

    measop_file_paths = []
    for i in range(len(measop_txt_list)):
        measop_file_path = os.path.join(analysis_path,
                                        f"spoint_tracks\\meas_ops\\{measop_txt_list[i]}")
        measop_file_paths.append(measop_file_path)

    """ ----- BSR measurement data titles, file names and paths ---------------------------- """

    BSR_titles = ['BSR-1.1', 'BSR-1.2', 'BSR-1.3', 'BSR-1.4',
                'BSR-2.1', 'BSR-2.2', 'BSR-3.1', 'BSR-4.1']
    MTP_titles = ['MTP257/76', 'MTP257/76', 'MTP257/76', 'MTP257/76',
                  'MTP272/91', 'MTP272/91',
                  'MTP276/95',
                  'MTP277/96']

    IQ_titles = ['IQ___DMEX__05376B74_2024-023T03-48-39_8F540240230137_00001.EXM',
                'IQ___DMEX__05376BCD_2024-030T10-30-15_8F540240300137_00001.EXM',
                'IQ___DMEX__05376C22_2024-037T09-42-58_8F540240370137_00001.EXM',
                'IQ___DMEX__05376C5B_2024-042T01-53-04_8F540240420137_00001.EXM',
                'IQ___DMEX__05377FA0_2025-079T23-40-39_8F540250790137_00001.EXM',
                'IQ___DMEX__05378060_2025-095T16-55-49_8F540250950137_00001.EXM',
                'IQ___DMEX__05378500_2025-192T16-16-12_8F540251920137_00001.EXM',
                'IQ___DMEX__053786AD_2025-227T19-38-57_8F540252270137_00001.EXM']
    
    """ ----- MOLA-retrieved slope data paths and model resolution ------------------------- """

    slopes_dir_path = os.path.join(analysis_path, "MOLA")
    slopes_file_path = os.path.join(slopes_dir_path, f"{n_pixels_slopes}pixel_slopes.csv")

    area_discretization = 1/n_pixels_slopes # Accuracy in degrees longitude and latitude
                                            # Defined by MOLA slope resolution
    
    n_lat_options = int(180//area_discretization)
    n_lon_options = int(360//area_discretization)
    standard_grid = np.zeros((n_lat_options, n_lon_options))

    lat_deg_options = np.linspace(-90 + area_discretization/2, 90 - area_discretization/2, n_lat_options)  # centers of latitudinal grid cells
    lon_deg_options = np.linspace(-180 + area_discretization/2, 180 - area_discretization/2, n_lon_options)  # centers of longitudinal grid cells


    """ ----- Mars background and parameter settings for plots ---------------------------------------------------- """

    img_MarsBG_Path = os.path.join(code_and_sim_path,
                                "data\\HRSC-MOLA_Blended-DEM\\RecBG_dem_Mars_1500mp.png")
    img_MarsBackground = plt.imread(img_MarsBG_Path)
    img_MarsBG_Greyscale_Path = os.path.join(code_and_sim_path,
                                    "data\\HRSC-MOLA_Blended-DEM\\RecBG_dem_Mars_1500mp_Greyscale.png")
    img_MarsBackground_Greyscale = plt.imread(img_MarsBG_Greyscale_Path)

    rcParams['font.size'] = 14
    rcParams['axes.prop_cycle'] = cycler(color=plt.cm.tab20.colors)
    
    # ******************************************************************************************
    
    """ Load SPICE kernels """
    
    MetaKernelsPath = os.path.join(code_and_sim_path, "kernels\\mk")

    main_gen_mk = os.path.join(MetaKernelsPath, 'gen_mk.tm') # General kernels
    main_MEX_mk = os.path.join(MetaKernelsPath, 'MEX_mk.tm') # MEX-specific kernels
    main_TGO_mk = os.path.join(MetaKernelsPath, 'TGO_mk.tm') # TGO-specific kernels

    spice.furnsh(main_gen_mk)
    spice.furnsh(main_MEX_mk)
    spice.furnsh(main_TGO_mk)
    
    # ******************************************************************************************
    
    """ Define global Mars and simulation parameters"""
    """ ----- Signal parameters ------------------------------------------------------------ """

    P_T = BSR_param['P_T']  # W; transmitter (MEX) power
    lambda_UHF = BSR_param['lambda_UHF']  # m; wavelength  
    P_noisefloor_dB = BSR_param['P_noisefloor_dB']  # dB; noise floor of the receiver (TGO)  
    
    MEX_gaussian_parameters = calc_gain_gaussian(BSR_param['MEX_antennapattern'][0],
                                                BSR_param['MEX_antennapattern'][1])
    
    TGO_gaussian_parameters = calc_gain_gaussian(BSR_param['TGO_antennapattern'][0],
                                                BSR_param['TGO_antennapattern'][1])
    
    """ ----- Mars ellipsoidal parameters --------------------------------------------------- """
    
    radii_ell = Mars_param['radii_ell'] # km; ellipsoidal Mars radii

    # ******************************************************************************************

    """ Retrieve global Mars grids """

    grid_latlon, grid_S, grid_GRSdc, grid_FRENDdc, grid_slopes, grid_slopes_RMS, grid_n_hat = global_grids(area_discretization, standard_grid, lat_deg_options, lon_deg_options, polar_host)

    # ******************************************************************************************

    """ Loop through BSR measurement files """
    
    measop_files_i = np.arange(len(measop_file_paths))

    for i in measop_files_i:
        # **************************************************************************************

        print(f"\nProcessing file: {measop_txt_list[i]}...")

        measop_data = readout_measop(measop_file_paths[i])
        et_steps = measop_data['et'][:]
        
        # **************************************************************************************

        """ Setup paths for loading data and saving results """

        BSR_title_i = BSR_titles[i]
        MTP_title_i = MTP_titles[i]
        BSR_dir_path = os.path.join(analysis_path, f"results\\{BSR_title_i}")

        BSRperm_dir_path = os.path.join(BSR_dir_path, f'permittivity-{polar_host}-{n_pixels_slopes}pix')
        BSRperm_data_dir_path = os.path.join(BSRperm_dir_path, 'data')
        BSRperm_visuals_dir_path = os.path.join(BSRperm_dir_path, 'visuals')

        BSRfp_dir_path = os.path.join(BSR_dir_path, 'footprints')
        BSRfp_data_dir_path = os.path.join(BSRfp_dir_path, 'data')
        BSRfp_visuals_dir_path = os.path.join(BSRfp_dir_path, 'visuals')

        BSR_dirs_to_create = [BSR_dir_path, 
                            BSRperm_dir_path, BSRperm_data_dir_path, BSRperm_visuals_dir_path,
                            BSRfp_dir_path, BSRfp_data_dir_path, BSRfp_visuals_dir_path]
        for BSR_dir in BSR_dirs_to_create:
            os.makedirs(BSR_dir, exist_ok=True)

        BSR_file_name_csv = f"{BSR_title_i}_permittivity_{area_discretization}deg.csv"
        BSR_file_path_csv = os.path.join(BSRperm_dir_path, BSR_file_name_csv)

        # **************************************************************************************

        AGC_data, P_R_data_dB, P_R_data_W = readout_IQ(BSR_title_i, IQ_titles[i], target_length=len(et_steps), AGC_baseline=124)

        # **************************************************************************************
        
        """ Run through et steps of the measurement """
        MEXbs_track = [[],[]]
        TGObs_track = [[],[]]

        for j in range(len(et_steps)):
            et = measop_data['et'][j]
            
            r_MEX = spice.spkpos('MEX', et, 'IAU_MARS', 'LT+S', 'MARS')[0] # km; MEX position
            r_TGO = spice.spkpos('TGO', et, 'IAU_MARS', 'LT+S', 'MARS')[0] # km; TGO position

            MEXbs_lon, MEXbs_lat = spice.reclat(r_MEX)[1:] # radians; MEX boresight in conventional attitude: nadir-pointing
            MEXbs_lon_deg, MEXbs_lat_deg = spice.convrt([MEXbs_lon, MEXbs_lat], 'RADIANS', 'DEGREES') # degrees; MEX boresight in conventional attitude: nadir-pointing
            TGObs_lon, TGObs_lat = spice.reclat(r_TGO)[1:] # radians; TGO boresight in conventional attitude: nadir-pointing 
            TGObs_lon_deg, TGObs_lat_deg = spice.convrt([TGObs_lon, TGObs_lat], 'RADIANS', 'DEGREES') # degrees; TGO boresight in conventional attitude: nadir-pointing  

            MEXbs_track[0].append(MEXbs_lon_deg)
            MEXbs_track[1].append(MEXbs_lat_deg)
            TGObs_track[0].append(TGObs_lon_deg)
            TGObs_track[1].append(TGObs_lat_deg)

        # **************************************************************************************

        """ Multiprocessing setup """

        manager = mp.Manager()
        manager_queue = manager.Queue()
        if n_pixels_slopes == 4:
            mp_processes = mp.cpu_count() - 2
        elif n_pixels_slopes > 4:
            mp_processes = mp.cpu_count() - 6
        pool = mp.Pool(mp_processes)
        watcher_process = mp.Process(target=multiproc_filewriter, args=(manager_queue, BSR_file_path_csv))
        watcher_process.start()

        multiproc_etsteps_partial = partial(multiproc_etsteps, 
                            measop_data=measop_data,
                            BSRperm_data_dir_path=BSRperm_data_dir_path, 
                            standard_grid=standard_grid,
                            BSRfp_data_dir_path=BSRfp_data_dir_path,
                            lat_deg_options=lat_deg_options,
                            lon_deg_options=lon_deg_options,
                            code_and_sim_path=code_and_sim_path,
                            img_MarsBackground=img_MarsBackground,
                            MEXbs_track=MEXbs_track,
                            TGObs_track=TGObs_track,
                            area_discretization=area_discretization,
                            BSRfp_visuals_dir_path=BSRfp_visuals_dir_path,
                            grid_S=grid_S,
                            grid_latlon=grid_latlon,
                            MEX_gaussian_parameters=MEX_gaussian_parameters,
                            TGO_gaussian_parameters=TGO_gaussian_parameters,
                            grid_GRSdc=grid_GRSdc,
                            grid_FRENDdc=grid_FRENDdc,
                            grid_slopes=grid_slopes,
                            grid_slopes_RMS=grid_slopes_RMS,
                            grid_n_hat=grid_n_hat,
                            P_R_data_W=P_R_data_W,
                            AGC_data=AGC_data,
                            P_T=P_T,
                            lambda_UHF=lambda_UHF,
                            radii_ell=radii_ell,
                            BSR_title_i=BSR_title_i,
                            MTP_title_i=MTP_title_i,
                            n_pixels_slopes=n_pixels_slopes
                            )

        jobs = []
        for j in range(len(et_steps)):
            job = pool.apply_async(multiproc_etsteps_partial, (j, manager_queue))
            jobs.append(job)

        for job in tqdm(jobs):
            job.get()

        pool.close()
        pool.join()
        manager_queue.put('STOP')
        watcher_process.join()
        
        # ***********************************************************************************

        """ Sort the BSR permittivity data """

        BSR_dataframe_unsorted = pd.read_csv(BSR_file_path_csv, header=[0])
        BSR_dataframe_unsorted.columns = ['utc'] + [col.split()[0].strip() for col in BSR_dataframe_unsorted.columns[1:]]
        BSR_data_sorted = BSR_dataframe_unsorted.sort_values(by='et', ascending=True)

        with open(BSR_file_path_csv, 'w') as BSR_file:
            BSR_file.write(f"# utc [-], et [s], R_T_spoint [km], R_R_spoint [km], R_MEXspointTGO [km], r_MEXTGO [km], spoint_lon [deg], spoint_lat [deg], midpoint_lon [deg], midpoint_lat [deg], emis_MEX [deg], emis_TGO [deg], emis_avg [deg], S_mutualfootprint [km^2], rho_data [-], rho_modelGRS [-], rho_modelFREND [-], dc_data [-], dc_modelGRS [-], dc_modelFREND [-], AGC_data [-], P_R_data [W], P_R_modelGRS [W], P_R_modelFREND [W], P_R_modelGRS_maxcont [W], P_R_modelFREND_maxcont [W], P_R_modelGRS_mincont [W], P_R_modelFREND_mincont [W], P_R_modelGRS_spoint [W], P_R_modelFREND_spoint [W], P_R_freespace [W], gamma_spoint [deg], C_spoint [-], topography_spoint [-], gamma_GRS_maxcont [deg], C_GRS_maxcont [-], topography_GRS_maxcont [-], gamma_FREND_maxcont [deg], C_FREND_maxcont [-], topography_FREND_maxcont [-], rho_modelGRS_spoint [-], dc_modelGRS_spoint [-], rho_modelGRS_maxcont [-], dc_modelGRS_maxcont [-], rho_modelFREND_spoint [-], dc_modelFREND_spoint [-], rho_modelFREND_maxcont [-], dc_modelFREND_maxcont [-], G_T_freespace [-], G_R_freespace [-], angle_MEX2TGO [deg], angle_TGO2MEX [deg], lat_GRS_maxcont [deg], lon_GRS_maxcont [deg], lat_FREND_maxcont [deg], lon_FREND_maxcont [deg], MEXbs_lon [deg], MEXbs_lat [deg], TGObs_lon [deg], TGObs_lat [deg] \n")

            for row in BSR_data_sorted.itertuples(index=False):
                line = ', '.join(str(item) for item in row)
                BSR_file.write(line + '\n')

        # ***********************************************************************************

        BSRfp_gif_path = os.path.join(BSRfp_dir_path, 'animation_footprints.gif')
        create_gif(BSRfp_visuals_dir_path, "footprints", frame_duration_ms=30, gif_path=BSRfp_gif_path)

        # ************************************************************************************
        
    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    run_time_s = end_time - start_time
    run_time_m = run_time_s / 60

    print("\nScript ended at:", end_time_str)

    if run_time_s < 100:
        print("Total runtime:", round(run_time_s,2), "seconds\n")
    else:
        print("Total runtime:", round(run_time_m,2), "minutes\n")

    # ******************************************************************************************
    
    spice.kclear()

    # ******************************************************************************************
