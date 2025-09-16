# **********************************************************************************************
#
#  main_MEXTGOBSR_visualizeresults.py
#
# **********************************************************************************************
#
#  Description: 
#  |  Visualisation of BSR analysis results, options include:
#     |  AGC_plot: For plotting out the AGC data and converted power data (from IQ files)
#     |  gridpoint_plot: For plotting out the power distributions over the mutual signal 
#        footprints (from gridpoint data files)
#     |  BSR_plot: For plotting out the BSR parameters and model results over time (from 
#        high-level BSR file)
#     |  BSR_plot_MatComp: For plotting out the comparison between regolith and co2 ice as 
#        polar host materials (from high-level BSR file)
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

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pandas as pd
from scipy.signal import savgol_filter
import spiceypy as spice
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm

from main_MEXTGOBSR_model import find_footprint, plot_split_footprint
from my_functions import create_gif
from my_parameters import BSR_param

# **********************************************************************************************

""" Visualization settings """

n_pixels = 16

AGC_plot = False
gridpoint_plot = True   
BSR_plot = False
BSR_plot_MatComp = False

# **********************************************************************************************

""" Directory paths, file / BSR titles and parameters """

code_and_sim_path = os.path.join(os.path.dirname(os.getcwd()),
                                "thesis\\code_and_simulations")
analysis_path = os.path.join(code_and_sim_path, "analysis")
IQ_dir_path = os.path.join(code_and_sim_path, "data\\BSR")

BSR_titles = ['BSR-1.1', 'BSR-1.2', 'BSR-1.3', 'BSR-1.4',
                'BSR-2.1', 'BSR-2.2', 'BSR-3.1', 'BSR-4.1']
IQ_titles = ['IQ___DMEX__05376B74_2024-023T03-48-39_8F540240230137_00001.EXM',
             'IQ___DMEX__05376BCD_2024-030T10-30-15_8F540240300137_00001.EXM',
             'IQ___DMEX__05376C22_2024-037T09-42-58_8F540240370137_00001.EXM',
             'IQ___DMEX__05376C5B_2024-042T01-53-04_8F540240420137_00001.EXM',
             'IQ___DMEX__05377FA0_2025-079T23-40-39_8F540250790137_00001.EXM',
             'IQ___DMEX__05378060_2025-095T16-55-49_8F540250950137_00001.EXM',
             'IQ___DMEX__05378500_2025-192T16-16-12_8F540251920137_00001.EXM',
             'IQ___DMEX__053786AD_2025-227T19-38-57_8F540252270137_00001.EXM']
BSR_titles_i = np.arange(len(BSR_titles))

lon_0, lon_f = -180, 180
lat_0, lat_f = -90, 90

img_MarsBG_Path = os.path.join(code_and_sim_path,
                                "data\\HRSC-MOLA_Blended-DEM\\RecBG_dem_Mars_1500mp.png")
img_MarsBackground = plt.imread(img_MarsBG_Path)

img_MarsBG_Greyscale_Path = os.path.join(code_and_sim_path,
                                "data\\HRSC-MOLA_Blended-DEM\\RecBG_dem_Mars_1500mp_Greyscale.png")
img_MarsBackground_Greyscale = plt.imread(img_MarsBG_Greyscale_Path)

P_noisefloor_W = BSR_param['P_noisefloor_W']
P_noisefloor_dB = BSR_param['P_noisefloor_dB']
AGC_baseline = BSR_param['AGC_baseline']

area_discretization = 1/n_pixels  # [deg]

# **********************************************************************************************

""" Functions """

def read_IQ_file(file_path):
    """
    read_IQ_file: Function used to readout the AGC data from IQ files, including conversion from 
        8 bits to unsigned integer values (0 - 255) 

    :param file_path: Location of the IQ file
    :type file_path: class 'str'
    :return: agc_data_NTC32: Unsigned integer values of the AGC data (0 - 255)
    :rtype: class 'np.ndarray'
    """

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

    agc_bits = words_NTC32 >> 4 # Moves AGC bits from bits 20:28, i.e. 4:11 from the right, to bits 16:24, i.e. 0:8 from the right
    agc_data_NTC32 = agc_bits & 255 # Mask to keep only bits 20:28, i.e. 4:11 from the right; 255 = 11111111, i.e. 8 bits

    return agc_data_NTC32

# **********************************************************************************************

""" Load SPICE kernels """

MetaKernelsPath = os.path.join(code_and_sim_path, "kernels\\mk")

main_gen_mk = os.path.join(MetaKernelsPath, 'gen_mk.tm') # General kernels
main_MEX_mk = os.path.join(MetaKernelsPath, 'MEX_mk.tm') # MEX-specific kernels
main_TGO_mk = os.path.join(MetaKernelsPath, 'TGO_mk.tm') # TGO-specific kernels

spice.furnsh(main_gen_mk)
spice.furnsh(main_MEX_mk)
spice.furnsh(main_TGO_mk)

# **********************************************************************************************

""" Prepare dictionaries for maximum off-boresight angles """
""" ----- At edge (horizon) of the footprint ----------------------------------------------- """
MEX_intpoint_angles_max = {'BSR-1.1': [], 'BSR-1.2': [], 'BSR-1.3': [], 'BSR-1.4': [],
            'BSR-2.1': [], 'BSR-2.2': [], 'BSR-3.1': [], 'BSR-4.1': []}
TGO_intpoint_angles_max = {'BSR-1.1': [], 'BSR-1.2': [], 'BSR-1.3': [], 'BSR-1.4': [],
            'BSR-2.1': [], 'BSR-2.2': [], 'BSR-3.1': [], 'BSR-4.1': []}

""" ----- Toward other S/C in free space --------------------------------------------------- """
MEX_freespaceoffbs_angles_max = {'BSR-1.1': [], 'BSR-1.2': [], 'BSR-1.3': [], 'BSR-1.4': [],
            'BSR-2.1': [], 'BSR-2.2': [], 'BSR-3.1': [], 'BSR-4.1': []}
TGO_freespaceoffbs_angles_max = {'BSR-1.1': [], 'BSR-1.2': [], 'BSR-1.3': [], 'BSR-1.4': [],
            'BSR-2.1': [], 'BSR-2.2': [], 'BSR-3.1': [], 'BSR-4.1': []}

# **********************************************************************************************

""" Loop through BSR measurements """

for BSR_i in BSR_titles_i:
 
    print(f"\nVisualising results for {BSR_titles[BSR_i]}")

    BSR_dir_path = os.path.join(analysis_path, f"results\\{BSR_titles[BSR_i]}")

    P_R_modelGRS_dict = {'co2': None,
                        'rego': None}
    P_R_modelGRS_spoint_dict = {'co2': None,
                                'rego': None}
    P_R_modelGRS_maxcont_dict = {'co2': None,
                                'rego': None}

    for mat in ['rego', 'co2']:

        print(f"\tPolar host material: {mat}")
        print(f"\tSpatial resolution: {n_pixels} pixels per degree")

        BSRperm_dir_path = os.path.join(BSR_dir_path, f'permittivity-{mat}-{n_pixels}pix')
        BSRperm_data_dir_path = os.path.join(BSRperm_dir_path, 'data')
        BSRperm_visuals_dir_path = os.path.join(BSRperm_dir_path, 'visuals')

        BSRfp_dir_path = os.path.join(BSR_dir_path, 'footprints')
        BSRfp_data_dir_path = os.path.join(BSRfp_dir_path, 'data')
        BSRfp_visuals_dir_path = os.path.join(BSRfp_dir_path, 'visuals')

        # *****************************************************************************************

        """ Readout BSR model results"""
        """ ----- High-level file (for complete footprint) ----------------------------------- """

        BSR_file_name_csv = f"{BSR_titles[BSR_i]}_permittivity_{area_discretization}deg.csv"
        BSR_file_path_csv = os.path.join(BSRperm_dir_path, BSR_file_name_csv)

        BSR_dataframe = pd.read_csv(BSR_file_path_csv, header=[0])
        BSR_dataframe.columns = ['utc'] + [col.split()[0].strip() for col in BSR_dataframe.columns[1:]]
        BSR_data = {col: BSR_dataframe[col].values for col in BSR_dataframe.columns}

        processed_data = {}
        for key in BSR_data:
            if key in ['utc', 'et', 'R_T_spoint', 'R_R_spoint', 'R_MEXspointTGO', 'emis_MEX', 'emis_TGO', 'emis_avg', 'S_mutualfootprint']:
                processed_data[key] = BSR_data[key][:]
            else:
                values = []
                for x in BSR_data[key][:]:
                    if isinstance(x, str):
                        if x.strip().lower() != 'nan':
                            values.append(float(x.strip()))
                        else:
                            values.append(np.nan)
                    else:
                        values.append(x)
                processed_data[key] = np.array(values)
        
        utc = processed_data['utc']
        et = processed_data['et']

        R_T_spoint = processed_data['R_T_spoint']
        R_R_spoint = processed_data['R_R_spoint']
        R_MEXspointTGO = processed_data['R_MEXspointTGO']
        r_MEXTGO = processed_data['r_MEXTGO']

        spoint_lon_deg = processed_data['spoint_lon']
        spoint_lat_deg = processed_data['spoint_lat']
        midpoint_lon_deg = processed_data['midpoint_lon']
        midpoint_lat_deg = processed_data['midpoint_lat']

        emission_MEX = processed_data['emis_MEX']
        emission_TGO = processed_data['emis_TGO']
        emis_avg = processed_data['emis_avg']
        
        S_mutualfootprint = processed_data['S_mutualfootprint']

        rho_data = processed_data['rho_data']
        rho_modelGRS = processed_data['rho_modelGRS']
        rho_modelFREND = processed_data['rho_modelFREND']
        
        dc_data = processed_data['dc_data']
        dc_modelGRS = processed_data['dc_modelGRS']
        dc_modelFREND = processed_data['dc_modelFREND']

        AGC_data = processed_data['AGC_data']
        P_R_data = processed_data['P_R_data']
        P_R_modelGRS = processed_data['P_R_modelGRS']
        P_R_modelGRS_dict[mat] = P_R_modelGRS.copy()
        P_R_modelFREND = processed_data['P_R_modelFREND']

        P_R_modelGRS_maxcont = processed_data['P_R_modelGRS_maxcont']
        P_R_modelGRS_maxcont_dict[mat] = P_R_modelGRS_maxcont.copy()
        P_R_modelFREND_maxcont = processed_data['P_R_modelFREND_maxcont']
        P_R_modelGRS_mincont = processed_data['P_R_modelGRS_mincont']
        P_R_modelFREND_mincont = processed_data['P_R_modelFREND_mincont']

        P_R_modelGRS_maxcont_max = np.nanmax(P_R_modelGRS_maxcont)
        P_R_modelFREND_maxcont_max = np.nanmax(P_R_modelFREND_maxcont)
        P_R_modelGRS_mincont_min = np.nanmin(P_R_modelGRS_mincont)
        P_R_modelFREND_mincont_min = np.nanmin(P_R_modelFREND_mincont)

        P_R_modelGRS_spoint = processed_data['P_R_modelGRS_spoint']
        P_R_modelGRS_spoint_dict[mat] = P_R_modelGRS_spoint.copy()
        P_R_modelFREND_spoint = processed_data['P_R_modelFREND_spoint']

        P_R_freespace = processed_data['P_R_freespace']

        P_R_combinedGRS = P_R_modelGRS + P_R_freespace
        P_R_combinedFREND = P_R_modelFREND + P_R_freespace

        gamma_spoint = processed_data['gamma_spoint']
        C_spoint = processed_data['C_spoint']
        topography_spoint = processed_data['topography_spoint']

        gamma_GRS_maxcont = processed_data['gamma_GRS_maxcont']
        C_GRS_maxcont = processed_data['C_GRS_maxcont']
        topography_GRS_maxcont = processed_data['topography_GRS_maxcont']

        gamma_FREND_maxcont = processed_data['gamma_FREND_maxcont']
        C_FREND_maxcont = processed_data['C_FREND_maxcont']
        topography_FREND_maxcont = processed_data['topography_FREND_maxcont']

        rho_modelGRS_spoint = processed_data['rho_modelGRS_spoint']
        dc_modelGRS_spoint = processed_data['dc_modelGRS_spoint']
        rho_modelGRS_maxcont = processed_data['rho_modelGRS_maxcont']
        dc_modelGRS_maxcont = processed_data['dc_modelGRS_maxcont']

        rho_modelFREND_spoint = processed_data['rho_modelFREND_spoint']
        dc_modelFREND_spoint = processed_data['dc_modelFREND_spoint']
        rho_modelFREND_maxcont = processed_data['rho_modelFREND_maxcont']
        dc_modelFREND_maxcont = processed_data['dc_modelFREND_maxcont']

        G_T_freespace = processed_data['G_T_freespace']
        G_R_freespace = processed_data['G_R_freespace']
        angle_MEX2TGO = processed_data['angle_MEX2TGO'] # deg
        angle_TGO2MEX = processed_data['angle_TGO2MEX'] # deg
        MEX_freespaceoffbs_angles_max[BSR_titles[BSR_i]] = angle_MEX2TGO.copy()
        TGO_freespaceoffbs_angles_max[BSR_titles[BSR_i]] = angle_TGO2MEX.copy()

        lat_GRS_maxcont = processed_data['lat_GRS_maxcont'] # deg
        lon_GRS_maxcont = processed_data['lon_GRS_maxcont'] # deg
        lat_FREND_maxcont = processed_data['lat_FREND_maxcont'] # deg
        lon_FREND_maxcont = processed_data['lon_FREND_maxcont'] # deg
        
        MEXbs_lon = processed_data['MEXbs_lon'] # deg
        MEXbs_lat = processed_data['MEXbs_lat'] # deg
        TGObs_lon = processed_data['TGObs_lon'] # deg
        TGObs_lat = processed_data['TGObs_lat'] # deg

        """ ----- Low-level file (per gridpoint) --------------------------------------------- """

        if gridpoint_plot:
            print("Reading out gridpoint data...")

            n_lat_options = int(180//area_discretization)
            n_lon_options = int(360//area_discretization)

            grid_latlon = np.zeros((n_lat_options, n_lon_options, 2))
            lat_deg_options = np.linspace(-90 + area_discretization/2, 90 - area_discretization/2, n_lat_options)  # centers of latitudinal grid cells
            lon_deg_options = np.linspace(-180 + area_discretization/2, 180 - area_discretization/2, n_lon_options)  # centers of longitudinal grid cells

            for i in range(n_lat_options):
                for j in range(n_lon_options):
                    grid_latlon[i, j, 0] = lat_deg_options[i]
                    grid_latlon[i, j, 1] = lon_deg_options[j]

            for grid_i in tqdm(range(len(utc))):
                utc_i = utc[grid_i]

                gridpoints_file_name = f"gridpoints_{utc_i.replace(' ', '.').replace(':', '.')}"
                
                gridpoints_file_name_csv = f"{gridpoints_file_name}.csv"
                gridpoints_file_path_csv = os.path.join(BSRperm_data_dir_path, f"{gridpoints_file_name_csv}")

                gridpoint_dataframe = pd.read_csv(gridpoints_file_path_csv, header=[0])
                gridpoint_dataframe.columns = ['lon_gridpoint'] + [col.split()[0].strip() for col in gridpoint_dataframe.columns[1:]]
                gridpoint_data = {col: gridpoint_dataframe[col].values for col in gridpoint_dataframe.columns}

                P_R_modelGRS_gridpoints = gridpoint_data['P_R_modelGRS']
                P_R_modelFREND_gridpoints = gridpoint_data['P_R_modelFREND']

                grid_P_R_modelGRS = np.zeros((n_lat_options, n_lon_options))
                grid_P_R_modelFREND = np.zeros((n_lat_options, n_lon_options))

                for idx, (lat, lon) in enumerate(zip(gridpoint_data["lat_gridpoint"], gridpoint_data["lon_gridpoint"])):
                    i = np.where(lat_deg_options == lat)[0][0]
                    j = np.where(lon_deg_options == lon)[0][0]
                    grid_P_R_modelGRS[i, j] = P_R_modelGRS_gridpoints[idx]
                    grid_P_R_modelFREND[i, j] = P_R_modelFREND_gridpoints[idx]

                mask_nondata_GRS = (grid_P_R_modelGRS == 0.0)
                mask_nondata_FREND = (grid_P_R_modelFREND == 0.0)
                grid_P_R_modelGRS[mask_nondata_GRS] = np.nan
                grid_P_R_modelFREND[mask_nondata_FREND] = np.nan

                footprint_file_name = 'footprint_' + utc_i.replace(" ", ".").replace(":", ".") + '.csv'
                footprint_file_path = os.path.join(BSRfp_data_dir_path, footprint_file_name)
                footprint_dataframe = pd.read_csv(footprint_file_path, header=0)
                footprint_dataframe.columns = ['MEX_intpoint_lon'] + [col.split()[0].strip() for col in footprint_dataframe.columns[1:]]
                footprint_data = {col: footprint_dataframe[col].values for col in footprint_dataframe.columns}

                MEX_intpoint_angle = footprint_data['MEX_intpoint_angle']
                TGO_intpoint_angle = footprint_data['TGO_intpoint_angle']
                MEX_intpoint_angles_max[BSR_titles[BSR_i]].append(np.max(MEX_intpoint_angle))
                TGO_intpoint_angles_max[BSR_titles[BSR_i]].append(np.max(TGO_intpoint_angle))

                MEX_intpoint_lon = footprint_data['MEX_intpoint_lon']
                MEX_intpoint_lat = footprint_data['MEX_intpoint_lat']
                TGO_intpoint_lon = footprint_data['TGO_intpoint_lon']
                TGO_intpoint_lat = footprint_data['TGO_intpoint_lat']
                
                footprint_grid = np.zeros((n_lat_options, n_lon_options))
                et_i = spice.utc2et(utc_i)
                r_MEX = spice.spkpos('MEX', et_i, 'IAU_MARS', 'LT+S', 'MARS')[0] # km; MEX position
                r_TGO = spice.spkpos('TGO', et_i, 'IAU_MARS', 'LT+S', 'MARS')[0] # km; TGO position
                MEXbs_lat = spice.reclat(r_MEX)[1:][1] # radians; MEX boresight in conventional attitude: nadir-pointing
                TGObs_lat = spice.reclat(r_TGO)[1:][1] # radians; TGO boresight in conventional attitude: nadir-pointingng

                grid_MEX_footprint, MEX_intpoint_lat, MEX_intpoint_lon = find_footprint(footprint_grid.copy(), lat_deg_options, lon_deg_options, MEX_intpoint_lat, MEX_intpoint_lon, MEXbs_lat)
                grid_TGO_footprint, TGO_intpoint_lat, TGO_intpoint_lon = find_footprint(footprint_grid.copy(), lat_deg_options, lon_deg_options, TGO_intpoint_lat, TGO_intpoint_lon, TGObs_lat)

                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(111)
                ax.imshow(img_MarsBackground_Greyscale, extent=[lon_0, lon_f, lat_0, lat_f])
                ax.scatter(spoint_lon_deg[grid_i], spoint_lat_deg[grid_i], facecolors='none', edgecolors='black', marker='*', s=70, linewidths=0.3)
                # ax.scatter(spoint_lon_deg[grid_i], spoint_lat_deg[grid_i], facecolors='none', edgecolors='black', marker='*', s=250, linewidths=0.8)
                masked_grid = np.ma.masked_where(grid_P_R_modelGRS == 0, grid_P_R_modelGRS)
                ax.set_xlabel('Longitude [$\degree$E]')
                ax.set_ylabel('Latitude [$\degree$N]')
                title = f'Power distribution over the mutual footprint for {BSR_titles[BSR_i]} (at {n_pixels} pixel/degree resolution)'
                fig.text(s=title, x=0.47, y=0.998, fontsize=14, ha='center', va='center', fontweight='bold')
                fig.text(s='(Background: HRSC_MOLA Blended DEM at 1500mp)', x=0.47, y=0.963, fontsize=12, ha='center', va='center')  

                fig.text(x=0.96, y=0.995, s=f'UTC =', fontsize=12, ha='center', va='center', fontweight='bold')
                fig.text(x=0.96, y=0.95, s=f'{utc_i[:11]}\n{utc_i[12:]}', fontsize=12, ha='center', va='center')
                fig.text(x=0.96, y=0.07, s=f't_step =', fontsize=12, ha='center', va='center', fontweight='bold')
                fig.text(x=0.96, y=0.035, s=f'{grid_i+1}', fontsize=12, ha='center', va='center')

                img = ax.imshow(np.flipud(masked_grid),
                                extent=[lon_0, lon_f, lat_0, lat_f],
                                cmap='plasma_r',
                                alpha=1.0,
                                norm=LogNorm(vmin=P_R_modelGRS_mincont_min, vmax=P_R_modelGRS_maxcont_max))
                cbar = plt.colorbar(img,
                                    label='Received power contribution [W]',
                                    ax=ax,
                                    fraction=0.022,
                                    pad=0.02)

                plot_split_footprint(MEX_intpoint_lon, MEX_intpoint_lat, SC_color='red')
                plot_split_footprint(TGO_intpoint_lon, TGO_intpoint_lat, SC_color='cyan')
                plt.grid(True, color='black', alpha=0.1)
                plt.tight_layout(rect=[-0.02, -0.02, 1.01, 0.98])
                gridpoints_file_name_png = f"{gridpoints_file_name}_GRS.png"
                P_R_png = os.path.join(BSRperm_visuals_dir_path, gridpoints_file_name_png)
                plt.savefig(P_R_png, bbox_inches='tight', pad_inches=0.2, dpi=150)
                # plt.show()
                plt.close(fig) 

                if BSR_titles[BSR_i] == 'BSR-3.1':
                    fig = plt.figure(figsize=(12, 6))
                    ax = fig.add_subplot(111)
                    ax.imshow(img_MarsBackground_Greyscale, extent=[lon_0, lon_f, lat_0, lat_f])
                    ax.scatter(spoint_lon_deg[grid_i], spoint_lat_deg[grid_i], facecolors='none', edgecolors='black', marker='*', s=70, linewidths=0.3)
                    # ax.scatter(spoint_lon_deg[grid_i], spoint_lat_deg[grid_i], facecolors='none', edgecolors='black', marker='*', s=250, linewidths=0.8)
                    masked_grid = np.ma.masked_where(grid_P_R_modelFREND == 0, grid_P_R_modelFREND)
                    ax.set_xlabel('Longitude [$\degree$E]')
                    ax.set_ylabel('Latitude [$\degree$N]')
                    title = f'Power distribution over the mutual footprint for {BSR_titles[BSR_i]} (at {n_pixels} pixel/degree resolution)'
                    fig.text(s=title, x=0.47, y=0.998, fontsize=14, ha='center', va='center', fontweight='bold')
                    fig.text(s='(Background: HRSC_MOLA Blended DEM at 1500mp)', x=0.47, y=0.963, fontsize=12, ha='center', va='center')  

                    fig.text(x=0.96, y=0.995, s=f'UTC =', fontsize=12, ha='center', va='center', fontweight='bold')
                    fig.text(x=0.96, y=0.95, s=f'{utc_i[:11]}\n{utc_i[12:]}', fontsize=12, ha='center', va='center')
                    fig.text(x=0.96, y=0.07, s=f't_step =', fontsize=12, ha='center', va='center', fontweight='bold')
                    fig.text(x=0.96, y=0.035, s=f'{grid_i+1}', fontsize=12, ha='center', va='center')

                    img = ax.imshow(np.flipud(masked_grid),
                                    extent=[lon_0, lon_f, lat_0, lat_f],
                                    cmap='plasma_r',
                                    alpha=1.0,
                                    norm=LogNorm(vmin=P_R_modelFREND_mincont_min, vmax=P_R_modelFREND_maxcont_max))
                    cbar = plt.colorbar(img,
                                        label='Received power contribution [W]',
                                        ax=ax,
                                        fraction=0.022,
                                        pad=0.02)

                    plot_split_footprint(MEX_intpoint_lon, MEX_intpoint_lat, SC_color='red')
                    plot_split_footprint(TGO_intpoint_lon, TGO_intpoint_lat, SC_color='cyan')
                    plt.grid(True, color='black', alpha=0.1)
                    plt.tight_layout(rect=[-0.02, -0.02, 1.01, 0.98])
                    gridpoints_file_name_png = f"{gridpoints_file_name}_FREND.png"
                    P_R_png = os.path.join(BSRperm_visuals_dir_path, gridpoints_file_name_png)
                    plt.savefig(P_R_png, bbox_inches='tight', pad_inches=0.2, dpi=150)
                    # plt.show()
                    plt.close(fig) 

                # ******************************************************************************

            create_gif(BSRperm_visuals_dir_path, img_name_end="GRS", frame_duration_ms=30, gif_path=os.path.join(BSRperm_dir_path, f"animation_GRS.gif"))
            
            if BSR_titles[BSR_i] == 'BSR-3.1':
                create_gif(BSRperm_visuals_dir_path, img_name_end="FREND", frame_duration_ms=30, gif_path=os.path.join(BSRperm_dir_path, f"animation_FREND.gif"))

        # **************************************************************************************

        """ Plot BSR data """ 

        t_plot = et - et[0]  # seconds; time since the first measurement

        """ ----- Model vs. Data comparison ------------------------------------------------ """

        if BSR_plot:
            fig_compHL, axs = plt.subplots(1, 1, figsize=(14, 7))
            axs.plot(t_plot, P_R_data, linestyle='-', marker='o', label='Data (MEX-TGO BSR)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3])
            axs.plot(t_plot, P_R_modelGRS, linestyle='-', marker='o', label='Model (GRS)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][5])
            axs.plot(t_plot, P_R_freespace, linestyle='-', marker='o', label='Model (freespace)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][4])
            axs.plot(t_plot, P_R_combinedGRS, linestyle='-', marker='o', label='Model (GRS + freespace)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2])
            axs.axhline(y=P_noisefloor_W, color='gray', linestyle='--', label='Noisefloor')
            axs.set_xlabel('Time [s]')
            axs.set_ylabel('Received Power [W]')
            axs.set_yscale('log')
            axs.grid()
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
            fig_compHL.suptitle(f'Model results for {BSR_titles[BSR_i]} (at {n_pixels} pixel/degree resolution)', fontsize=20, fontweight='bold')
            fig_compHL.text(0.5, 0.92, 'Comparison of Model vs. Data', ha='center', va='center', fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            combined_pngHL = os.path.join(BSRperm_dir_path, str(BSR_titles[BSR_i]) + '_modeldatacomparison-highlevel_' + str(area_discretization) + 'deg.png')
            plt.savefig(combined_pngHL, bbox_inches='tight', pad_inches=0.05, dpi=150)
            # plt.show()
            plt.close(fig_compHL)

            fig_compPR, axs = plt.subplots(1, 1, figsize=(14, 7))
            axs.plot(t_plot, P_R_modelGRS_spoint, linestyle='-', marker='o', label='Model (GRS); return of specular point', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
            axs.plot(t_plot, P_R_modelGRS_maxcont, linestyle='-', marker='o', label='Model (GRS); Point of max. P_R', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
            axs.plot(t_plot, P_R_modelGRS, linestyle='-', marker='o', label='Model (GRS)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][5])
            axs.axhline(y=P_noisefloor_W, color='gray', linestyle='--', label='Noisefloor')
            axs.set_xlabel('Time [s]')
            axs.set_ylabel('Received Power [W]')
            axs.set_yscale('log')
            axs.grid()
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
            fig_compPR.suptitle(f"Model results for {BSR_titles[BSR_i]} (at {n_pixels} pixel/degree resolution)", fontsize=20, fontweight='bold')
            fig_compPR.text(0.5, 0.92, 'Comparison of the Power Contributions over Time', ha='center', va='center', fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            combined_pngPR = os.path.join(BSRperm_dir_path, str(BSR_titles[BSR_i]) + '_modeldatacomparison-receivedpower_' + str(area_discretization) + 'deg.png')
            plt.savefig(combined_pngPR, bbox_inches='tight', pad_inches=0.05, dpi=150)
            # plt.show()
            plt.close(fig_compPR)

            if BSR_titles[BSR_i] == 'BSR-3.1':
                fig_compPR, axs = plt.subplots(1, 1, figsize=(14, 7))
                axs.plot(t_plot, P_R_modelFREND_spoint, linestyle='-', marker='o', label='Model (FREND); return of specular point', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
                axs.plot(t_plot, P_R_modelFREND_maxcont, linestyle='-', marker='o', label='Model (FREND); Point of max. P_R', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
                axs.plot(t_plot, P_R_modelFREND, linestyle='-', marker='o', label='Model (FREND)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][5])
                axs.axhline(y=P_noisefloor_W, color='gray', linestyle='--', label='Noisefloor')
                axs.set_xlabel('Time [s]')
                axs.set_ylabel('Received Power [W]')
                axs.set_yscale('log')
                axs.grid()
                plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
                fig_compPR.suptitle(f"Model results for {BSR_titles[BSR_i]} (at {n_pixels} pixel/degree resolution)", fontsize=20, fontweight='bold')
                fig_compPR.text(0.5, 0.92, 'Comparison of the Power Contributions over Time', ha='center', va='center', fontsize=18)
                plt.tight_layout(rect=[0, 0, 1, 0.97])
                combined_pngPR = os.path.join(BSRperm_dir_path, str(BSR_titles[BSR_i]) + '_modeldatacomparison-receivedpowerFREND_' + str(area_discretization) + 'deg.png')
                plt.savefig(combined_pngPR, bbox_inches='tight', pad_inches=0.05, dpi=150)
                # plt.show()
                plt.close(fig_compPR)

                fig_compFvsG, axs = plt.subplots(1, 1, figsize=(14, 7))
                axs.plot(t_plot, P_R_modelGRS, linestyle='-', marker='o', label='Model (GRS)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][5])
                axs.plot(t_plot, P_R_modelFREND, linestyle='-', marker='o', label='Model (FREND)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3])
                axs.plot(t_plot, np.abs(P_R_modelFREND-P_R_modelGRS), linestyle='-', marker='o', label='Absolute difference', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][9])
                axs.set_xlabel('Time [s]')
                axs.set_ylabel('Received Power [W]')
                axs.set_yscale('log')
                axs.grid()
                plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
                fig_compFvsG.suptitle(f"Model results for {BSR_titles[BSR_i]} (at {n_pixels} pixel/degree resolution)", fontsize=20, fontweight='bold')
                fig_compFvsG.text(0.5, 0.92, 'Effect of the Surface Composition: GRS vs. FREND comparison', ha='center', va='center', fontsize=18)
                plt.tight_layout(rect=[0, 0, 1, 0.97])
                combined_pngFvsG = os.path.join(BSRperm_dir_path, str(BSR_titles[BSR_i]) + '_modeldatacomparison-FRENDvsGRS_' + str(area_discretization) + 'deg.png')
                plt.savefig(combined_pngFvsG, bbox_inches='tight', pad_inches=0.05, dpi=150)
                # plt.show()
                plt.close(fig_compFvsG)

            window_size = 61
            P_R_combinedGRS_movingavg = pd.Series(P_R_combinedGRS).rolling(window=window_size, min_periods=1, center=True).mean().values
            P_R_combinedGRS_savgol = savgol_filter(P_R_combinedGRS, window_length=window_size, polyorder=2)
            P_R_combinedGRS_lowess = lowess(P_R_combinedGRS, range(len(P_R_combinedGRS)), frac=0.09)[:, 1]

            if np.any(np.isnan(P_R_combinedGRS)):
                P_R_combinedGRS_lowess_list = list(P_R_combinedGRS_lowess)
                nan_indices = np.where(np.isnan(P_R_combinedGRS))[0]
                for nan_index in nan_indices:
                    P_R_combinedGRS_lowess_list.insert(nan_index, np.nan)
                P_R_combinedGRS_lowess = np.array(P_R_combinedGRS_lowess_list)

            fig_compSM, axs = plt.subplots(1, 1, figsize=(14, 7))
            axs.plot(t_plot, P_R_data, linestyle='-', marker='o', label='Data (MEX-TGO BSR)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3])
            axs.plot(t_plot, P_R_combinedGRS, linestyle='-', marker='o', label='Model (GRS + freespace)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2])
            axs.plot(t_plot, P_R_combinedGRS_movingavg, linestyle='-', marker='o', label='Moving Average of model (GRS + freespace)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][7])
            axs.plot(t_plot, P_R_combinedGRS_lowess, linestyle='-', marker='o', label='Lowess of model (GRS + freespace)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][6])
            # axs.plot(t_plot, P_R_combinedGRS_savgol, linestyle='-', marker='o', label='Savgol Filter of model (GRS + freespace)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][8])
            axs.axhline(y=P_noisefloor_W, color='gray', linestyle='--', label='Noisefloor')
            axs.set_xlabel('Time [s]')
            axs.set_ylabel('Received Power [W]')
            axs.set_yscale('log')
            axs.grid()
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)

            fig_compSM.suptitle(f'Model results for {BSR_titles[BSR_i]} (at {n_pixels} pixel/degree resolution)', fontsize=20, fontweight='bold')
            fig_compSM.text(0.5, 0.92, 'Comparison of Model vs. Data (with smoothing)', ha='center', va='center', fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            combined_pngSM = os.path.join(BSRperm_dir_path, str(BSR_titles[BSR_i]) + '_modeldatacomparison-smoothed_wonotes_' + str(area_discretization) + 'deg.png')
            plt.savefig(combined_pngSM, bbox_inches='tight', pad_inches=0.05, dpi=150)
            # plt.show()
            plt.close(fig_compSM)

            if BSR_titles[BSR_i] in ['BSR-2.1', 'BSR-2.2'] and n_pixels == 4:
                fig_compSMwN, axs = plt.subplots(1, 1, figsize=(14, 7))
                axs.plot(t_plot, P_R_data, linestyle='-', marker='o', label='Data (MEX-TGO BSR)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3])
                axs.plot(t_plot, P_R_combinedGRS, linestyle='-', marker='o', label='Model (GRS + freespace)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2])
                axs.plot(t_plot, P_R_combinedGRS_movingavg, linestyle='-', marker='o', label='Moving Average of model (GRS + freespace)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][7])
                axs.plot(t_plot, P_R_combinedGRS_lowess, linestyle='-', marker='o', label='Lowess of model (GRS + freespace)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][6])
                # axs.plot(t_plot, P_R_combinedGRS_savgol, linestyle='-', marker='o', label='Savgol Filter of model (GRS + freespace)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][8])
                axs.axhline(y=P_noisefloor_W, color='gray', linestyle='--', label='Noisefloor')

                if BSR_titles[BSR_i] == 'BSR-2.1':
                    axs.axvline(x=25, color='black', linestyle='--')
                    plt.text(25+2, 3e-12, 'A')
                    axs.axvline(x=315, color='black', linestyle='--')
                    plt.text(315+2, 3e-12, 'B1')
                    axs.axvline(x=365, color='black', linestyle='--')
                    plt.text(365+2, 3e-12, 'B2')
                    axs.axvline(x=507, color='black', linestyle='--')
                    plt.text(507+2, 3e-12, 'C')
                    axs.axvline(x=580, color='black', linestyle='--')
                    plt.text(580+2, 3e-12, 'D')

                axs.set_xlabel('Time [s]')
                axs.set_ylabel('Received Power [W]')
                axs.set_yscale('log')
                axs.grid()
                plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)

                fig_compSMwN.suptitle(f'Model results for {BSR_titles[BSR_i]} (at {n_pixels} pixel/degree resolution)', fontsize=20, fontweight='bold')
                fig_compSMwN.text(0.5, 0.92, 'Comparison of Model vs. Data (with smoothing)', ha='center', va='center', fontsize=18)
                plt.tight_layout(rect=[0, 0, 1, 0.97])
                combined_pngSM = os.path.join(BSRperm_dir_path, str(BSR_titles[BSR_i]) + '_modeldatacomparison-smoothed_' + str(area_discretization) + 'deg.png')
                plt.savefig(combined_pngSM, bbox_inches='tight', pad_inches=0.05, dpi=150)
                # plt.show()
                plt.close(fig_compSMwN)

            # fig_comp, axs = plt.subplots(1, 1, figsize=(11, 6))
            # # axs.plot(t_plot, P_R_modelGRS_maxcont, linestyle='-', marker='o', label='Model (GRS); Point of max. P_R', markersize=1.5)
            # # axs.plot(t_plot, P_R_modelGRS_spoint, linestyle='-', marker='o', label='Model (GRS); return of specular point', markersize=1.5)
            # # axs.plot(t_plot, P_R_modelGRS, linestyle='-', marker='o', label='Model (GRS)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][5])
            # axs.plot(t_plot, P_R_data, linestyle='-', marker='o', label='Data (MEX-TGO BSR)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3])
            # # axs.plot(t_plot, P_R_modelFREND_maxcont, linestyle='-', marker='o', label='Model (FREND); Point of max. P_R', markersize=1.5)
            # # axs.plot(t_plot, P_R_modelFREND_spoint, linestyle='-', marker='o', label='Model (FREND); Contribution of specular point', markersize=1.5)
            # # axs.plot(t_plot, P_R_modelFREND, linestyle='-', marker='o', label='Model (FREND)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3])
            # # axs.plot(t_plot, P_R_freespace, linestyle='-', marker='o', label='Model (freespace)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][4])
            # axs.plot(t_plot, P_R_combinedGRS, linestyle='-', marker='o', label='Model (GRS + freespace)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2])
            # axs.axhline(y=P_noisefloor_W, color='gray', linestyle='--', label='Noisefloor')
            # # axs.axvline(x=85, color='black', linestyle='--')
            # # plt.text(85+2, 1.5e-13, 'Signal Mirror')
            # # plt.text(85+2, 1.2e-13, 't = 85 s')
            # # axs.axvline(x=76, color='black', linestyle='--')
            # # plt.text(76+2, 1.5e-13, 'Signal Mirror')
            # # plt.text(76+2, 1.2e-13, 't = 76 s')
            # # axs.axvline(x=433, color='black', linestyle='--')
            # # plt.text(433+2, 1.6e-12, 'Signal Mirror')
            # # plt.text(433+2, 1.1e-12, 't = 433 s')
            # # axs.axvline(x=517, color='black', linestyle='--')
            # # plt.text(517+2, 1.6e-12, 'Signal Mirror')
            # # plt.text(517+2, 1.1e-12, 't = 517 s')
            # axs.set_xlabel('Time [s]')
            # axs.set_ylabel('Received Power [W]')
            # axs.set_yscale('log')
            # axs.grid()
            # plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
            # fig_comp.suptitle(f'Model results for {BSR_titles[BSR_i]} (at {n_pixels} pixel/degree resolution)', fontsize=20, fontweight='bold')
            # fig_comp.text(0.5, 0.91, 'Comparison of Model vs. Data', ha='center', va='center', fontsize=18)
            # plt.tight_layout(rect=[0.01, 0, 1, 0.97])
            # combined_png = os.path.join(BSRperm_dir_path, str(BSR_titles[BSR_i]) + '_modeldatacomparison_' + str(area_discretization) + 'deg.png')
            # # plt.savefig(combined_png, bbox_inches='tight', pad_inches=0.05, dpi=150)
            # plt.show()
            # # plt.close(fig_comp)

            """ ----- Model parameters ---------------------------------------------------- """

            fig_param = plt.figure(figsize=(14, 10))
            gs_param = gridspec.GridSpec(3, 3)
            ax1_param = fig_param.add_subplot(gs_param[:2, 0])
            ax2_param = fig_param.add_subplot(gs_param[0, 1])
            ax3_param = fig_param.add_subplot(gs_param[0, 2])
            ax4_param = fig_param.add_subplot(gs_param[1, 1])
            ax5_param = fig_param.add_subplot(gs_param[1, 2])
            ax6_param = fig_param.add_subplot(gs_param[2, 0])
            ax7_param = fig_param.add_subplot(gs_param[2, 1])
            ax8_param = fig_param.add_subplot(gs_param[2, 2])

            tab20_colors_all = plt.get_cmap('tab20').colors
            tab20_colors = tuple(tab20_colors_all[i] for i in range(0, 20, 2)) + tuple(tab20_colors_all[i] for i in range(1, 20, 2))

            ax1_param.plot(t_plot, R_MEXspointTGO, linestyle='-', marker='o', label="Complete", markersize=1.5, color=tab20_colors[0])
            ax1_param.plot(t_plot, R_T_spoint, linestyle='-', marker='o', label="R_T", markersize=1.5, color=tab20_colors[1])
            ax1_param.plot(t_plot, R_R_spoint, linestyle='-', marker='o', label="R_R", markersize=1.5, color=tab20_colors[2])
            ax1_param.plot(t_plot, r_MEXTGO, linestyle='-', marker='o', label="r_MEXTGO", markersize=1.5, color=tab20_colors[3])
            ax1_param.set_xlabel('Time [s]')
            ax1_param.set_ylabel('Signal path\n(via spoint) [km]')
            ax1_param.legend(ncols=2, loc='lower center', bbox_to_anchor=(0.5, 1.01), borderaxespad=0.)
            ax1_param.grid()

            ax2_param.plot(t_plot, rho_modelGRS, linestyle='-', marker='o', label='Average (GRS)', markersize=1.5, color=tab20_colors[4])
            ax2_param.plot(t_plot, rho_modelGRS_spoint, linestyle='-', marker='o', label='At spoint (GRS)', markersize=1.5, color=tab20_colors[5])
            ax2_param.plot(t_plot, rho_modelGRS_maxcont, linestyle='-', marker='o', label='At point of max. P_R (GRS)', markersize=1.5, color=tab20_colors[6])
            ax2_param.set_xlabel('Time [s]')
            ax2_param.set_ylabel('Reflectivity [-]')
            ax2_param.legend(ncols=2, loc='lower center', bbox_to_anchor=(0.5, 1.01), borderaxespad=0.)
            ax2_param.grid()

            ax3_param.plot(t_plot, dc_modelGRS, linestyle='-', marker='o', label='Average (GRS)', markersize=1.5, color=tab20_colors[4])
            ax3_param.plot(t_plot, dc_modelGRS_spoint, linestyle='-', marker='o', label='At spoint (GRS)', markersize=1.5, color=tab20_colors[5])
            ax3_param.plot(t_plot, dc_modelGRS_maxcont, linestyle='-', marker='o', label='At point of max. P_R (GRS)', markersize=1.5, color=tab20_colors[6])
            ax3_param.set_xlabel('Time [s]')
            ax3_param.set_ylabel('Permittivity [-]')
            ax3_param.legend(ncols=2, loc='lower center', bbox_to_anchor=(0.5, 1.01), borderaxespad=0.)
            ax3_param.grid()

            # ax4_param.plot(t_plot, emission_MEX, label= 'MEX', linestyle='-', marker='o', markersize=1.5)
            # ax4_param.plot(t_plot, emission_TGO, label= 'TGO', linestyle='-', marker='o', markersize=1.5)
            ax4_param.plot(t_plot, emis_avg, label='Average', linestyle='-', marker='o', markersize=1.5, color=tab20_colors[10])
            ax4_param.set_xlabel('Time [s]')
            ax4_param.set_ylabel('Equal-emission angle\n(at spoint) [deg]')
            # ax4_param.legend(ncols=2, loc='upper center', bbox_to_anchor=(0.5, -0.2))
            ax4_param.grid()

            ax5_param.plot(t_plot, S_mutualfootprint, linestyle='-', marker='o', markersize=1.5, color=tab20_colors[8])
            ax5_param.set_xlabel('Time [s]')
            ax5_param.set_ylabel('Mutual surface\nfootprint [km^2]')
            ax5_param.grid()

            ax6_param.plot(t_plot, gamma_spoint, linestyle='-', marker='o', label='At spoint', markersize=1.5, color=tab20_colors[9])
            ax6_param.plot(t_plot, gamma_GRS_maxcont, linestyle='-', marker='o', label='At point of max. P_R (GRS)', markersize=1.5, color=tab20_colors[7])
            ax6_param.set_xlabel('Time [s]')
            ax6_param.set_ylabel('Required tilt angle [deg]')
            ax6_param.legend(ncols=2, loc='lower center', bbox_to_anchor=(0.5, 1.01), borderaxespad=0.)
            ax6_param.grid()

            ax7_param.plot(t_plot, C_spoint, linestyle='-', marker='o', label='At spoint', markersize=1.5, color=tab20_colors[9])
            ax7_param.plot(t_plot, C_GRS_maxcont, linestyle='-', marker='o', label='At point of max. P_R (GRS)', markersize=1.5, color=tab20_colors[7])
            ax7_param.set_xlabel('Time [s]')
            ax7_param.set_ylabel('Width parameter (C) [-]')
            ax7_param.set_yscale('log')
            ax7_param.legend(ncols=2, loc='lower center', bbox_to_anchor=(0.5, 1.01), borderaxespad=0.)
            ax7_param.grid()

            ax8_param.plot(t_plot, topography_spoint, linestyle='-', marker='o', label='At spoint', markersize=1.5, color=tab20_colors[9])
            ax8_param.plot(t_plot, topography_GRS_maxcont, linestyle='-', marker='o', label='At point of max. P_R (GRS)', markersize=1.5, color=tab20_colors[7])
            ax8_param.set_xlabel('Time [s]')
            ax8_param.set_ylabel('Topography scaling factor [-]')
            ax8_param.set_yscale('log')
            ax8_param.legend(ncols=2, loc='lower center', bbox_to_anchor=(0.5, 1.01), borderaxespad=0.)
            ax8_param.grid()

            fig_param.suptitle(f"Model results for {BSR_titles[BSR_i]} (at {n_pixels} pixel/degree resolution)", fontsize=20, fontweight='bold')
            fig_param.text(0.5, 0.935, 'Effects of the BSR model parameters', ha='center', va='center', fontsize=18)
            plt.tight_layout(rect=[0, 0.03, 1, 0.965])
            fig_param.subplots_adjust(wspace=0.25, hspace=0.35)
            param_png = os.path.join(BSRperm_dir_path, str(BSR_titles[BSR_i]) + '_parameters_' + str(area_discretization) + 'deg.png')
            plt.savefig(param_png, bbox_inches='tight', pad_inches=0.05, dpi=150)
            # plt.show()
            plt.close(fig_param)

            if BSR_titles[BSR_i] == 'BSR-3.1':
                fig_param = plt.figure(figsize=(14, 10))
                gs_param = gridspec.GridSpec(3, 3)
                ax1_param = fig_param.add_subplot(gs_param[:2, 0])
                ax2_param = fig_param.add_subplot(gs_param[0, 1])
                ax3_param = fig_param.add_subplot(gs_param[0, 2])
                ax4_param = fig_param.add_subplot(gs_param[1, 1])
                ax5_param = fig_param.add_subplot(gs_param[1, 2])
                ax6_param = fig_param.add_subplot(gs_param[2, 0])
                ax7_param = fig_param.add_subplot(gs_param[2, 1])
                ax8_param = fig_param.add_subplot(gs_param[2, 2])

                ax1_param.plot(t_plot, R_MEXspointTGO, linestyle='-', marker='o', label="Complete", markersize=1.5, color=tab20_colors[0])
                ax1_param.plot(t_plot, R_T_spoint, linestyle='-', marker='o', label="R_T", markersize=1.5, color=tab20_colors[1])
                ax1_param.plot(t_plot, R_R_spoint, linestyle='-', marker='o', label="R_R", markersize=1.5, color=tab20_colors[2])
                ax1_param.plot(t_plot, r_MEXTGO, linestyle='-', marker='o', label="r_MEXTGO", markersize=1.5, color=tab20_colors[3])
                ax1_param.set_xlabel('Time [s]')
                ax1_param.set_ylabel('Signal path\n(via spoint) [km]')
                ax1_param.legend(ncols=2, loc='lower center', bbox_to_anchor=(0.5, 1.01), borderaxespad=0.)
                ax1_param.grid()

                ax2_param.plot(t_plot, rho_modelFREND, linestyle='-', marker='o', label='Average (FREND)', markersize=1.5, color=tab20_colors[4])
                ax2_param.plot(t_plot, rho_modelFREND_spoint, linestyle='-', marker='o', label='At spoint (FREND)', markersize=1.5, color=tab20_colors[5])
                ax2_param.plot(t_plot, rho_modelFREND_maxcont, linestyle='-', marker='o', label='At point of max. P_R (FREND)', markersize=1.5, color=tab20_colors[6])
                ax2_param.set_xlabel('Time [s]')
                ax2_param.set_ylabel('Reflectivity [-]')
                ax2_param.legend(ncols=2, loc='lower center', bbox_to_anchor=(0.5, 1.01), borderaxespad=0.)
                ax2_param.grid()

                ax3_param.plot(t_plot, dc_modelFREND, linestyle='-', marker='o', label='Average (FREND)', markersize=1.5, color=tab20_colors[4])
                ax3_param.plot(t_plot, dc_modelFREND_spoint, linestyle='-', marker='o', label='At spoint (FREND)', markersize=1.5, color=tab20_colors[5])
                ax3_param.plot(t_plot, dc_modelFREND_maxcont, linestyle='-', marker='o', label='At point of max. P_R (FREND)', markersize=1.5, color=tab20_colors[6])
                ax3_param.set_xlabel('Time [s]')
                ax3_param.set_ylabel('Permittivity [-]')
                ax3_param.legend(ncols=2, loc='lower center', bbox_to_anchor=(0.5, 1.01), borderaxespad=0.)
                ax3_param.grid()

                # ax4_param.plot(t_plot, emission_MEX, label= 'MEX', linestyle='-', marker='o', markersize=1.5)
                # ax4_param.plot(t_plot, emission_TGO, label= 'TGO', linestyle='-', marker='o', markersize=1.5)
                ax4_param.plot(t_plot, emis_avg, label='Average', linestyle='-', marker='o', markersize=1.5, color=tab20_colors[10])
                ax4_param.set_xlabel('Time [s]')
                ax4_param.set_ylabel('Equal-emission angle\n(at spoint) [deg]')
                # ax4_param.legend(ncols=2, loc='upper center', bbox_to_anchor=(0.5, -0.2))
                ax4_param.grid()

                ax5_param.plot(t_plot, S_mutualfootprint, linestyle='-', marker='o', markersize=1.5, color=tab20_colors[8])
                ax5_param.set_xlabel('Time [s]')
                ax5_param.set_ylabel('Mutual surface\nfootprint [km^2]')
                ax5_param.grid()

                ax6_param.plot(t_plot, gamma_spoint, linestyle='-', marker='o', label='At spoint', markersize=1.5, color=tab20_colors[9])
                ax6_param.plot(t_plot, gamma_FREND_maxcont, linestyle='-', marker='o', label='At point of max. P_R (FREND)', markersize=1.5, color=tab20_colors[7])
                ax6_param.set_xlabel('Time [s]')
                ax6_param.set_ylabel('Required tilt angle [deg]')
                ax6_param.legend(ncols=2, loc='lower center', bbox_to_anchor=(0.5, 1.01), borderaxespad=0.)
                ax6_param.grid()

                ax7_param.plot(t_plot, C_spoint, linestyle='-', marker='o', label='At spoint', markersize=1.5, color=tab20_colors[9])
                ax7_param.plot(t_plot, C_FREND_maxcont, linestyle='-', marker='o', label='At point of max. P_R (FREND)', markersize=1.5, color=tab20_colors[7])
                ax7_param.set_xlabel('Time [s]')
                ax7_param.set_ylabel('Width parameter (C) [-]')
                ax7_param.set_yscale('log')
                ax7_param.legend(ncols=2, loc='lower center', bbox_to_anchor=(0.5, 1.01), borderaxespad=0.)
                ax7_param.grid()

                ax8_param.plot(t_plot, topography_spoint, linestyle='-', marker='o', label='At spoint', markersize=1.5, color=tab20_colors[9])
                ax8_param.plot(t_plot, topography_FREND_maxcont, linestyle='-', marker='o', label='At point of max. P_R (FREND)', markersize=1.5, color=tab20_colors[7])
                ax8_param.set_xlabel('Time [s]')
                ax8_param.set_ylabel('Topography scaling factor [-]')
                ax8_param.set_yscale('log')
                ax8_param.legend(ncols=2, loc='lower center', bbox_to_anchor=(0.5, 1.01), borderaxespad=0.)
                ax8_param.grid()

                fig_param.suptitle(f"Model results for {BSR_titles[BSR_i]} (at {n_pixels} pixel/degree resolution)", fontsize=20, fontweight='bold')
                fig_param.text(0.5, 0.935, 'Effects of the BSR model parameters', ha='center', va='center', fontsize=18)
                plt.tight_layout(rect=[0, 0.03, 1, 0.965])
                fig_param.subplots_adjust(wspace=0.25, hspace=0.35)
                param_png = os.path.join(BSRperm_dir_path, str(BSR_titles[BSR_i]) + '_parametersFREND_' + str(area_discretization) + 'deg.png')
                plt.savefig(param_png, bbox_inches='tight', pad_inches=0.05, dpi=150)
                # plt.show()
                plt.close(fig_param)

            # **************************************************************************************

            """ Plots of factors for free-space/direct signal contribution """

            fig_freespace, axs = plt.subplots(2, 2, figsize=(14, 9))

            axs[0,0].plot(t_plot, P_R_freespace, linestyle='-', marker='o', label="Complete", markersize=1.5)
            axs[0,0].set_xlabel('Time [s]')
            axs[0,0].set_ylabel('Received Power [W]')
            axs[0,0].grid()

            axs[0,1].plot(t_plot, r_MEXTGO, linestyle='-', marker='o', label="r_MEXTGO", markersize=1.5)
            axs[0,1].set_xlabel('Time [s]')
            axs[0,1].set_ylabel('MEX-TGO distance [km]')
            axs[0,1].grid()

            axs[1,0].plot(t_plot, G_T_freespace, linestyle='-', marker='o', label='G_T (MEX)', markersize=1.5)
            axs[1,0].plot(t_plot, G_R_freespace, linestyle='-', marker='o', label='G_R (TGO)', markersize=1.5)
            axs[1,0].set_xlabel('Time [s]')
            axs[1,0].set_ylabel('Direct free-space gain [-]')
            axs[1,0].legend(ncols=2, loc='lower center', bbox_to_anchor=(0.5, 1.01), borderaxespad=0.)
            axs[1,0].grid()

            axs[1,1].plot(t_plot, angle_MEX2TGO, linestyle='-', marker='o', label='MEX to TGO', markersize=1.5)
            axs[1,1].plot(t_plot, angle_TGO2MEX, linestyle='-', marker='o', label='TGO to MEX', markersize=1.5)
            axs[1,1].set_xlabel('Time [s]')
            axs[1,1].set_ylabel('Off-boresight angle [deg]')
            axs[1,1].legend(ncols=2, loc='lower center', bbox_to_anchor=(0.5, 1.01), borderaxespad=0.)
            axs[1,1].grid()

            fig_freespace.suptitle(f"Model results for {BSR_titles[BSR_i]}", fontsize=20, fontweight='bold')
            fig_freespace.text(0.5, 0.93, 'Free-space direct signal contribution', ha='center', va='center', fontsize=18)
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            fig_freespace.subplots_adjust(wspace=0.25, hspace=0.3)
            freespace_png = os.path.join(BSRperm_dir_path, str(BSR_titles[BSR_i]) + '_freespace_' + str(area_discretization) + 'deg.png')
            plt.savefig(freespace_png, bbox_inches='tight', pad_inches=0.05, dpi=150)
            # plt.show()
            plt.close(fig_freespace)

            # **************************************************************************************

            """ Plots of topography factors """

            fig_topomax, ax_topomax = plt.subplots(2, 3, figsize=(15, 8))

            ax_topomax[0,0].plot(t_plot, gamma_GRS_maxcont, label="Point of max. P_R", color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
            ax_topomax[0,0].set_ylabel('Required tilt angle [deg]')
            ax_topomax[0,0].set_xlabel("Time [s]")
            ax_topomax[0,0].grid()

            ax_topomax[0,1].plot(t_plot, C_GRS_maxcont, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
            ax_topomax[0,1].set_ylabel('Width parameter (C) [-]')
            ax_topomax[0,1].set_xlabel("Time [s]")
            ax_topomax[0,1].set_yscale('log')
            ax_topomax[0,1].grid()

            ax_topomax[0,2].plot(t_plot, topography_GRS_maxcont, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
            ax_topomax[0,2].set_ylabel('Topography scaling factor [-]')
            ax_topomax[0,2].set_xlabel("Time [s]")
            ax_topomax[0,2].set_yscale('log')
            ax_topomax[0,2].grid()

            ax_topomax[1,0].plot(t_plot, gamma_spoint, label="Specular point", color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
            ax_topomax[1,0].set_ylabel('Required tilt angle [deg]')
            ax_topomax[1,0].set_xlabel("Time [s]")
            ax_topomax[1,0].grid()

            ax_topomax[1,1].plot(t_plot, C_spoint, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
            ax_topomax[1,1].set_ylabel('Width parameter (C) [-]')
            ax_topomax[1,1].set_xlabel("Time [s]")
            ax_topomax[1,1].set_yscale('log')
            ax_topomax[1,1].grid()

            ax_topomax[1,2].plot(t_plot, topography_spoint, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
            ax_topomax[1,2].set_ylabel('Topography scaling factor [-]')
            ax_topomax[1,2].set_xlabel("Time [s]")
            ax_topomax[1,2].set_yscale('log')
            ax_topomax[1,2].grid()

            fig_topomax.suptitle(f"Model results for {BSR_titles[BSR_i]} (at {n_pixels} pixel/degree resolution)", fontsize=20, fontweight='bold')
            fig_topomax.text(0.5, 0.92, 'Periodicity and Topography Effects', ha='center', va='center', fontsize=18)
            fig_topomax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            topomax_png = os.path.join(BSRperm_dir_path, str(BSR_titles[BSR_i]) + '_TopographyFactors_' + str(area_discretization) + 'deg.png')
            plt.savefig(topomax_png, bbox_inches='tight', pad_inches=0.05, dpi=150)
            # plt.show()
            plt.close(fig_topomax)

            # **************************************************************************************

            """ Tracks of max. contribution vs. spoint """

            fig_maxconttrack, ax_maxconttrack = plt.subplots(figsize=(12, 10))
            fig_maxconttrack.set_size_inches(19.2, 10.8)  # Nominal full screen: 1920x1080
            plt.imshow(img_MarsBackground, extent=[-180, 180, -90, 90], aspect='equal')

            plt.scatter(MEXbs_lon, MEXbs_lat, label='MEX boresight', s=1, color='red')
            plt.scatter(TGObs_lon, TGObs_lat, label='TGO boresight', s=1, color='cyan')
            plt.scatter(spoint_lon_deg, spoint_lat_deg, label='Spoint', s=1, color='fuchsia')
            plt.scatter(midpoint_lon_deg, midpoint_lat_deg, label='Midpoint', s=1, color='pink')
            plt.scatter(lon_GRS_maxcont, lat_GRS_maxcont, label='Point of max. P_R (GRS)', s=1, color='limegreen')

            title = f'Ground track comparisons for {BSR_titles[BSR_i]} (at {n_pixels} pixel/degree resolution)'
            fig_maxconttrack.text(s=title, x=0.42, y=0.72+0.09, fontsize=18, ha='center', va='center', fontweight='bold')
            fig_maxconttrack.text(s='(Background: HRSC_MOLA Blended DEM at 1500mp)', x=0.42, y=0.695+0.09, fontsize=12, ha='center', va='center')

            lon_major_ticks = np.arange(-180, 181, 30)
            lon_minor_ticks = np.arange(-180, 181, 5)
            lat_major_ticks = np.arange(-90, 91, 30)
            lat_minor_ticks = np.arange(-90, 91, 5)

            ax_maxconttrack.set_xticks(lon_major_ticks)
            ax_maxconttrack.set_xticks(lon_minor_ticks, minor=True)
            ax_maxconttrack.set_yticks(lat_major_ticks)
            ax_maxconttrack.set_yticks(lat_minor_ticks, minor=True)

            ax_maxconttrack.tick_params(axis='both', which='major', labelsize=10)

            ax_maxconttrack.grid(which='minor', alpha=0.2, color='black', linestyle='-', linewidth=0.3)
            ax_maxconttrack.grid(which='major', alpha=0.7, color='black', linestyle='-', linewidth=0.3)

            ax_maxconttrack.set_axisbelow(True)

            box_maxconttrack = ax_maxconttrack.get_position()
            ax_maxconttrack.set_position([box_maxconttrack.x0, box_maxconttrack.y0, box_maxconttrack.width * 0.8, box_maxconttrack.height])
            ax_maxconttrack.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax_maxconttrack.set_xlabel('Longitude [$\degree$E]')
            ax_maxconttrack.set_ylabel('Latitude [$\degree$N]')

            maxconttrack_png = os.path.join(BSRperm_dir_path, str(BSR_titles[BSR_i]) + '_MaxContTrack_' + str(area_discretization) + 'deg.png')
            plt.savefig(maxconttrack_png, bbox_inches='tight', pad_inches=0.05, dpi=150)
            # plt.show()
            plt.close(fig_maxconttrack)

            if BSR_titles[BSR_i] == 'BSR-3.1':
                fig_maxconttrack, ax_maxconttrack = plt.subplots(figsize=(12, 10))
                fig_maxconttrack.set_size_inches(19.2, 10.8)  # Nominal full screen: 1920x1080
                plt.imshow(img_MarsBackground, extent=[-180, 180, -90, 90], aspect='equal')

                plt.scatter(MEXbs_lon, MEXbs_lat, label='MEX boresight', s=1, color='red')
                plt.scatter(TGObs_lon, TGObs_lat, label='TGO boresight', s=1, color='cyan')
                plt.scatter(spoint_lon_deg, spoint_lat_deg, label='Spoint', s=1, color='fuchsia')
                plt.scatter(midpoint_lon_deg, midpoint_lat_deg, label='Midpoint', s=1, color='pink')
                plt.scatter(lon_FREND_maxcont, lat_FREND_maxcont, label='Point of max. P_R (FREND)', s=1, color='limegreen')

                title = f'Ground track comparisons for {BSR_titles[BSR_i]} (at {n_pixels} pixel/degree resolution)'
                fig_maxconttrack.text(s=title, x=0.42, y=0.72+0.09, fontsize=18, ha='center', va='center', fontweight='bold')
                fig_maxconttrack.text(s='(Background: HRSC_MOLA Blended DEM at 1500mp)', x=0.42, y=0.695+0.09, fontsize=12, ha='center', va='center')

                lon_major_ticks = np.arange(-180, 181, 30)
                lon_minor_ticks = np.arange(-180, 181, 5)
                lat_major_ticks = np.arange(-90, 91, 30)
                lat_minor_ticks = np.arange(-90, 91, 5)

                ax_maxconttrack.set_xticks(lon_major_ticks)
                ax_maxconttrack.set_xticks(lon_minor_ticks, minor=True)
                ax_maxconttrack.set_yticks(lat_major_ticks)
                ax_maxconttrack.set_yticks(lat_minor_ticks, minor=True)

                ax_maxconttrack.tick_params(axis='both', which='major', labelsize=10)

                ax_maxconttrack.grid(which='minor', alpha=0.2, color='black', linestyle='-', linewidth=0.3)
                ax_maxconttrack.grid(which='major', alpha=0.7, color='black', linestyle='-', linewidth=0.3)

                ax_maxconttrack.set_axisbelow(True)

                box_maxconttrack = ax_maxconttrack.get_position()
                ax_maxconttrack.set_position([box_maxconttrack.x0, box_maxconttrack.y0, box_maxconttrack.width * 0.8, box_maxconttrack.height])
                ax_maxconttrack.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                ax_maxconttrack.set_xlabel('Longitude [$\degree$E]')
                ax_maxconttrack.set_ylabel('Latitude [$\degree$N]')

                maxconttrack_png = os.path.join(BSRperm_dir_path, str(BSR_titles[BSR_i]) + '_MaxContTrackFREND_' + str(area_discretization) + 'deg.png')
                plt.savefig(maxconttrack_png, bbox_inches='tight', pad_inches=0.05, dpi=150)
                # plt.show()
                plt.close(fig_maxconttrack)

    # **********************************************************************************************

    """ Comparison of polar host materials """

    if BSR_plot_MatComp:
        fig_comp, axs = plt.subplots(1, 1, figsize=(14, 7))
        axs.plot(t_plot, P_R_modelGRS_maxcont_dict['rego'], linestyle='-', marker='o', label='Model (Regolith); Point of max. P_R', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
        axs.plot(t_plot, P_R_modelGRS_maxcont_dict['co2'], linestyle='-', marker='o', label='Model (CO$_2$); Point of max. P_R', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
        axs.plot(t_plot, P_R_modelGRS_spoint_dict['rego'], linestyle='-', marker='o', label='Model (Regolith); return of specular point', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2])
        axs.plot(t_plot, P_R_modelGRS_spoint_dict['co2'], linestyle='-', marker='o', label='Model (CO$_2$); return of specular point', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3])
        axs.plot(t_plot, P_R_modelGRS_dict['rego'], linestyle='-', marker='o', label='Model (Regolith)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][4])
        axs.plot(t_plot, P_R_modelGRS_dict['co2'], linestyle='-', marker='o', label='Model (CO$_2$)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][5])
        axs.set_xlabel('Time [s]')
        axs.set_ylabel('Received Power [W]')
        axs.set_yscale('log')
        axs.grid()
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
        fig_comp.suptitle(f"Model results for {BSR_titles[BSR_i]} (at {n_pixels} pixel/degree resolution)", fontsize=20, fontweight='bold')
        fig_comp.text(0.5, 0.92, 'Effect of the Surface Composition: Regolith vs. CO$_2$ host comparison', ha='center', va='center', fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        combined_png = os.path.join(BSR_dir_path, str(BSR_titles[BSR_i]) + '_modeldatacomparison-regoVSco2_' + str(area_discretization) + 'deg-all.png')
        plt.savefig(combined_png, bbox_inches='tight', pad_inches=0.05, dpi=150)
        # plt.show()
        plt.close(fig_comp)

        fig_comp, axs = plt.subplots(1, 1, figsize=(14, 7))
        axs.plot(t_plot, P_R_modelGRS_dict['rego'], linestyle='-', marker='o', label='Model (Regolith)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][4])
        axs.plot(t_plot, P_R_modelGRS_dict['co2'], linestyle='-', marker='o', label='Model (CO$_2$)', markersize=1.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][5])
        axs.set_xlabel('Time [s]')
        axs.set_ylabel('Received Power [W]')
        axs.set_yscale('log')
        axs.grid()
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
        fig_comp.suptitle(f"Model results for {BSR_titles[BSR_i]} (at {n_pixels} pixel/degree resolution)", fontsize=20, fontweight='bold')
        fig_comp.text(0.5, 0.92, 'Effect of the Surface Composition: Regolith vs. CO$_2$ host comparison', ha='center', va='center', fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        combined_png = os.path.join(BSR_dir_path, str(BSR_titles[BSR_i]) + '_modeldatacomparison-regoVSco2_' + str(area_discretization) + 'deg.png')
        plt.savefig(combined_png, bbox_inches='tight', pad_inches=0.05, dpi=150)
        # plt.show()
        plt.close(fig_comp)

    # **********************************************************************************************

    """ Plots of (converted) AGC data """

    if AGC_plot:
        print(f"Plotting AGC data...")

        IQ_file_datapath = os.path.join(IQ_dir_path, IQ_titles[BSR_i])
        AGC_data_complete = read_IQ_file(IQ_file_datapath)
        t_plot_AGC_data_complete = np.linspace(t_plot[0], t_plot[-1], len(AGC_data_complete))

        fig_agc, ax_agc = plt.subplots(figsize=(10, 6))
        ax_agc.plot(t_plot_AGC_data_complete[::3], AGC_data_complete[::3].astype(np.float32), label="Complete AGC data")
        ax_agc.plot(t_plot, AGC_data, label="Downsampled AGC data")
        ax_agc.axhline(y=AGC_baseline, color='gray', linestyle='--', label='Baseline (noise floor)')
        title = f"AGC data for {BSR_titles[BSR_i]}"
        fig_agc.text(s=title, x=0.42, y=0.98, fontsize=16, ha='center', va='center', fontweight='bold')
        fig_agc.text(s=f'Data retrieved from {IQ_titles[BSR_i]}', x=0.42, y=0.94, fontsize=12, ha='center', va='center')  
        ax_agc.set_ylabel('AGC Value (0 to 255)')
        ax_agc.set_xlabel("Time [s]")
        ax_agc.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax_agc.grid()
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        agc_png = os.path.join(BSR_dir_path, str(BSR_titles[BSR_i]) + '_AGC_' + str(area_discretization) + 'deg.png')
        plt.savefig(agc_png, bbox_inches='tight', pad_inches=0.05, dpi=150)
        # plt.show()
        plt.close(fig_agc)

        P_R_data_dB = 10 * np.log10(P_R_data * 1000)
        dB_step = 1
        AGC_data_complete_atbaseline = np.where(AGC_data_complete < AGC_baseline, AGC_baseline, AGC_data_complete) # Set all values below baseline to baseline
        P_R_dB_data_complete = P_noisefloor_dB + (AGC_data_complete_atbaseline - AGC_baseline) * dB_step # Convert AGC values to dB
        P_R_W_data_complete = 1.0 * (10.0**(P_R_dB_data_complete / 10))/1000 # Convert dB to Watts

        fig_agc, ax_agc = plt.subplots(1, 2, figsize=(10, 5))

        ax_agc[0].plot(t_plot_AGC_data_complete[::3].astype(np.float32), P_R_dB_data_complete[::3].astype(np.float32), label="Complete P_R data")
        ax_agc[0].plot(t_plot, P_R_data_dB.astype(np.float32), label="Downsampled P_R data")
        ax_agc[0].axhline(y=P_noisefloor_dB, color='gray', linestyle='--', label='Baseline (noise floor)')
        ax_agc[0].set_ylabel('Received Power [dB]')
        ax_agc[0].set_xlabel("Time [s]")
        ax_agc[0].grid()

        ax_agc[1].plot(t_plot_AGC_data_complete[::3], P_R_W_data_complete[::3].astype(np.float32))
        ax_agc[1].plot(t_plot, P_R_data.astype(np.float32))
        ax_agc[1].axhline(y=P_noisefloor_W, color='gray', linestyle='--')
        ax_agc[1].set_ylabel('Received Power [W]')
        ax_agc[1].set_xlabel("Time [s]")
        ax_agc[1].grid()

        title = f"Converted Power data for {BSR_titles[BSR_i]}"
        fig_agc.text(s=title, x=0.5, y=0.96, fontsize=16, ha='center', va='center', fontweight='bold')
        fig_agc.text(s=f'Data retrieved from {IQ_titles[BSR_i]}', x=0.5, y=0.91, fontsize=12, ha='center', va='center')  
        handles, labels = ax_agc[0].get_legend_handles_labels()
        fig_agc.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3)
        plt.tight_layout(rect=[0, 0.05, 1, 0.9])
        agcpow_png = os.path.join(BSR_dir_path, str(BSR_titles[BSR_i]) + '_AGC-Power_' + str(area_discretization) + 'deg.png')
        plt.savefig(agcpow_png, bbox_inches='tight', pad_inches=0.05, dpi=150)
        # plt.show()
        plt.close(fig_agc)

# **********************************************************************************************

""" Uncomment code below to plot maximum horizon off-boresight angles over time """

# fig_horizon, ax_horizon = plt.subplots(1, 2, figsize=(10, 4))

# for MEX_key in MEX_intpoint_angles_max.keys():
#     if len(MEX_intpoint_angles_max[MEX_key]) != 0:
#         ax_horizon[0].plot(np.linspace(0, 600, len(MEX_intpoint_angles_max[MEX_key])), MEX_intpoint_angles_max[MEX_key], label=MEX_key)
# ax_horizon[0].axhline(y=65.0, color='gray', linestyle='--')
# ax_horizon[0].set_ylabel('Maximum MEX Horizon Angle [deg]')
# ax_horizon[0].set_xlabel("Time [s]")
# ax_horizon[0].grid()

# for TGO_key in TGO_intpoint_angles_max.keys():
#     if len(TGO_intpoint_angles_max[TGO_key]) != 0:
#         ax_horizon[1].plot(np.linspace(0, 600, len(TGO_intpoint_angles_max[TGO_key])), TGO_intpoint_angles_max[TGO_key])
# ax_horizon[1].axhline(y=90.0, color='gray', linestyle='--')
# ax_horizon[1].set_ylabel('Maximum TGO Horizon Angle [deg]')
# ax_horizon[1].set_xlabel("Time [s]")
# ax_horizon[1].grid()

# handles, labels = ax_horizon[0].get_legend_handles_labels()
# fig_horizon.legend(ncols=4, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=1)
# plt.tight_layout(rect=[0, 0.1, 1, 1])
# horizon_png = os.path.join(analysis_path, 'results\\MaxHorizonAngles_' + str(area_discretization) + 'deg.png')
# plt.savefig(horizon_png, bbox_inches='tight', pad_inches=0.05, dpi=150)
# # plt.show()
# plt.close(fig_horizon)

# **********************************************************************************************

""" Uncomment code below to plot maximum relative S/C off-boresight angles over time """

# fig_freespaceoffbs, ax_freespaceoffbs = plt.subplots(1, 2, figsize=(10, 4))

# for MEX_key in MEX_freespaceoffbs_angles_max.keys():
#     if len(MEX_freespaceoffbs_angles_max[MEX_key]) != 0:
#         ax_freespaceoffbs[0].plot(np.linspace(0, 600, len(MEX_freespaceoffbs_angles_max[MEX_key])), MEX_freespaceoffbs_angles_max[MEX_key], label=MEX_key)
# ax_freespaceoffbs[0].axhline(y=65.0, color='gray', linestyle='--')
# ax_freespaceoffbs[0].set_ylabel('MEX to TGO off-boresight angle [deg]')
# ax_freespaceoffbs[0].set_xlabel("Time [s]")
# ax_freespaceoffbs[0].grid()

# for TGO_key in TGO_freespaceoffbs_angles_max.keys():
#     if len(TGO_freespaceoffbs_angles_max[TGO_key]) != 0:
#         ax_freespaceoffbs[1].plot(np.linspace(0, 600, len(TGO_freespaceoffbs_angles_max[TGO_key])), TGO_freespaceoffbs_angles_max[TGO_key])
# ax_freespaceoffbs[1].axhline(y=90.0, color='gray', linestyle='--')
# ax_freespaceoffbs[1].set_ylabel('TGO to MEX off-boresight angle [deg]')
# ax_freespaceoffbs[1].set_xlabel("Time [s]")
# ax_freespaceoffbs[1].grid()

# handles, labels = ax_freespaceoffbs[0].get_legend_handles_labels()
# fig_freespaceoffbs.legend(ncols=4, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=1)
# plt.tight_layout(rect=[0, 0.1, 1, 1])
# freespaceoffbs_png = os.path.join(analysis_path, 'results\\FreespaceOffbsAngles_' + str(area_discretization) + 'deg.png')
# plt.savefig(freespaceoffbs_png, bbox_inches='tight', pad_inches=0.05, dpi=150)
# # plt.show()
# plt.close(fig_freespaceoffbs)

# **********************************************************************************************

spice.kclear()

# **********************************************************************************************
