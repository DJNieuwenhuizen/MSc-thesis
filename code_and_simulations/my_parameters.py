# **********************************************************************************************
#
#  my_parameters.py
#
# **********************************************************************************************
#
#  Description: 
#  |  Parameters used for the MEX-TGO BSR measurement planning and analysis
#     |  Mars_param contains planetary parameters for Mars
#     |  BSR_param contains radar signal and S/C parameters
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

import numpy as np

# **********************************************************************************************

""" Required calculations before writing to parameter dictionaries """

radii_ell = [3396.19, 3396.19, 3376.20]

eccentricity = np.sqrt(1 - ((radii_ell[2]**2) / (radii_ell[0]**2)))
surfacearea = (2 * np.pi * radii_ell[0]**2) + (np.pi * (radii_ell[2]**2 / eccentricity) * np.log((1+eccentricity)/(1-eccentricity)))
flattening_factor = (radii_ell[0] - radii_ell[2]) / radii_ell[0]

f_UHF = 437.1 * 10**6
c = 3 * 10**8
lambda_UHF = c / f_UHF

k_Boltzmann = 1.380649 * 10**-23 # J/K; Boltzmann constant
T_noise = 500 # K; noise temperature
B = 1 * 10**6 # Hz; bandwidth
P_noisefloor_W = k_Boltzmann * T_noise * B # W; noise power floor
P_noisefloor_dB = 10 * np.log10(P_noisefloor_W * 1000) # dB; noise power floor

# **********************************************************************************************

""" Definition of parameter dictionaries for use in the main scripts """

Mars_param = {# Radii values taken from original 'compute_equiemission_point.py' code:
              'radii_ell': radii_ell, # km; radii ellipsoidal Mars
              'radii_sph': [3396.19, 3396.19, 3396.19], # km; radii spherical Mars

              'eccentricity': eccentricity, # -; eccentricity ellipsoidal Mars
              'surfacearea': surfacearea, # km^2; surface area ellipsoidal Mars
              'flattening_factor': flattening_factor, # -; flattening factor ellipsoidal Mars
}

BSR_param = {'f_UHF': f_UHF, # Hz; frequency
             'c': c, # m/s; speed of light in vacuum
             'lambda_UHF': lambda_UHF, # m; wavelength
             
             # MEX parameters
             'P_T': 5, # W; transmitter (MEX) power
             'MEX_antennapattern': [[-65, -50, -35, -10, 0, 10, 35, 50, 65], # deg; off-boresight angles
                                    [-10.0, -2.0, 3.0, 5.5, 6.0, 5.5, 3.0, -2.0, -10.0]], # dB; off-boresight gains
             
             # TGO parameters
             # 'TGO_antennapattern': [[-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90], # deg; off-boresight angles
             #                        [-7.1, -4.1, -1.3, 0.8, 2.1, 3.8, 5.2, 5.6, 5.9, 6.1, 5.9, 5.6, 5.2, 3.8, 2.1, 0.8, -1.3, -4.1, -7.1]], # dB; off-boresight gains (401 MHz return link)
             'TGO_antennapattern': [[-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90], # deg; off-boresight angles
                                    [-16.8, -9.2, -4.9, -1.6, 0.9, 3.1, 4.4, 5.1, 5.9, 6.1, 5.9, 5.1, 4.4, 3.1, 0.9, -1.6, -4.9, -9.2, -16.8]], # dB; off-boresight gains (437.1 MHz forward link)

             # Transfer function parameters
             'k_Boltzmann': k_Boltzmann, # J/K; Boltzmann constant
             'T_noise': T_noise, # K; noise temperature
             'B': B, # Hz; bandwidth
             'AGC_baseline': 124,
             'P_noisefloor_W': P_noisefloor_W, # W; noise power floor
             'P_noisefloor_dB': P_noisefloor_dB # dB; noise power floor
}

# **********************************************************************************************
