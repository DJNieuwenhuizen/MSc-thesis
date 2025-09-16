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

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from scipy.optimize import curve_fit

# **********************************************************************************************

""" Functions for the MEX and TGO antenna patterns """

def gaussian(x, A, B, C):
    return A * np.exp(-B * x**2) + C

def calc_gain_gaussian(SC_offbs_data_angles, SC_offbs_data_gains):
    """
    calc_gain_gaussian Calculate the Gaussian parameters for antenna pattern fitting

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
