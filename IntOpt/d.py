import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import curve_fit
from PIL import Image
import matplotlib.pyplot as plt
import json
from IntOptFnc import *

class Fitting():
    def __init__(self, image, center, d):
        self.center = center
        self.d = d
        self.image = image[center[0] - d//2: center[0] + d//2, center[1] - d//2: center[1] + d//2]
        self.sat = self.check_saturation()

    def check_saturation(self):
        threshold = 254
        saturated_mask = im >= threshold
        saturated_per = np.sum(saturated_mask) / (im.shape[0] * im.shape[1]) * 100
        if saturated_per > 1:
            return True
        else:
            return False

    def start_fit(self):
        # return the fitted amplitude
        def gaussian_2d(xy, amplitude, x_mean, y_mean, x_stddev, y_stddev, rotation):
            x, y = xy
            x_rot = (x - x_mean) * np.cos(rotation) - (y - y_mean) * np.sin(rotation)
            y_rot = (x - x_mean) * np.sin(rotation) + (y - y_mean) * np.cos(rotation)
            exponent = -((x_rot / x_stddev) ** 2 + (y_rot / y_stddev) ** 2) / 2
            return amplitude * np.exp(exponent)

        x = np.arange(self.image.shape[1])
        y = np.arange(self.image.shape[0])
        xx, yy = np.meshgrid(x, y)

        xy = (xx.flatten(), yy.flatten())
        data = self.image.flatten()

        bounds = (0, [np.inf, np.inf, np.inf, np.inf, np.inf, 2 * np.pi])
        popt, _ = curve_fit(gaussian_2d, xy, data, p0=[np.max(self.image), self.d//2, self.d//2, 10, 10, 0],
                            bounds=bounds)

        self.fitted_data = gaussian_2d((xx, yy), *popt).reshape(self.image.shape)
        self.I0 = np.max(self.fitted_data)

    def show_image(self):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(self.image, cmap='viridis')
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(self.fitted_data, cmap='viridis')
        plt.title(f'Fitted Gaussian with I0 = {self.I0:.2f}')

        plt.tight_layout()
        plt.show()

def show_local_maxima(image, local_max):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')
    plt.scatter(np.nonzero(local_max)[1], np.nonzero(local_max)[0], color='red', s=5)
    plt.title('Local Maxima')

    plt.tight_layout()
    plt.show()


# Load the image, need specify the dimension manually
int_ref = 154.14
attempt = 0
im = np.array(Image.open(f'IntOptD/example_i.png'))[450:510, 640-180:640+180]

# Load the parameters
# ntrap: number of traps; positions: center positions of the traps; amplitudes: RF amplitudes of each trap
# with open(f'./IntOptD/params_{attempt}.json', 'r') as f:
#     params = json.load(f)
#     ntrap, positions, amplitudes = params['ntrap'], np.array(params['positions']), np.array(params['amplitudes'])
#     print(f'Initial ntrap = {ntrap} \n Initial RF amplitudes = {amplitudes}')
#
# if attempt == 0:
#     # Find the local maxima using maximum_filter
#     # Specify the positions in the first run, and use it thereafter in one optimization
#     neighborhood_size = 50  # Define the size of the neighborhood for finding local maxima
#     local_max = (im == maximum_filter(im, neighborhood_size))
#     positions = np.flip(np.argwhere(local_max), axis=0)
#     # positions = np.argwhere(local_max)


# get rid of the points that are too close to each other
# i = 0
# while i < len(positions):
#     if np.abs(positions[i][1] - positions[i-1][1]) < 5:
#         positions = np.delete(positions, i, axis=0)
#     elif positions[i][0] < 10 or positions[i][0] > 50:
#         positions = np.delete(positions, i, axis=0)
#     elif positions[i][1] < 15 or positions[i][1] > 345:
#         positions = np.delete(positions, i, axis=0)
#     else:
#         i += 1
# # maybe adjusting neighbourhood_size can help
# if len(positions) > ntrap:
#     print(f'Too many local maxima!!! -- {len(positions)}')
#     print(positions)
#     exit()
# elif len(positions) < ntrap:
#     print(f'Not enough local maxima!!! -- {len(positions)}')
#     print(positions)
#
#     exit()
# show_local_maxima(im, local_max)

# start fitting, get I0 of each trap, and feedback

print(im.shape)
r = Fitting(im, [30, 180], 28)
r.start_fit()
print(r.I0)
r.show_image()

# intensities = []
# saturations = []
# for i in range(ntrap):
#     r = Fitting(im, positions[i], 28)
#     r.start_fit()
#     intensities.append(r.I0)
#     saturations.append(r.sat)
# if np.any(saturations):
#     # print the saturated traps
#     print(f'Saturated traps: {np.nonzero(saturations)[0]}')
#     exit()
# else:
#     print(f'Mean Intensity = {np.mean(intensities)}')
#     print(f'stdev = {np.std(intensities)}')
#     for i in range(amplitudes.shape[0]):
#         amplitudes[i] *= np.sqrt(int_ref / intensities[i])
#     amplitudes = np.round(amplitudes, 4)
#     print(f'Final RF amplitudes = {amplitudes}')
#
#     # save the parameters
#     data = {'ntrap': ntrap, 'positions': positions.tolist(), 'amplitudes': amplitudes.tolist()}
#     with open(f'IntOptD/params_{attempt+1}.json', 'w') as f:
#         json.dump(data, f)
