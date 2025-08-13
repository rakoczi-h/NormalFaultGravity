import numpy as np
from fault import Fault
import matplotlib.pyplot as plt
import os

directory = './plots/'
if not os.path.exists(directory):
    os.mkdir(directory)

# Making the fault
parameters = {"cx": 1.0,
              "cy": 0.0,
              "l": 2.0,
              "alpha": np.pi/4}

ranges = [[-2, 2], [-2,2]]
X = np.linspace(ranges[0][0], ranges[0][1], num=100)
Y = np.linspace(ranges[1][0], ranges[1][1], num=100)
X, Y = np.meshgrid(X, Y, indexing='ij')
X = np.expand_dims(X, axis=2)
Y = np.expand_dims(Y, axis=2)
Z = np.zeros(np.shape(X))
grid = np.c_[X, Y, Z]

fault = Fault(parameters=parameters, grid=grid)
fault.make_fault()
fault.plot_pixels(filename=os.path.join(directory, 'fault.png'))

# Forward modelling
ranges = [[-1.0, 1.0], [-1.0, 1.0]]
X = np.linspace(ranges[0][0], ranges[0][1], num=50)
Y = np.linspace(ranges[1][0], ranges[1][1], num=50)
X, Y = np.meshgrid(X, Y, indexing='ij')
X = np.expand_dims(X, axis=2)
Y = np.expand_dims(Y, axis=2)
Z = np.zeros(np.shape(X))
survey_coordinates = np.c_[X, Y, Z]

gz ,_ ,_ ,_  = fault.forward_model(num_components=50, # number of fourier transforms to include in sum
                                survey_coordinates=survey_coordinates, 
                                zero_pad=True, # zero padding before forward model
                                pad_width=[50, 50], # width of the zero padding
                                win=('tukey', 0.1) # applying tukey window
                                )

# Plotting the gravity grid
plt.imshow(gz, extent=(ranges[0][0], ranges[0][1], ranges[1][1], ranges[1][0]))
plt.colorbar(label=r'$\Delta$g [mGal]')
plt.ylabel('x [km]')
plt.xlabel('y [km]')
plt.savefig(os.path.join(directory, 'gravity.png'))
plt.close()