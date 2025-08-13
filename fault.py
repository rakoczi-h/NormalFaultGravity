import numpy as np
from scipy.interpolate import splprep, splev
from skimage.filters import window
from scipy.interpolate import RegularGridInterpolator
import math
import plotly.io
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime

from utils import points_within_area, distance_to_line_segment, normalize, pad_grid

class Fault:
    """
    Class for normal fault models represented by vertical dispalcement profiles on a uniform grid.
    Parameters:
        parameters (dict): Contains the parameter values defining the location and geometry of the fault.
            keys:   cx : Central location in x in km. Default: 0.25
                    cy : Central location in y in km. Default: 0.25
                    alpha : Rotation angle of the trace around the z axis in radians. Default: pi/4
                    l : The length of the trace in km. Default: 1.5
                    DL_ratio : The ratio of downward displacement to trace length. Assuming linear relationship. Default: 0.02
                    dip : The dip angle in radians. Default: 70/180*pi
                    sym_factor : The ratio up uplift ot downthrow. Default: 0.2
                    Extent_ratio : The ratio of the extent of the displaced area perpendicular to the trace to the trace lenght. Default: 1.5
                    Displacement_order : The parameter controlling the width of the central displaced area. Default: 1.2
                    Blend_order: The parameter controlling the width of tapering at the tip. Default: 1.2
                    cz: The depth of the fault in km. Default: 0.1
                    density: The density contrast between the layers above and below the displaced surface in kg/m^3. Default: 800
        grid (array): The grid on which the displacement profile is computed.
                        A 3D array with shape m x n x 3, where m is the number of grid points in the x dimension and n is the number of gridpoint in y. Default: None
        displacement_profile (array): A 2D array with shape m x n with vertical displacement values describing the shape of the fault, in km. Default: None

    """
    def __init__(self, parameters: dict, grid=None, displacement_profile=None):
        self.parameters = parameters
        self.grid = grid
        self.displacement_profile = displacement_profile

    def __setattr__(self, name, value):
        if name == 'parameters':
            if value is not None:
                if not isinstance(value, dict):
                    raise ValueError("parameters has to be a dictionary")
                default_keys = ["cx", "cy", "alpha", "l", "DL_ratio", "Extent_ratio", "dip", "sym_factor", "Displacement_order", "Blend_order", "cz", "density"]
                for key in value.keys():
                    if not key in default_keys:
                        raise ValueError('At least one of the keys in parameters is not recognised.')
                value.setdefault("cx", 0.25)
                value.setdefault("cy", 0.25)
                value.setdefault("alpha", np.pi/4)
                value.setdefault("l", 1.5)
                value.setdefault("DL_ratio", 0.02)
                value.setdefault("dip", 70/180*np.pi)
                value.setdefault("sym_factor", 0.2)
                value.setdefault("Extent_ratio", 1.5)
                value.setdefault("Displacement_order", 1.2)
                value.setdefault("Blend_order", 1.2)
                value.setdefault("cz", 0.1)
                value.setdefault("density", 800.0)
                for i, key in enumerate(default_keys):
                    super().__setattr__(key, value[key])
        super().__setattr__(name, value)

    def make_fault(self, grid=None):
        """
        Function to make displacement profile from parameters.
        Parameters:
            grid (array): If None, the grid class attribute is used. Otherwise, it is overwritten. Default: None
                A 3D array with shape m x n x 3, where m is the number of grid points in the x dimension and n is the number of gridpoint in y. 
        Returns:
            array: The vertical displacement at each gridpoint. Shape is 2D with m x n. 
        """
        if grid is None:
            if self.grid is not None:
                grid = self.grid
            else:
                raise ValueError('Need to provide a grid to create the displacement profile')
        else:
            self.grid = grid

        Z = grid[:,:,2].flatten()
        original_shape = np.shape(grid[:,:,2])
        grid = grid[:,:,0:2].reshape(np.shape(grid)[0]*np.shape(grid)[1], 2) # flattening the grid

        DL_ratio = self.parameters['DL_ratio']
        dip = self.parameters['dip']
        sym_factor = self.parameters['sym_factor']
        Displacement_Order = self.parameters['Displacement_order']
        Blend_Order = self.parameters['Blend_order']
        l = self.parameters['l']

        if self.parameters['alpha'] < 0.0 or self.parameters['alpha'] > 2*np.pi:
            raise ValueError('The rotation angle should be between 0.0 and 2*pi.')
        elif self.parameters['alpha'] <= np.pi:
            sense = 'rhs'
        elif self.parameters['alpha'] > np.pi:
            sense = 'lhs'

        if dip > np.pi/2 or dip < 0.0:
            raise ValueError('The dip needs to be between 0 and 90 degrees.')

        sides = ['lhs', 'rhs']
        for side in sides:
            if side == 'rhs':
                if sense == 'rhs':
                    max_displacement = l * DL_ratio * sym_factor
                elif sense == 'lhs':
                    max_displacement = -l * DL_ratio
                xyboundary, _ = self.make_rupture_area()
            elif side == 'lhs':
                if sense == 'rhs':
                    max_displacement = -l * DL_ratio
                elif sense == 'lhs':
                    max_displacement = l * DL_ratio * sym_factor
                _, xyboundary= self.make_rupture_area()

            if max_displacement == 0:
                print('There is no displacement.')
            # Using the spline method
            n = 100 # must be even
            xyboundary = self.make_angular_bend_curve(xyboundary[:,0], xyboundary[:,1], n)
            displaced_gridpoints = points_within_area(xyboundary, grid)
            points_inside = grid[displaced_gridpoints]
            if not points_inside.size:
                self.displacement_profile = self.grid[:,:,2]
                continue
            # dividing it up into two sides
            xyboundary_a = xyboundary[:int(n/2),:]
            xyboundary_b = xyboundary[int(n/2):,:]

            # Calculating the distance from the edge and the corresponding displacement
            dist_a = self.distance_to_curve(xyboundary_a, points_inside)
            dist_b = self.distance_to_curve(xyboundary_b, points_inside)

            distance_from_edge = np.abs((dist_a**(-Displacement_Order) + dist_b**(-Displacement_Order))**(-Displacement_Order))
            max_distance_from_edge = np.abs(((self.l/2)**(-Displacement_Order) + (self.l/2)**(-Displacement_Order))**(-Displacement_Order))
            d, _, _ = normalize(distance_from_edge, minx=0, maxx=max_distance_from_edge)
            taper = ((np.cos(d*np.pi)+1)/2)**(Blend_Order)
            dz1 = (1-taper)*max_displacement

            # Calculating the distance from the trace line
            x, y = self.translate_coords()
            trace_end_1 = np.array([x[0], y[0]])
            trace_end_2 = np.array([x[1], y[1]])
            distance_to_trace = distance_to_line_segment(points_inside, np.repeat(np.expand_dims(trace_end_1, axis=0), np.shape(points_inside)[0], axis=0), np.repeat(np.expand_dims(trace_end_2, axis=0), np.shape(points_inside)[0], axis=0))

            if max_displacement < 0:
                dz2 = -np.tan(dip)*distance_to_trace
                dz = np.max(np.array([dz1, dz2]), axis=0)
            else:
                dz2 = np.tan(dip)*distance_to_trace
                dz = np.min(np.array([dz1, dz2]), axis=0)
            Z[displaced_gridpoints] = Z[displaced_gridpoints] + dz

        self.displacement_profile = np.reshape(Z, original_shape)
        
        return self.displacement_profile

    # --------------- Functions for making the fault model -----------------
    def make_rupture_area(self):
        """
        Create a polygonal rupture area around a fault trace.
        Returns:
            xyForward (array): Points on the "forward" side of the fault.
            xyBackward (array): Points on the "backward" side of the fault.
        """
        x, y = self.translate_coords()
        Extent_Ratio = self.parameters['Extent_ratio']

        # Two different values can be given for the backwards and forwards part of the fault
        # if one is given, then it is reused for both sides
        if np.isscalar(Extent_Ratio):
            Extent_Ratio = [Extent_Ratio, Extent_Ratio]
        elif len(Extent_Ratio) == 1:
            Extent_Ratio = [Extent_Ratio[0], Extent_Ratio[0]]
        elif len(Extent_Ratio) > 2:
            raise ValueError('The maximum number of elements in Extent_Ratio is 2.')

        # Compute cumulative distances
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        trace_length = np.sqrt(dx**2 + dy**2)

        xyForward = [[x[0], y[0]]]
        xyBackward = [[x[0], y[0]]]

        xMid = (x[0] + x[1]) / 2
        yMid = (y[0] + y[1]) / 2
        dist = trace_length/2

        widthF = dist * Extent_Ratio[0]
        widthB = dist * Extent_Ratio[1]

        if (y[1]-yMid) == 0: # need to deal with this separately, as otherwise dividing by 0
            alpha = np.pi/2
        else:
            alpha = np.arctan((x[1]-xMid)/(y[1]-yMid))

        xp1 = xMid + widthF*np.cos(alpha)
        yp1 = yMid - widthF*np.sin(alpha)
        xyForward.append([xp1, yp1])

        xp2 = xMid - widthB*np.cos(alpha)
        yp2 = yMid + widthB*np.sin(alpha)
        xyBackward.append([xp2, yp2])

        # Append the final point
        xyForward.append([x[-1], y[-1]])
        xyBackward.append([x[-1], y[-1]])

        xyForward = np.array(xyForward)
        xyBackward = np.array(xyBackward)

        return xyForward, xyBackward
    
    def make_angular_bend_curve(self, xp, yp, n):
        """
        Generates a curve from points (xp, yp) using parametric spline interpolation.
        Parameters:
            xp (array): x-coordinates of input points.
            yp (array): y-coordinates of input points.
            n (int): Number of points to interpolate on the curve.
        Returns:
            array: An n x 2 array with interpolated x and y coordinates.
        """
        if not isinstance(xp, np.ndarray) or not isinstance(yp, np.ndarray):
            raise ValueError('xp and yp have to be arrays')
        if xp.ndim != 1 or yp.ndim !=1:
            raise ValueError('xp and yp have to be vectors.')

        # Define angular parameter over [0, 2π]
        th = np.linspace(0, 2 * np.pi, len(xp))

        # Create a periodic spline parameterization
        tck, _ = splprep([xp, yp], u=th, k=2) # the degree is set to 2
        # Evaluate the spline at n points
        th_new = np.linspace(0, 2 * np.pi, n)
        x_new, y_new = splev(th_new, tck)

        xy_curve = np.vstack((x_new, y_new)).T
        return xy_curve
    
    def translate_coords(self):
        """
        Defines translation between end points x1, y1, x2, y2 and central location, lenght and rotation angle (cx, cy, l, alpha)
        Returns:
            [x1, x2], [y1, y2]: two lists of the x and y coordinates of the endpoints of the trace.
        """
        cx = self.parameters['cx']
        cy = self.parameters['cy']
        alpha= self.parameters['alpha']
        l = self.parameters['l']
        x1 = l/2*np.cos(alpha)+cx
        y1 = l/2*np.sin(alpha)+cy
        x2 = -l/2*np.cos(alpha)+cx
        y2 = -l/2*np.sin(alpha)+cy
        return [x1, x2], [y1, y2]
    
    def distance_to_curve(self, xyboundary, points):
        """
        Finding the minimum distance to an arbitrary curve from each point on a grid.
        Parameters:
            xyboundary (array): An n x 2 array containing the x-y coordinates of the points on the arbitrary curve.
            points (array): An m x 2 array containing the x-y coordinates of the points on the grid.
        Returns:
            array: An m long vector of minimum distances fro each grid point.
        """
        # Getting the arrays to the right shape for parallelization
        x = points[:,0]
        x = np.repeat(np.expand_dims(x, axis=0), np.shape(xyboundary)[0], axis=0)
        y = points[:,1]
        y = np.repeat(np.expand_dims(y, axis=0), np.shape(xyboundary)[0], axis=0)
        xyboundary = np.expand_dims(xyboundary, axis=2)
        xyboundary = np.repeat(xyboundary, np.shape(points)[0], axis=2)
        # finding the minimum distance for each grid point and the point on the curve it corresponds to
        dist = np.sqrt((xyboundary[:,0,:]-x)**2 + (xyboundary[:,1,:]-y)**2)
        min_loc = np.argmin(dist, axis=0)
        min_dist = np.min(dist, axis=0)
        min_points = xyboundary[min_loc, :, 0] # the minimum distance from the original set of points on the boundary

        # Finding the neighbouring points to the original points corresponding to the minimal distances
        min_points_reshaped = np.repeat(np.expand_dims(min_points, axis=0), np.shape(xyboundary)[0], axis=0)
        dist = np.sqrt((xyboundary[:,0,:]-min_points_reshaped[:,:,0])**2 + (xyboundary[:,1,:]-min_points_reshaped[:,:,1])**2)
        dist_ordered = np.argsort(dist, axis=0)
        min_points1 = xyboundary[dist_ordered[1], :, 0]
        min_points2 = xyboundary[dist_ordered[2], :, 0]
        # drawing lines from the original point to the two neighbouring points and finding the minimum distance to those lines
        min_dist1 = distance_to_line_segment(points, min_points, min_points1)
        min_dist2 = distance_to_line_segment(points, min_points, min_points2)
        # picking the minimum value from the final three obtained values.
        min_dist = np.min(np.vstack([min_dist, min_dist1, min_dist2]), axis=0)
        return min_dist
    
    # --------------- Functions to compute survey ----------------------

    def forward_model(self, num_components=50, depth=None, survey_coordinates=None, remove_min=True, zero_pad=True, pad_width=[50, 50], win=None):
        """
        The fourier domain forward model after R.L Parker (1972)
        Takes the displacement model, depth and density contrast and turns it into a gravity signal on the surface. The surface is assume to be completely flat.
        Parameters:
            num_compontnents (int): The number of fourier components to consider.
            depth (float): Depth of the fault in km.
                If not given, try to use the depth from the parameter dictionary. Default: None
            survey_coordinates (array): n x m x 3 the survey coordinats, assumed to lie on a vertically flat surface.
                If not given, then the same grid is used as for the fault.
            remove_min (bool): If True, the computed gravity is shifted to 0 minimum. Default: True
            zero_pad (bool): If True, the fault grid is padded with zeros before forward modelling. Default: True
            pad_width (list): List of ints with two elements defining the width of the zero padding area in number of grid points in the x and y dimensions. Default: [50, 50]
            win (list): If None, it is passed as the first argument to skimage.filters.window. Example: ('tukey', 0.1). Default: None
        Returns:
            g (array): Gravity values either on the grid of survey_coordinates or on the same grid as the displacement_profile.
            k_mag (array): The magnitude of the k-vector that was computed from the displacement grid.
            R1 (array): The intermediate fourier domain step of the forward model, same size as the padded grid. Useful when augmenting.
            grid (array): The grid that was used during the forward model, including padding.
        """
        if self.displacement_profile is None:
            self.make_fault()
        displacement_profile = self.displacement_profile

        if depth is None:
            depth = self.parameters["cz"]
            if depth is None:
                raise ValueError("Need to provide the depth of the fault")
        depth = depth*1000 # in m

        if np.isnan(displacement_profile).any():
            raise ValueError('Found NaN in displacement model.')
        
        if win is not None:
            w = window(win, np.shape(displacement_profile))
            displacement_profile = w*displacement_profile
        if zero_pad:
            displacement_profile = np.pad(displacement_profile, 
                                               pad_width=((pad_width[0], pad_width[0]),(pad_width[1], pad_width[1])))
            # making padded grid
            grid = pad_grid(self.grid, pad_width, square=False)
        else:
            grid = self.grid

        k_mag = self.wavenumber(grid=grid*1000) # changing grid to m and computing k vector
        R1 = np.zeros(np.shape(displacement_profile))
        for n in range(num_components):
            f1 = np.fft.fft2((displacement_profile*1000)**(n+1)) # changing to m
            r1 = np.complex128(k_mag**(n)/math.factorial(n+1)*f1) # r represent spatial domain, k represent k domain
            R1 = R1+r1
        G = 6.67430*10**(-11)# Nm**2kg**(-2)
        f_g = -2*np.pi*G*np.exp((-k_mag)*depth)*R1*self.parameters['density']
        g = np.fft.ifft2(f_g)
        g_orig = np.real(g) * 1e5 # changing to mGal

        if survey_coordinates is None:
            if remove_min:
                return g_orig-np.min(g_orig), k_mag, R1, grid
            else:
                return g_orig, k_mag, R1, grid
        else:
            x = np.linspace(np.min(grid[:,:,0]), np.max(grid[:,:,0]), num=np.shape(grid)[1])
            y = np.linspace(np.min(grid[:,:,1]), np.max(grid[:,:,1]), num=np.shape(grid)[0])
            func = RegularGridInterpolator((x, y), g_orig)
            g_new = func(survey_coordinates[:,:,:2].flatten())
            g_new = np.reshape(g_new, np.shape(survey_coordinates[:,:,0]))
            if remove_min:
                return g_new-np.min(g_new), k_mag, R1, grid
            else:
                return g_new, k_mag, R1, grid
            
    def wavenumber(self, grid=None):
        """
        Calculates the magnitude of the wavevector at each point in the frequency domain.
        Parameters:
            grid (array): The spatial grid, assumed to be in m. Default: None
        Returns:
            array: An array of the wavevector magnitudes, with same shape as grid.
        """
        if grid is None:
            grid = self.grid*1000
        X = grid[:,:,0]
        Y = grid[:,:,1] 
        numrows = np.shape(X)[0]
        numcolumns = np.shape(Y)[1]
        longx = np.max(X)-np.min(X)
        longy = np.max(Y)-np.min(Y)

        # first quadrant of the frequency magnitude
        frequency = np.zeros((abs((numrows // 2) + 1), abs((numcolumns // 2) + 1)))
        for f in range(1, int((numrows/2) + 2)):
            for g in range(1, int((numcolumns/2) + 2)):
                frequency[f-1, g-1] = np.sqrt(((f-1) / longx) ** 2 + ((g-1) / longy) ** 2)

        # mirroring into full frequency range
        frequency2 = np.fliplr(frequency)
        frequency3 = np.flipud(frequency)
        frequency4 = np.fliplr(np.flipud(frequency))

        # ensuring there are not duplicates at the centre
        entero = round(numcolumns / 2)
        if ((numcolumns / 2) - entero) == 0:
            frequency2 = np.delete(frequency2, 0, axis=1)
            frequency3 = np.delete(frequency3, 0, axis=0)
            frequency4 = np.delete(np.delete(frequency4, 0, axis=1), 0, axis=0)
            frequencytotal = np.concatenate((np.concatenate((frequency, frequency2), axis=1), # combine into full grid
                                            np.concatenate((frequency3, frequency4), axis=1)))
        else:
            frequencytotal = np.concatenate((np.concatenate((frequency, frequency2), axis=1),
                                            np.concatenate((frequency3, frequency4), axis=1)))
        frequencytotal = frequencytotal[:-1, :]
        frequencytotal = frequencytotal[:, :-1]
        frequencytotal = frequencytotal * (2 * np.pi) # in units of radians per meter
        return frequencytotal

    def forward_from_fourier(self, k_mag, R1, grid, survey_coordinates=None, depths=None, densities=None, remove_min=True):
        """
        Continue forward model from fourier domain. Useful when augmenting with density or depth.
        Parameters:
            k_mag (array): Array of wavevector magnitudes. Has to be same shape as R1 (and correspond to the same grid).
            R1 (array): The fourier domain representation of the fault displacement.
            grid (array): The grid on which the displacement profile is defined.
            survey_coordinates (array): 3D array with the survey coordinates on a flat surface. Default: None
            depths (list): Depths to compute the forward model for. If None, taken from the class. Default: None
            desities (list): Densities to compute the forward model for. If None, taken from the claws. Default: None
            remove_min (bool): If True, the minimum of the computed gravity survey is shifted to 0. Default: True
        Returns:
            gravity (list): The list of of gravity arrays corresponding to each required set of densities and depths.
            depth_output (list): The depths corresponding to each gravity grid.
            density_output (list): The densities corresponding to each gravity grid.
        """
        G = 6.67430*10**(-11)# Nm**2kg**(-2)
        if depths is None:
            depths = [self.parameters['cz']]
        if densities is None:
            densities = [self.parameters['density']]
        gravity = []
        depth_output = []
        density_output = []
        for d in depths:
            for rho in densities:
                f_g = -2*np.pi*G*np.exp((-k_mag)*d*1000)*R1*rho
                g = np.fft.ifft2(f_g)
                g_orig = np.real(g) * 1e5 # changing to mGal
                if survey_coordinates is None:
                    if remove_min:
                        gravity.append(g_orig-np.min(g_orig))
                    else:
                        gravity.append(g_orig)
                else:
                    x = np.linspace(np.min(grid[:,:,0]), np.max(grid[:,:,0]), num=np.shape(grid)[1])
                    y = np.linspace(np.min(grid[:,:,1]), np.max(grid[:,:,1]), num=np.shape(grid)[0])
                    func = RegularGridInterpolator((x, y), g_orig)
                    g_new = func(survey_coordinates[:,:,:2].flatten())
                    g_new = np.reshape(g_new, np.shape(survey_coordinates[:,:,0]))
                    if remove_min:
                        gravity.append(g_new-np.min(g_new))
                    else:
                        gravity.append(g_new)
                depth_output.append(d)
                density_output.append(rho)
        return gravity, depth_output, density_output
    # ------------------- Plotting Tools ---------------------------

    def plot_pixels(self, filename='pixels_fault.png', survey_coordinates=None):
        """
        Creates a simple pixelised image of the survey. Can only be done for gridded data.
        Parameters:
            filename (str): The name where to save the image.
            survey_coordinates: 3D array of survey coordinates on a flat grid. Default: None
        """
        if self.displacement_profile is None:
            raise ValueError("displacement profile not given")

        plt.imshow(np.reshape(self.displacement_profile, np.shape(self.grid[:,:,0])), extent=(np.min(self.grid[:,:,1]), np.max(self.grid[:,:,1]), np.max(self.grid[:,:,0]), np.min(self.grid[:,:,0])))
        plt.colorbar(label=r'$\Delta$z [km]')
        if survey_coordinates is not None:
            plt.scatter(survey_coordinates[:,:,0].flatten(), survey_coordinates[:,:,1].flatten(), s=1, marker='o', color='black')
        plt.scatter(self.parameters['cy'], self.parameters['cx'], color='red', marker='x', s=2)
        plt.xlabel('y [km]')
        plt.ylabel('x [km]')
        plt.savefig(filename)
        plt.close()

    def plot_3D_surface(self, survey_coordinates=None, filename='3D_fault.png', depth=None):
        """
        Makes 3D plot of the fault.
        Parameters
        ----------
            filename (str): The location where the image will be saved. The file format is infered from this string, can either be .png or .html.
            depth (float): The depth of the fault. If not given, try to get from parameters dictionary or set to 0. Default: None
            survey_coordinates: 3D array of survey coordinates on a flat grid. Default: None
        """
        plotly.io.templates.default = 'plotly_white'
        if self.displacement_profile is None:
            raise ValueError("displacement profile not given")
        X = self.grid[:,:,0]
        Y = self.grid[:,:,1] # in km
        model = self.displacement_profile/1000.0
        depth = self.parameters['cz']
        model = np.reshape(model, np.shape(X))
        aspect_ratios = [1, 1, 0.2]
        fig = go.Figure(data=[go.Surface(z=model-depth, x=X, y=Y, cmax=np.max(-depth)-depth/100, cmin=np.min(-depth)+depth/100,
                                      contours = {"z": {"show": True, "start": np.min(model-depth), "end": np.max(model-depth), "size": (np.max(model-depth)-np.min(model-depth))/10, "color":"black"}},
                                      colorbar={"title": 'z [km]'})])

        if not survey_coordinates is None:
            fig.add_scatter3d(x=survey_coordinates[:,:,0].flatten(), y=survey_coordinates[:,:,1].flatten(), z=survey_coordinates[:,:,2].flatten(), mode='markers', marker={'color': 'black', 'size': 1, 'opacity': 0.2}, showlegend=None)
            fig.update_layout(coloraxis_showscale=False)

        fig.update_scenes(aspectratio={"x":aspect_ratios[0], "y":aspect_ratios[1], "z":aspect_ratios[2]},
                          zaxis={"nticks": 2, "range": [np.min(model-depth), 0.01]},
                          xaxis={'range': [np.min(X), np.max(X)]},
                          yaxis={'range': [np.min(Y), np.max(Y)]},
                          xaxis_title="x [km]",
                          yaxis_title="y [km]",
                          zaxis_title="z [km]"
        )

        fig.update_layout(scene=dict(aspectmode="data"))
        if filename[-5:] == '.html':
            fig.write_html(filename)
        elif filename[-4:] == '.png':
            fig.write_image(filename)
        else:
            raise ValueError("Only .html and .png file extensions are allowed")
        plt.close()


