import numpy as np
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt

def load_stream(filename):
    return Stream(filename)


class Stream:
    def __init__(self,
                 filename,
    ):
        if filename.endswith(".str"):
            self.filename = filename
        else:
            self.filename = filename+".str"
              
        self.read_stream()

    def read_stream(self):
        """
        Read the stream file into a list of the lines. This shouldn't be called directly, but rather it gets called upon the initialization of the class.
        """
        with open(self.filename) as stream_file:
            # get all the lines as strings
            file_lines = stream_file.readlines() 

        # get number of points as int
        self.numpoints = int(file_lines[2])
        
        self.dwells = []
        self.x_points = []
        self.y_points = []
        self.point_number = []
        
        #dwells start in line 4 (pythonic 3)
        for n in range(3, len(file_lines)):
            split_line = file_lines[n].strip().split(' ')
            dwell = int(split_line[0])
            x = int(split_line[1])
            y = int(split_line[2])
            self.dwells.append(dwell)
            self.x_points.append(x)
            self.y_points.append(y)
            self.point_number.append(int(n))
            
    def points_to_array(self):
        """
        Conver the list of points from the stream file into an array of shape (3, N) for N points
        This shouldn't be called directly either. It sets object variables
        """
        
        self.x_array = np.asarray(self.x_points)
        self.y_array = np.asarray(self.y_points)
        self.dwell_array = np.asarray(self.dwells, dtype=np.float64)
        ## Gotta do some weird reshaping/transposing to make this work.
        self.point_array = np.concatenate((self.dwell_array, self.x_array, self.y_array)).reshape(3, -1).T
           
    def calculate_total_time(self, units='seconds'):
        """
        Using the point_array, this calculates the total time from the stream file. This *can* be called directly. It prints the value to the terminal, but also returns the result
        
        Parameters
        ----------
        units : str
            'seconds' or 'steps', with seconds being real-time and steps being the ThermoFisher 100 ns time steps. Default 'seconds'
            
        Returns
        --------
        total_time : float
            time in whatever units specified above
        
        """
    
        if hasattr(self, 'point_array') == False:
            self.points_to_array()
        print(self.dwell_array)
        total_time = np.sum(self.dwell_array)
        
        if units == 'seconds':
            total_time = total_time/1e7 # 100 ns steps are used in stream files
            print(f'The total time of the loaded stream file is {total_time:.2f} seconds')
        elif units == 'steps':
            total_time = total_time
            print(f'The total time of the loaded stream file is {total_time} stream time steps (100 ns)')
            
        return total_time

    def separate_pseudolayer(self, index_range):
        """
        Separate the stream points into pseudolayers. This should not be called directly, but gets used by condense_points
        """
        if hasattr(self, 'point_array') == False:
            self.points_to_array()
        approximate_number_of_arrays = self.numpoints/index_range
        point_array_regions = np.array_split(self.point_array, round(approximate_number_of_arrays))
        return point_array_regions
        
    def condense_points(self, radius, index_range, calculate_new_time = True):
        """
        Condense the points in the stream file but add their original dwell times. 
        
        This is a bit convoluted. It separates the stream files into "pseudolayers" based on the sequence of the points. This means that the first ~index_range of points is flattened into one layer of x,y points. These are clustered with DBSCAN. The total dwell time in each region and the centroid of the cluster are both calculated. The returned array is in the form N x [dwell, x, y] points. The size is then [N, 3] for N number of centroids.  
        
        
        Parameters
        -----------
        radius : float
            Radius to consider points as in the same region (not the total radius of region. More like nearest-neighbor distance for each point in the region)
            
        index_range : int
            The number of stream file points to consider as being in each "layer". Very arbitrary.
            
            
        Returns
        ---------
        condensed_points : [N, 3] array of N points with dwell, x, y
        
        """
        
        ### break the total array of points into sub-arrays based on the order they're fabricated. This is very basic, and separates into ~index_range sized chunks. The index_range will vary depending on structure complexity, since the following effectively flattens all of those points into a single "layer" of x,y points (and dwells)
        point_array_regions = self.separate_pseudolayer(index_range=index_range)
        
        condensed_point_list = []
        ### Iterate through the regions, then cluster the points using DBSCAN
        for n in range(len(point_array_regions)):
            dwells = point_array_regions[n][:, 0].astype(float)
            points = point_array_regions[n][:, 1:].astype(float)
            clusters = DBSCAN(eps=radius, min_samples = int(10)).fit(points)
            labels = clusters.labels_
            unique_labels = set(labels)
            
            ### for each cluster of points, calculate the total dwell time in that region (in seconds)
            for regions in range(len(unique_labels)):
                point_in_region = points[labels==regions, :]
                centroid = np.mean(point_in_region, axis=0).astype(int)
                dwell_sum = int(np.sum(dwells[labels==regions]))
                if (np.all((centroid>0)) and dwell_sum>0):
                    condensed_point = np.append(dwell_sum, centroid)
                    condensed_point_list.append(condensed_point)
                else:
                    pass
        
        condensed_point_array = np.asarray(condensed_point_list)
        
        if calculate_new_time == True:
            dwells = condensed_point_array[:, 0]
            print(f"The total time of the condensed points is {np.sum(dwells)/1e7:.2f} seconds")

        self.condensed_points = condensed_point_array
        return condensed_point_array
         
    def plot_cluster(self, points, labels, clusters):
        
        """
        This can plot the clusters generated by the condense_points function. It's unwieldy to plot every psuedolayer, so it's not currently built into the other functions. That's easy to do, but requires a bit of selectivity for figuring out which pseudolayers you actually want to see.
        
        """
        
        
        unique_labels = set(labels)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[clusters.core_sample_indices_] = True

        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            class_member_mask = labels == k
            xy = points[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], markerfacecolor=tuple(col))
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        plt.title(f"Estimated number of clusters: {n_clusters_}")
        plt.show()

    def output_stream(self, output_filename, type='condensed', **kwargs):
        """
        Output a stream file, either the original or a condensed one (so far)
        
        Parameters
        ----------
        type : str
            'original' or 'condensed'
            

        kwargs
            Options for editing/condensing the stream file
            
        Keyword Arguments
        ------------------
        'index range' : int
            number of points in each pseudolayer (approx)
            
        'radius' : int
            radius to consider DBSCAN points as neighbors
            
        'detailed_name' : bool
            if true, appends the condensing parameters to the filename
        
        """ 
        if output_filename.endswith(".str"):
            outfile = output_filename
        else:
            outfile= output_filename+".str"
        
    
        if type == 'condensed':
            index_range = kwargs.get('index_range', 1000)
            radius = kwargs.get('radius', 30)
            detailed_name = kwargs.get('detailed_name', False)
            condensed_output = self.condense_points(radius, index_range, calculate_new_time=False)
            stream_array = condensed_output
        elif type == 'original':
            stream_array = self.point_array
    

        if detailed_name == True:
            bare_filename = output_filename[:-4]
            new_filename = bare_filename + f"-i{index_range}-r{radius}.str"
            outfile = new_filename
            
        dwell_times = stream_array[:,0]
        xpos = stream_array[:, 1]
        ypos = stream_array [:, 2]
        numpoints = str(len(xpos))
        self.write_stream(outfile, numpoints, dwell_times, xpos, ypos)

    def write_stream(self, output_file, numpoints, dwell_times, xpos, ypos):

         with open(output_file, 'w') as f:
            f.write('s16\n1\n')
            f.write(str(numpoints)+'\n')
            for k in range(int(numpoints)):
                xstring = str(xpos[k])
                ystring = str(ypos[k])
                dwellstring = str(dwell_times[k])
                linestring = dwellstring+" "+xstring+" "+ystring+" "
                if k < int(numpoints)-1:
                    f.write(linestring + '\n')
                else:
                    f.write(linestring + " "+"0")   
            
    def dither_stream(self, rad, sigma, dwell_time_max_ms = 3.5):
        "Taking the centroids and dwells, space them out a little bit into multiple points so that the dwells are shorter. This uses the sigma from the gaussian spot to keep points in the same general neighborhood from the fab perspective."
        
        condensed_output = self.condense_points(70, 10000, calculate_new_time=False)
        stream_array = condensed_output
        dwell_time_max = dwell_time_max_ms * 1e-3 * 1e7 
        dwell_times = stream_array[:,0]
        xpos = stream_array[:, 1]
        ypos = stream_array [:, 2]

        out_xpos = []
        out_ypos = []
        out_dwells = []
        total_dwell = 0
        for number, _ in enumerate(xpos):
            x = xpos[number]
            y = ypos[number]
            mean = (x,y)
            cov = [[rad**2/(sigma**2),0], [0, rad**2/(sigma**2)]]
            number_dwells = int(dwell_times[number]/dwell_time_max)
            distributed_points = np.random.multivariate_normal(mean, cov=cov, size=(number_dwells)).astype(int)
            for points in distributed_points:
                distance = np.linalg.norm(points-np.array((x, y)))
                new_dwell = int(np.exp(-(distance**2) / (2*sigma**2))*(dwell_times[number]/number_dwells))
                if new_dwell > 10 :
                    out_xpos.append(points[0])
                    out_ypos.append(points[1])
                    out_dwells.append(new_dwell)
                    total_dwell += new_dwell
                else:
                    pass

        
        numpoints = len(out_xpos)
        outfile = "DitheredGyroid-GR220-3x3x2.str"
        self.write_stream(numpoints=numpoints, dwell_times=out_dwells, xpos=out_xpos, ypos=out_ypos, output_file=outfile)

        print(f"The total time of the dithered points is {total_dwell/1e7:.2f} seconds")  
    
        

if __name__ == "__main__":
    
    infile = "3x3x2-Gyroid-GR220-1063.2s.str"
    stream = load_stream(infile)
    
    
    stream.calculate_total_time()

    stream.dither_stream(rad = 5, sigma = 3)
    
    # outfile = "ThreeLeg45Deg-GR225-Condensed.str"
    # stream.output_stream(outfile, type='condensed', detailed_name = True, index_range = 400, radius=30)
    