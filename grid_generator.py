# Script to generate a grid of models
# Standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Progress bar
from tqdm import tqdm

# Function used for Sobol sampling
from scipy.stats import qmc

# For integrating shell commands in Python
from subprocess import call

# For multiprocessing/parallel computing
import multiprocessing as mp

# Grid generator class with samples as variables
class GridGenerator:
    def __init__(self, lower_bounds, upper_bounds, params, n_samples, new = False):
        self.parameters = params
        self.number_of_samples = n_samples[0]   # Existing number of samples
        self.new_number_of_samples = n_samples[1]   # New number of samples
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.new = new    # If new = True, new samples are generated
        self.samples = None     # To store the newest samples for which models have to be generated
        self.all_samples = None     # To store all samples which have been generated so far
        self.output_filestrings = ["pdr", "cool", "heat", "line", "spop", "opdp"]
        
    # Function to sample the initial parameter space
    def sample_generator(self):
        """
        Lower and upper bounds are in log scale, but the final samples are normal.
        n_samples should be a power of 2.
        new: if True, new samples are generated 
        """
        index_numbers = np.linspace(0, self.number_of_samples - 1, self.number_of_samples, 
                                    dtype = "int")
        indices = [f"model_{index_number}" for index_number in index_numbers]
        sobol_sampler = qmc.Sobol(d = 3, seed = 1234)
        sobol_sample = sobol_sampler.random(n = self.number_of_samples)
        scaled_sobol_sample = qmc.scale(sobol_sample, self.lower_bounds, self.upper_bounds)
        df_sobol_sample = pd.DataFrame(data = scaled_sobol_sample, 
                                       columns = self.parameters, 
                                       index = indices)    
        self.samples = df_sobol_sample
        
        # Generating new samples over and above the existing samples
        if self.new:
            _ = sobol_sampler.reset()
            _ = sobol_sampler.fast_forward(n = self.number_of_samples)
            index_numbers = np.linspace(self.number_of_samples, 
                                        self.number_of_samples + self.new_number_of_samples - 1,
                                        self.new_number_of_samples, dtype = "int")
            indices = [f"model_{index_number}" for index_number in index_numbers]
            new_sobol_sample = sobol_sampler.random(n = self.new_number_of_samples)
            new_scaled_sobol_sample = qmc.scale(new_sobol_sample, self.lower_bounds, self.upper_bounds)
            df_sobol_sample = pd.DataFrame(data = new_scaled_sobol_sample, 
                                           columns = self.parameters, 
                                           index = indices)
            self.all_samples = self.samples
            self.samples = df_sobol_sample
            
        self.samples = 10**self.samples
        self.all_samples = 10**self.samples
        

    # Creates a file initializing gas density at various points in the cloud
    def create_density_file(self, n_H, n_points, x_max = 100):
        """
        x_max is in pc.
        n_points is the number of points in the cloud where the density is initialized.
        Name of the file is 1DnXX.dat, where XX refers to log10(n_H) rounded to 2 decimal places.
        """
        max_size = x_max  # maximum depth of cloud in the x-axis; may need to vary this
        n_H_array = np.zeros((n_points, 4))
        n_H_array[:,0] = np.logspace(-8, np.log10(max_size), n_points)
        n_H_array[:,3] = n_H*np.ones((n_points,))
        file_string = f"1Dn{int(np.round(np.log10(n_H), decimals = 2))}.dat"
        call(["touch", f"1Dn{int(np.round(np.log10(n_H), decimals = 2))}.dat"])
        np.savetxt(file_string, n_H_array, delimiter = "  ")
        print(f"Created and saved {file_string}")
        return file_string
       
    # Function to modify params.dat with the necessary initial parameter values
    def modify_params(self, g_uv, n_H, cr_zeta, output_prefix, min_Av = 1e-6, 
                      max_Av = 20, n_points = 300):
        """
        output_prefix is the same as the name of the folder where output files are stored.
        """
        print(f"Running {output_prefix}...")
        # Creating the density array
        density_filestring = self.create_density_file(n_H, n_points)
        call(f"cp params.dat params_{output_prefix}.dat", shell = True)

        # Reading params.dat and making the necessary changes
        with open("params.dat", "r") as params, open(f"params_{output_prefix}.dat", 
                                                     "w") as params_output:
            params_lines = params.readlines()
            params_lines[3] = f"{density_filestring}\t\t\t !Input file (20 char. max)\n"
            params_lines[4] = f"{output_prefix}\t\t\t !Output file (20 char. max)\n"
            params_lines[11] = f"{cr_zeta}\t\t\t !Cosmic Rays (s^-1)\n"
            params_lines[40] = f"{g_uv}\t\t\t !G0 (in Draine field units) -x to +x in 1D\n"
            for params_line in params_lines:
                params_output.write(params_line)
                
            # Renaming the output file
            os.rename(f"params_{output_prefix}.dat", "params.dat")
        print("Finished modifying params.dat!")
        
        
    # Creates output directories for each model
    def create_output_directories(self):
        """Creates directories where outputs for each model are stored.
        """
        # prefixes = self.samples.index.to_list()
        print("Creating directories...")
        if not self.new:
            last_model_index = len(self.samples) - 1
            call(f"mkdir all_runs/model_{{0..{last_model_index}}}", shell = True)
        else:
            first_model_index = int(grid_generator.samples.index[0].split("_")[-1])
            last_model_index = int(grid_generator.samples.index[-1].split("_")[-1])
            call(f"mkdir all_runs/model_{{{first_model_index}..{last_model_index}}}", shell = True)
            
    
    # Executable function that is fed to the MultiProcessing Pool
    def executable_function(self, prefix):
        """
        Input is the index name of each sample; the sample can be accessed from the class.
        """
        sample = self.samples.loc[prefix]
        print(sample)
        print(f"Creating directory {prefix}...")
        call(["mkdir", f"all_runs/{prefix}"])
        
        # Modifying params.dat and calling 3DPDR for each file
        print(f"--------------------MODEL: {prefix}--------------------")
        self.modify_params(g_uv = sample[0], n_H = sample[1], cr_zeta = sample[2],
                           output_prefix = prefix)
        print("Calling 3DPDR...")
        call("3DPDR")
    
        
if __name__ in "__main__":
    
    # Bounds and samples are in log space
    lb = np.array([1, 2, -17])
    ub = np.array([5, 7, -15]) 
    
    # Generating samples
    number_of_samples = 1024
    new_number_of_samples = 8192
    print("Sample generation...")
    grid_generator = GridGenerator(lower_bounds = lb, upper_bounds = ub, 
                                   params = ["g_uv", "n_H", "zeta_CR"],
                                   n_samples = [number_of_samples,
                                                new_number_of_samples],
                                   new = True)
    grid_generator.sample_generator()
    
    # Uncomment if you wish to visualize old and new samples on a scatter plot
    # fig, ax = plt.subplots(1,1)
    # ax.scatter(old[:, 0], old[:, 1], s = 3, color = "green", label = "Old")
    # ax.scatter(new[:, 0], new[:, 1], s = 3, color = "blue", label = "New")
    # ax.legend()
    # fig.savefig("sample_comparison.png", dpi = 400)
    # fig.show()
    
    grid_generator.samples.to_csv("samples.csv")
    grid_generator.all_samples.to_csv("samples_old.csv")
    
    # Creating output directories
    prefixes = grid_generator.samples.index.to_list()
    grid_generator.create_output_directories()

    # Running a grid of models
    print("Running a grid of models: ")
    print("Number of models: ", len(prefixes))
    
    for i, prefix in enumerate(tqdm(prefixes), start = 1):
        with open("grid_gen_progress.txt", "a+") as f:
            print(f"{i} of {len(prefixes)} models: {i*100/len(prefixes)} percent done.", 
                  file = f, flush = True)
        
        print(f"--------------------{prefix}--------------------")
        sample = grid_generator.samples.loc[prefix]
        grid_generator.modify_params(g_uv = np.round(sample[0], 3), 
                                     n_H = np.round(sample[1], 3), 
                                     cr_zeta = np.round(sample[2], 3),
                                     output_prefix = prefix)
        print("Running 3DPDR")
        call("3DPDR", shell = True)
        
        # Moving output files to their respective directories
        print("Moving files...")
        call(f"mv {prefix}.*.fin all_runs/{prefix}/", shell = True) 
        
    print("DONE!!!!!")    
