##############################################
	    	DCC-Tree
##############################################

DCC-Tree is a method for exploring the posterior distribution of Bayesian decision trees that combines the work of the Divide Conquer Combine algorithm (Zhou et al. (2020)) with Hamiltonian Monte Carlo sampling. 


##############################################
	Requirements/Installation
##############################################

DCC-Tree is implemented in Python. Use the following commands to create the required environment to run the method.

- ensure you are in the correct directory
- run ./scripts/setup.sh via terminal/command prompt

Manual changes have been made to the NumPyro package when running setup.sh, which need to be copied across correctly to run the method.

##############################################
		  Usage
##############################################
Once requirements have been installed the DCC-Tree method can be run. Use the following commands:

Activate environment from the correct folder (the dcc-env must be active to correctly use DCC-Tree) using:

source activate ./envs/dcc-env

Example usage:

python ./src/dcc-tree/DCC-Tree.py --dataset toy-class --out_dir ./results/ --init_id 1 --h_init 0.1 --h_final 0.001 --num_chains 16 --num_samps 100 --num_warmup 1000 --T0 100 --T 200 --save 1 --alpha 0.95 --beta 1.0 --M 10 
python ./src/dcc-tree/DCC-Tree.py --dataset toy-non-sym --out_dir ./results/ --init_id 1 --tag 1 --h_init 0.1 --h_final 0.001 --num_chains 16 --num_samps 100 --num_warmup 1000 --T0 100 --T 500 --save 1 --alpha 0.95 --beta 1.0 --M 10

For further help and information on arguments:

python ./src/dcc-tree/DCC-Tree.py -h 

All methods (both final runs and cross-validation) were run using shell scripts located in the "scripts" directory. Uncomment the relevant lines corresponding to the desired method script to run.


##############################################
	    Training/Evaluation
##############################################

Running the DCC-Tree.py file will run the training. Tree information/statistics are saved and output in a pickle file in the specified output directory (changed using the --out_dir argument).  Performance metrics can be computed from this using the convert.py file. This will convert the output results to a matlab data file.

There are several synthetic testing datasets that can be selected via the --dataset argument, available options are listed in the load_data method in utils.py. More information about formatting datasets into the appropriate structure can be found in ./datasets/README.txt


##############################################
		 Results
##############################################

Results from the paper were generated via CPU cores, either using Intel Cascade Lake CPU's (OS: Rocky Linux release 8.9 (Green Obsidian)) via a HPC cluster or using a MacBook Pro (16-inch, 2021) with an Apple M1 Pro processor. Note that the raw output results are too large to be included for submission (~10-20GB due to number of samples requested). However, these have been converted into matlab data files (in results folder) which display performance metrics.

If Matlab is installed, figures and tabular results from the paper can be reproduced by running plot_results.m. Converted results for all methods are included under the matlab subdirectory. Uncomment the dataset of interest on lines 6-11 and run in the plot_results.m file and run to produce results relating to that dataset.

Note that, although attempts have been made to ensure the exact reproducibility of the results (i.e. random seeds set), at the current moment this has not yet been achieved.

