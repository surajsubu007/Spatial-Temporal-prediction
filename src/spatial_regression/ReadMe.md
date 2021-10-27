# Spatial regression part of spaito-temporal prediction
_____________________
### Dependencies
* sklearn 0.24.1
* xgboost 1.4.2

### How to use the control file
* The control file path needs to be given while executing each script
* This file contains all the paths and selections required for the execution of the scripts
   * nc_file_path : specifies the path of the nc file inside the folder structure
   * output_file_path and output_path : specifies the path for outputs to be saved
   * regressor_used : Specifies the regressor to be used to run the regressor.py code. Here, the user can choose from GBR, RFR, SVR, XGBR.
   * hyperprameter_tuning_timestep : time step at which hyper parameter tuning is to be done. Here the user can choose a time step from the training set
   * output_comparison_timestep : time step at which the outputs are noted for plotting and comparison purposes.
   * saved_model_path : This specifies the path where the trained model and other parameters of the model is saved.
   * regressor_output_to_plot : This specifies the regressor model whose output is to be plotted. The model is to be selected from the list of regressors only.
    
------------------------
### How to use the codes
1. Execute the data_frame_generator.py script first to generate the imputed dataset from the input ncfile
   * To execute this script, after the python file name add the path to the control file which is located in the input folder.
   * For instance the sample execution code will be "python data_frame_generator.py <path to the control file>/control_file.json"
   * This data is stored in the execution folder inside the output subfolder as dataframe.csv
2. Execute the regressor.py script to perform spatial regression on this dataframe
    * Execute the script in the similar fashion as the data_frame_generator script.
    * The output and the trained models are saved in the Execution folder.
    * One has to run the script again for a different regressor.
3. Run the plotter.py to generate the output plots of the various regressor
    * The execution style is same as before 
    * The plots are saved in the Execution/output/plots folder
    
-----------------------------
    
#### pylint scores
1. data_frame_generator.py = 8.81
   
   ************* Module data_frame_generator
data_frame_generator.py:13:0: E0611: No name 'Dataset' in module 'netCDF4' (no-name-in-module)
   this is the only issue 

2. regressor.py = 9.65
3. plotter.py = 9.72
