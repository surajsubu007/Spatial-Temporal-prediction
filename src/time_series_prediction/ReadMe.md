# Time series part of the Spatial temporal prediciton model

---------
### Dependencies
* python 3.8
* tensorflow 2.2.0-rc2
* keras 2.4.3
### How to use the control file
* The control file is located in the input folder of the root.
* The paths and the important selection needed to run the 4 python files in the src is controlled by this control file.
   * regression_data_path : specifies the location of the regression output
   * dataframe_path : specifies the path to the dataframe which was generated with the nc file.
   * spatial_regressor_selection : selects the spatial regessor output to be used for the time series predicion model. The inputs to this can be GBR, SVR, RFR, XGBR.
   * altitude_level : selects the altitude level for time series predtion (for arima only, as the lstms and grus will give outputs for all altitude levels)
   * test_size : sets the size of the test data
   * output_file_path : defines the path for the output and the saved models for arima
   * sequence_length : specifies the sequence length to be uses for LSTM and GRU
   * epochs : specifies the number of epochs to be run for LSTM and GRU
   * saved_model_path : specifies the path to save the model for LSTM and GRU
   * spatio_temporal_plotter : these are the set of attributes used to get the final output. Here the time series model input can be either GRU or LSTM.
   
### How to use the codes
1. Run the lstm_gru.py script for deep learning based time series prediction
    * Execute the script along with the path to the control file.
    * Sample code to run the script "python lstm_gru.py <path to the control file>/control_file.json"  
    * The Paths and selections for the script are taken from the control file.
    * The trained model and other values required by the plotter.py will be saved in the Execution folder
2. Run the arima.py script for arima based time series.
   * [note: the arima model script is written only for one altitude level just to give a comparison]
   * Execute the script from terminal with the command in the same fashion as for lstm_gru.py
   * The trained model and other values required by the plotter.py will be saved in the EXECUTION/trained_model folder

3. Run the plotter.py script for the plots on the time series outputs
   * Run the script in the similar fashion as before
   * The output plot will be saved in the EXECUTION/output/plot folder.

4. Run the spatio_temporal_output_plotter.py script in the similar fashion to get the final output plot. Note that this script will only run if the required time series or the spatial regressor model has completely implemented
   * The process to run the script is same as before.
   
#### pylint scores
1. arima.py = 9.79
2. lstm_gru.py = 8.42
   *  Module lstm_gru
C:\Users\suraj\Desktop\zeus_projects\STP\time_series_prediction\src\lstm_gru.py:14:0: E0611: No name 'models' in module 'LazyLoader' (no-name-in-module)
C:\Users\suraj\Desktop\zeus_projects\STP\time_series_prediction\src\lstm_gru.py:60:19: E1121: Too many positional arguments for method call (too-many-function-args)
C:\Users\suraj\Desktop\zeus_projects\STP\time_series_prediction\src\lstm_gru.py:68:20: E1121: Too many positional arguments for method call (too-many-function-args)
   * in first error pylint isnt recognizing the model from tensorflow
   * in the next error pylint considers reshapes third positional argument to be wrong
3. plotter.py = 10
4. spatio_temporal_output_plotter = 10
