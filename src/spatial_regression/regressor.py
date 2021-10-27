"""
Created on 16/06/2021

@author: SURAJ SUBRAMANIAN
"""

import statistics
import pickle
import time
import re
import fnmatch
import sys
import json
import os
import subprocess
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

def regressor(xInput, yInput, regressorModel, parameters):
	"""The function takes in the regressor model selected and gives the predictions"""

	xTrain, xTest, yTrain, yTest = train_test_split(xInput, yInput, test_size=0.25, random_state=0, shuffle=True)  #shuffle splitting the data for test and train

	regressorModel.set_params(**parameters)  # setting the parameters from gridsearchcv
	regressorModel.fit(xTrain, yTrain) #fitting the model to the selected regressor

	## Predicting the values for train and test data
	yTrainPred = regressorModel.predict(xTrain)
	yTestPred = regressorModel.predict(xTest)

	## Calculating insample and outsample absolute percentage errors
	trainError = []
	for i in range(len(yTrain)):
		error = (abs(yTrain[i] - yTrainPred[i]) / yTrain[i]) * 100
		trainError.append(error)
	trainError = np.array(trainError)

	testError = []
	for i in range(len(yTest)):
		error = (abs(yTest[i] - yTestPred[i]) / yTest[i]) * 100
		testError.append(error)
	testError = np.array(testError)

	# calculating insample and outsample MAPE
	trainMape = np.mean(trainError)
	testMape = np.mean(testError)

	# calculating rms errors
	trainRms = np.sqrt(mean_squared_error(yTrain, yTrainPred))
	testRms = np.sqrt(mean_squared_error(yTest, yTestPred))

	# r2 score for regresion
	trainR2 = r2_score(yTrain, yTrainPred)
	testR2 = r2_score(yTest, yTestPred)

	# calculating variance in error
	trainVariance = statistics.variance(yTrain - yTrainPred)
	testVariance = statistics.variance(yTest - yTestPred)

	return regressorModel, yTrainPred, yTestPred, trainError, testError, trainMape, testMape, trainRms, testRms, trainVariance, testVariance, trainR2, testR2


def grid_search(xInput, yInput, regressorModel, modelName):
	"""uses grid search to crossvalidate and finds the optimum hyperparameter"""

	## Hyperparameter tuning inputs for the regressor models
	if modelName == "GBR":
		tuningParameters = [{
			'max_depth': [2, 3, 4, 5, 10],
			'min_samples_leaf': [1, 2, 3, 4],
			'min_samples_split': [2, 3, 4],
			'n_estimators': [600, 1000, 1500, 2000]}]

	elif modelName == "RFR":
		tuningParameters = [{'bootstrap': [True, False],
							 'max_depth': [5, 10, 20, 3],
							 'max_features': ["auto"],
							 'min_samples_leaf': [2, 3],
							 'min_samples_split': [4, 5, 6],
							 'n_estimators': [600, 700, 750]}]

	elif modelName == "XGBR":
		tuningParameters = [{
			'max_depth': [5, 10, 15, 20, 30],
			'n_estimators': [5, 10, 20, 40, 60, 70]}]

	elif modelName == "SVR":
		tuningParameters = [{'kernel': ['rbf'], "epsilon": [0.001, 0.1, 0.01],
							 'C': [ 1,10,100,300,500,800,1000,2000,3000], "gamma":[0.1,1,0.01,10,20]}]

	tuningGrid = GridSearchCV(regressorModel, tuningParameters, cv=10, return_train_score=True, verbose=5) #hyperparameter tuning using 10 fold cv
	bestParameters = tuningGrid.fit(xInput, yInput)
	bestParameters = bestParameters.best_params_
	return bestParameters


if __name__ == "__main__":

	delimiters = r"\\", "/", "\\", "//"
	regexPattern = '|'.join(map(re.escape, delimiters))
	opArg = re.split(regexPattern, sys.argv[1])
	opDict = {}
	opDir = ""
	opCount = 0
	for e in opArg:
		opCount += 1
		opDict["key{}".format(opCount)] = e
		if e == opArg[-1]:
			opDir = opDir + opDict["key{}".format(opCount)]
		else:
			opDir = opDir + opDict["key{}".format(opCount)] + os.path.sep

	opDir = os.path.abspath(opDir)

	with open(opDir, "r") as controller:
		jsonFile = json.load(controller)
	modelSelection = jsonFile["regressor_inputs"]["regressor_used"]
	tuningTimeStep = jsonFile["regressor_inputs"]["hyperparameter_tuning_timestep"]
	outputTimeStep = jsonFile["regressor_inputs"]["output_comparison_timestep"]
	outputPath = jsonFile["regressor_inputs"]["output_path"]
	savedModelPath = jsonFile["regressor_inputs"]["saved_model_path"]


	dataFrame = pd.read_csv(os.path.join(outputPath,"dataframe.csv")) # inputting the dataframe created by the dataframe_generator.py code
	dataArray = dataFrame.to_numpy()

	## setting up empty lists and constants
	E_in = []	#for storing mean absolute percentage errors
	E_out = []

	unique_time_data = []
	error_data = [] #for storing all the different types of error

	timeCounter = 0 #to keep track of the time step


	## model initialization
	if modelSelection == "SVR":
		model = svm.SVR()
	elif modelSelection == "RFR":
		model = RandomForestRegressor()
	elif modelSelection == "GBR":
		model = GradientBoostingRegressor()
	elif modelSelection == "XGBR":
		model = XGBRegressor()

	unique_time, indices = np.unique(dataArray[:, 0], return_index=True) #storing each time step value
	unique_time_data.append(unique_time)

	## iterating through each time step and tuning the hyperparameter for the selected time step
	for i in range(len(indices)):

		#storing each time step data in a temp variable batch_data
		if i + 1 == len(indices):
			batch_data = dataArray[indices[i]:]
		else:
			batch_data = dataArray[indices[i]: indices[i + 1]]

		xyz = batch_data[:, 1:4]
		S = xyz		#the variations in latitude, longitude and altitude
		t = batch_data[:, 4].reshape(-1, 1) #the atmospheric parameter values

		#initializing the scaler transforms for both
		sc_S = MinMaxScaler(feature_range=(1, 2))
		sc_t = MinMaxScaler(feature_range=(1, 2))

		S2 = sc_S.fit_transform(S)
		t2 = np.ravel(sc_t.fit_transform(t))

		#getting the values of the tuned hyperparameter for the selected time step data
		if i == int(tuningTimeStep):
			best_params = grid_search(S2, t2, model, modelSelection)

			print("parameters", best_params)
			print("unique time", unique_time[i])

			break
	print("finished with the hyperparameter tuning")

	##initializing empty lists to store the performance metric values for all the time steps
	all_train_mape = []
	all_test_mape = []
	all_train_rms = []
	all_test_rms = []
	all_train_var = []
	all_test_var = []
	all_train_r2 = []
	all_test_r2 = []

	start_time = time.time()  #to account for the training time

	## performing spatial regression for each time step data
	print("training the spatial models")
	for i in range(len(indices)):
		#     print(unique_time[i])
		if i + 1 == len(indices):
			batch_data = dataArray[indices[i]:]
		else:
			batch_data = dataArray[indices[i]: indices[i + 1]]

		xyz = batch_data[:, 1:4]
		S = xyz

		t = batch_data[:, 4].reshape(-1, 1)
		sc_S = MinMaxScaler(feature_range=(1, 2))
		sc_t = MinMaxScaler(feature_range=(1, 2))
		#         sc_S = StandardScaler()
		#         sc_t = StandardScaler()
		S2 = sc_S.fit_transform(S)
		t2 = np.ravel(sc_t.fit_transform(t))
		model, y_train_pred, y_test_pred, train_error, test_error, train_mape, test_mape, train_rms, test_rms, train_var, test_var, train_r2, test_r2 = regressor(
			S2, t2, model, best_params)
		all_train_mape.append(train_mape)
		all_test_mape.append(test_mape)
		all_train_rms.append(train_rms)
		all_test_rms.append(test_rms)
		all_train_var.append(train_var)
		all_test_var.append(test_var)
		all_train_r2.append(train_r2)
		all_test_r2.append(test_r2)

		# storing the error and score values for the required output time step
		if i == int(outputTimeStep):
			error_data = [y_train_pred, y_test_pred, train_error, test_error, train_mape, test_mape, train_rms, test_rms, train_var, test_var, train_r2, test_r2]
			print("unique time", unique_time[i])

		# pickling all the required data for further use
		file_name = os.path.join(savedModelPath, f"{modelSelection}", f"{modelSelection}_model_{timeCounter}.pkl")
		pickle.dump(model, open(file_name, "wb"))
		pickle.dump(sc_S, open(os.path.join(savedModelPath, f"{modelSelection}/sc_S_{timeCounter}.pkl"), "wb"))
		pickle.dump(sc_t, open(os.path.join(savedModelPath, f"{modelSelection}/sc_t_{timeCounter}.pkl"), "wb"))
		pickle.dump(batch_data, open(os.path.join(savedModelPath, f"{modelSelection}/batch_data_{timeCounter}.pkl"), "wb"))

		timeCounter += 1

	print("time required to train the mdoel = ", time.time() - start_time)

	## getting all the data back for the required output time step for comparison
	y_train_pred, y_test_pred, train_error, test_error, train_mape, test_mape, train_rms, test_rms, train_var, test_var, train_r2, test_r2 = error_data[0], error_data[1], error_data[2], error_data[3], error_data[4], error_data[5], error_data[6], error_data[7], error_data[8], error_data[9], error_data[10], error_data[11]

	print(f"train and test R2 scores:{train_r2, test_r2}")
	print(f"train and test MAPE:{train_mape, test_mape}")

	## pickling output data for each time step
	pickle.dump([unique_time_data, timeCounter], open(os.path.join(savedModelPath, f"{modelSelection}/constants.pkl"), "wb"))
	pickle.dump([all_train_mape, all_test_mape, all_train_rms, all_test_rms, all_train_var, all_test_var, all_train_r2, all_test_r2], open(os.path.join(savedModelPath, f"{modelSelection}/errors.pkl"), "wb"))
	pickle.dump([y_train_pred, y_test_pred, train_error, test_error], open(os.path.join(savedModelPath, f"{modelSelection}/step_error.pkl"), "wb"))

	## Prediction using the trained model
	print("initializing prediction")
	start_time = time.time() # to get the prediction time
	# xi = input("lat_value in between 17.15 - 17.6")
	# yi = input("lon_value in between 78.14 - 78.69")
	unique_time_data = np.array(unique_time_data)
	unique_time_data = unique_time_data.flatten()
	output_array = np.empty([timeCounter, 101])

	## iterating over each time step to get the predicted altitude profile data for that time step
	for i in range(0, timeCounter):
		file = os.path.join(savedModelPath, f"{modelSelection}/{modelSelection}_model_{i}.pkl")
		model = pickle.load(open(file, "rb"))
		mm_S = pickle.load(open(os.path.join(savedModelPath, f"{modelSelection}/sc_S_{i}.pkl"), "rb"))
		mm_t = pickle.load(open(os.path.join(savedModelPath, f"{modelSelection}/sc_t_{i}.pkl"), "rb"))
		batch_data = pickle.load(open(os.path.join(savedModelPath, f"{modelSelection}/batch_data_{i}.pkl"), "rb"))
		xi = batch_data[2, 1]
		yi = batch_data[2, 2]
		z = batch_data[:, 3]
		rh_check = batch_data[:, 4]
		zi = np.linspace(min(z), max(z), 100)
		xyz_p = []
		for j in range(len(zi)):
			xyz_p.append([xi, yi, zi[j]])

		S = np.array(xyz_p)
		S = mm_S.transform(S)

		cp = model.predict(S)
		cp = cp.reshape(-1, 1)
		cp = mm_t.inverse_transform(cp)

		# Storing the regression output data which will be fed to the time series model
		for op in range(len(cp) + 1):
			if op == 0:
				output_array[i, op] = unique_time_data[i]
			else:
				if cp[op - 1] < min(rh_check):
					output_array[i, op] = min(rh_check)
				elif cp[op - 1] > max(rh_check):
					output_array[i, op] = max(rh_check)
				else:
					output_array[i, op] = cp[op - 1]

	print("time requried for prediction = ", time.time() - start_time)

	np.savetxt(os.path.join(outputPath, f"{modelSelection}_output.csv"), output_array, delimiter=",")
