"""
Created on 16/06/2021

@author: SURAJ SUBRAMANIAN
"""
import pickle
import os
import json
import re
import sys
import numpy as np
from pandas import read_csv as rc
import pandas as pd
import pmdarima as pm
from pmdarima.arima import ndiffs
from pmdarima.metrics import smape


def forecast_one_step():
	""" makes a single step forecast"""
	prediction, confidanceInterval = model.predict(n_periods=8, return_conf_int=True) #confidence interval can be used in future if needed
	return (prediction.tolist()[0], np.asarray(confidanceInterval).tolist()[0])

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
	spatialRegressor = jsonFile["arima_inputs"]["spatial_regressor_selection"]
	regressionDataPath = jsonFile["regression_data_path"]
	outputPath = jsonFile["arima_inputs"]["output_file_path"]
	altitudeLevel = int(jsonFile["arima_inputs"]["altitude_level"])
	testSize = int(jsonFile["arima_inputs"]["test_size"])

	data = rc(os.path.join(regressionDataPath,f"{spatialRegressor}_output.csv"), header=None)  #importing the entire input data
	dataFrame = data.iloc[:, altitudeLevel]            #selecting only the 10th altitude level for our dataframe

	## setting up the train and test sizes
	trainSize = len(dataFrame) - testSize


	## setting up train and test data
	trainData = dataFrame.iloc[:trainSize]
	testData = dataFrame.iloc[trainSize:]

	## performing tests to find the optimal values of d
	kpssDifference = ndiffs(dataFrame, alpha=0.05, test='kpss', max_d=50)
	adfDifference = ndiffs(dataFrame, alpha=0.05, test='adf', max_d=50)
	n_diffs = max(adfDifference, kpssDifference)          #selecting the max values of the two tests
	print(f"Estimated differencing term d : {n_diffs}")

	## performing auto arima without the seasonal term
	# auto arima iterates over different combinations of p and q to get the optimum value
	auto = pm.auto_arima(trainData, d = n_diffs, seasonal=False, stepwise=True,
						 suppress_warnings=True, error_action="ignore", max_p=8,
						 max_order=None, trace=True)

	print("fitting the ARIMA model is complete")
	# insample prediciton
	inSample = auto.predict_in_sample()

	# Calculating the train error based on the insample prediciton
	trainError = []
	for i in range(len(trainData)):
		error = abs(((trainData[i] - inSample[i]) / trainData[i]) * 100)
		print(f"{trainData[i]} - {inSample[i]}", trainData[i] - inSample[i], error)
		trainError.append(error)


	print("mean absolute precentage error for train data:", np.mean(trainError))
	print("maximun train error:", max(trainError))
	print("minimum train error:",min(trainError))


	## Forecasting using the trained model and then calculating the errors based on the test data

	model = auto  # seeded from the model we've already fit
	forecasts = []
	confidence_intervals = []   #can be considered later if needed

	## Updates the existing model with a single steps for future forecasts
	for new_ob in testData:
		forecast, conf = forecast_one_step()
		forecasts.append(forecast)
		confidence_intervals.append(conf)
		model.update(new_ob)

	print(f"mean absolute percentage error for test data: {smape(testData, forecasts)}")

	## converting to series for indexing and plotting purpose
	forecasts = pd.Series(list(forecasts[:]), index = testData.index)
	inSample = pd.Series(list(inSample[:]), index = trainData.index)

	testData = pd.Series(testData[:], index = testData.index)
	trainData = pd.Series(trainData[:], index = trainData.index)

	## pickling the model performance data for plotting
	pickle.dump([inSample, trainData, forecasts, testData], open(os.path.join(outputPath,"pred.pkl"), "wb"))
