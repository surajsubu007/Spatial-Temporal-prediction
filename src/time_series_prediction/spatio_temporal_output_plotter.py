"""
Created on 16/06/2021

@author: SURAJ SUBRAMANIAN

This module plots the final spatial temporal prediction model output
"""
import pickle
import matplotlib.pyplot as plt
import os
import json
import re
import sys
import pandas as pd
import numpy as np

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
modelSelection = jsonFile["plotter_inputs"]["model_selection"]
outputPath = jsonFile["spatio_temporal_plotter"]["output_path"]
dataFramePath = jsonFile["dataframe_path"]
modelPath = jsonFile["lstm_gru_inputs"]["saved_model_path"]
timeStep = int(jsonFile["spatio_temporal_plotter"]["altitude_time_step"])
altRangeLower = float(jsonFile["spatio_temporal_plotter"]["altitude_range"]["lower"])
altRangeUpper = float(jsonFile["spatio_temporal_plotter"]["altitude_range"]["upper"])


dataFrame = pd.read_csv(os.path.join(dataFramePath,"dataframe.csv"))
convertedData = dataFrame.to_numpy()

plt.rcParams.update({'font.size': 15}) # increase the font size
latitude = []
longitude = []
altitude = []
rh = []

## Arranging the input data for the altitude profile comparison
for i in range(len(convertedData)):

	if convertedData[i, 0] == 8682.0 and convertedData[i, 1] == convertedData[2, 1] and convertedData[i, 2] == convertedData[2, 2]:
		altitude.append(convertedData[i, 3])
		rh.append(convertedData[i, 4])

#################################

## Retriving the prrdicted output and arranging it based on one particular time instance
parameterAltitudePredicted = []
parameterAltitude = []
for i in range(100):
	features_set, features_set_test, labels, labels_test, scaler = pickle.load(open(os.path.join(modelPath,f"{modelSelection}/{modelSelection}_constants_{i}.pkl"), "rb"))
	pred1, train_error, pred2, test_error = pickle.load(open(os.path.join(modelPath,f"{modelSelection}/{modelSelection}_preds_{i}.pkl"), "rb"))

	parameterAltitudePredicted.append(scaler.inverse_transform(pred2)[timeStep]) # here 324 is the time step taken from the total of 500 time steps in the test data

	parameterAltitude.append(scaler.inverse_transform(labels_test)[timeStep])


altitudeRange = np.linspace(altRangeLower, altRangeUpper, 100) #the range of altitudes as per the input data
plt.figure(figsize=(30,5))
plt.scatter(altitude,rh, label = "actual relative humidity values")
plt.plot(altitudeRange, parameterAltitude, label =" regressed relative humidity values", color ="r")
plt.xlabel("Altitude")
plt.ylabel("Relative humidity %")
plt.ylim(-5,105)
plt.legend()
plt.grid()
plt.savefig(os.path.join(outputPath,"output.png"))
# plt.show()
