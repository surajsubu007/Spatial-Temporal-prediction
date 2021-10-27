"""
Created on 16/06/2021

@author: SURAJ SUBRAMANIAN

This module takes in the outputs from the time series model and plots it
"""

import pickle
import matplotlib.pyplot as plt
import os
import json
import re
import sys

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
outputPath = jsonFile["plotter_inputs"]["output_path"]

## selects the model chosen by the user
if modelSelection == "GRU":
	pred1, labels, pred2, labels_test = pickle.load(open("../../../output/spatial_temporal_prediction/time_series_prediction/trained_model/GRU/pred.pkl","rb"))

elif modelSelection == "LSTM":
	pred1, labels, pred2, labels_test = pickle.load(open("../input/trained_model_generated/LSTM/pred.pkl","rb"))

elif modelSelection == "ARIMA":
	pred1, labels, pred2, labels_test = pickle.load(open("../input/trained_model_generated/ARIMA/pred.pkl","rb"))

plt.figure(figsize = (30,5))
plt.plot(labels_test[:100], color = "black",linestyle='dashed', label = "test data")
plt.plot(pred2[:100], color = "red", label = "out of sample prediction")

plt.plot(labels[-100:], color ="blue",linestyle='dashed', label = "train_data")
plt.plot(pred1[-100:], color = "green", label = "in sample prediction")
plt.ylim(0,100)
plt.legend()
plt.xlabel("Time steps")
plt.ylabel("Relative humidity %")
plt.grid()
plt.savefig(os.path.join(outputPath,f"{modelSelection}/{modelSelection}_output.png"))

