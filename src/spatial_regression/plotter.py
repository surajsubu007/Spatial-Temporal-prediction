"""
Created on 16/06/2021

@author: SURAJ SUBRAMANIAN

This module takes in the outputs from the spatial regression model and plots it
"""

import matplotlib.pyplot as plt
import pickle
import re
import sys
import json
import os

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
modelSelection = jsonFile["plotter_inputs"]["regressor_output_to_plot"]
outputPath = jsonFile["plotter_inputs"]["output_path"]
savedModelPath = jsonFile["regressor_inputs"]["saved_model_path"]

plt.rcParams.update({'font.size': 15})

unique_time_data,time_counter = pickle.load(open(os.path.join(savedModelPath, f"{modelSelection}/constants.pkl"),"rb"))
all_train_mape, all_test_mape, all_train_rms, all_test_rms, all_train_var, all_test_var, all_train_r2, all_test_r2 = pickle.load(open(os.path.join(savedModelPath, f"{modelSelection}/errors.pkl"),"rb"))
y_train_pred, y_test_pred, train_error, test_error = pickle.load(open(os.path.join(savedModelPath, f"{modelSelection}/step_error.pkl"), "rb"))


## Plots the absolute percentage train error
fig = plt.figure(figsize = (15,5))
# fig = plt.figure(figsize = (7,5))
plt.plot(train_error, color = "b", label = "train error")
plt.xlabel("spatial data points")
plt.ylabel("absolute percentage error")
plt.legend()
plt.grid()
plt.savefig(os.path.join(outputPath,f"{modelSelection}/{modelSelection}_train_absolute%error_single_timestep.png"))
plt.clf()

## Plots the absolute percentage test error
fig = plt.figure(figsize = (7,5))
plt.plot(test_error, color = "r", label = "test error")
plt.xlabel("spatial data points")
plt.ylabel("absolute percentage error")
plt.legend()
plt.grid()
plt.savefig(os.path.join(outputPath,f"{modelSelection}/{modelSelection}_test_absolute%error_single_timestep.png"))
plt.clf()

## Plots the MAPE for all time steps
fig = plt.figure(figsize = (15,5))
plt.plot(all_test_mape, color = "r", label = "test")
plt.plot(all_train_mape, color = "b", label = "train")
plt.xlabel("time steps")
plt.ylabel("Mean absolute percentage error")
plt.legend()
plt.grid()
plt.savefig(os.path.join(outputPath,f"{modelSelection}/{modelSelection}_MAPE.png"))
# plt.show()


## Plots the R2 scores for all time steps
fig = plt.figure(figsize = (15,5))
plt.plot(all_test_r2, color = "r", label = "test")
plt.plot(all_train_r2, color = "b" , label = "train")
plt.xlabel("time steps")
plt.ylabel("R2 score")
plt.legend(loc = "lower right")
plt.grid()
plt.savefig(os.path.join(outputPath,f"{modelSelection}/{modelSelection}_R2_scores"))
plt.clf()
