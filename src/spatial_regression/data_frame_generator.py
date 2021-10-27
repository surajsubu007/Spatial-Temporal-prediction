"""
Created on 16/06/2021

@author: SURAJ SUBRAMANIAN

This module takes in the netCDF files, cleans the data and creates a data frame in csv format
"""
import math
import re
import sys
import json
import os
from datetime import datetime
from time import mktime
import pandas as pd
from sklearn.impute import SimpleImputer
from netCDF4 import Dataset
import numpy as np

## takes in the path of the json file
delimiters = r"\\", "/","\\", "//"
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

## takes in the paths from the json file
with open(opDir, "r") as controller:
	jsonFile = json.load(controller)
	ncFilePath = jsonFile["data_frame_generator"]["nc_file_path"]
	outputPath = jsonFile["data_frame_generator"]["output_file_path"]


## specifying the time origin
timeData = Dataset(ncFilePath)
timeOrigin = int(timeData.variables["time"][0])

print(timeOrigin)
timeOrigin = datetime.utcfromtimestamp(timeOrigin)
print(timeOrigin)

timeOrigin = mktime(timeOrigin.timetuple())
print(timeOrigin)


data = Dataset(ncFilePath)
time = data.variables["time"][:]
lon = data.variables["longitude"][:]
lat = data.variables["latitude"][:]
plevel = data.variables["plevel"][:]
rh = data.variables["RH_prl"]

index = rh.shape[0] * len(plevel) * rh.shape[2] * rh.shape[3]
dataArray = np.empty([index, 5])

counter = 0

pSeaLevelValue = 1013.25

for m, t in enumerate(time):
	for n, p in enumerate(plevel):
		for i, la in enumerate(lat):
			for j, lo in enumerate(lon):
				l = datetime.utcfromtimestamp(int(t))
				time_stamp = mktime(l.timetuple())
				time_stamp = (time_stamp - timeOrigin) / 3600
				dataArray[counter, 0] = time_stamp
				dataArray[counter, 3] = ((1 - math.pow((p / pSeaLevelValue), 0.190284)) * 145366.45) * 0.3048
				dataArray[counter, 1] = la
				dataArray[counter, 2] = lo
				dataArray[counter, 4] = rh[m][n][i][j]
				counter += 1

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
dataArray = imputer.fit_transform(dataArray)

## putting into a pandas data frame and saving it
dataFrame = pd.DataFrame(dataArray)
dataFrame.to_csv(os.path.join(outputPath,"dataframe.csv"), index = False)
