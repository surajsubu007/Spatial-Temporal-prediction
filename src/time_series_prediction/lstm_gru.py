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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

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
	spatialRegressorSelection = jsonFile["lstm_gru_inputs"]["spatial_regressor_selection"]
	modelSelection = jsonFile["lstm_gru_inputs"]["timeseries_model"]
	regressionDataPath = jsonFile["regression_data_path"]
	outputPath = jsonFile["lstm_gru_inputs"]["saved_model_path"]
	altitudeInput = int(jsonFile["lstm_gru_inputs"]["altitude_level"])
	seqLenInput = int(jsonFile["lstm_gru_inputs"]["sequence_length"])
	testSize = int(jsonFile["lstm_gru_inputs"]["test_size"])
	epochs = int(jsonFile["lstm_gru_inputs"]["epochs"])


	data = rc(os.path.join(regressionDataPath,f"{spatialRegressorSelection}_output.csv"), header=None)
	data_frame = data.iloc[:,1:]
	seq_len = seqLenInput

	## preparing train and test data in the form of supervised learning problem
	scaler = MinMaxScaler(feature_range=(1, 2))
	for i in range(100):

		train = len(data_frame) - testSize
		test = len(data_frame) - train

		trainData = np.array(data_frame.iloc[:train, i])
		trainData = trainData.reshape(-1, 1)
		testData = np.array(data_frame.iloc[train - seq_len:, i])
		testData = testData.reshape(-1, 1)

		print(testData.shape)

		trainDataNormalized = scaler.fit_transform(trainData)
		testDataNormalized = scaler.transform(testData)

		trainFeatureSet = []
		trainLables = []

		for j in range(seq_len, train):
			trainFeatureSet.append(trainDataNormalized[j - seq_len:j, 0])
			trainLables.append(trainDataNormalized[j, 0])

		testFeatureSet = []
		testLables = []

		for j in range(seq_len, test+seq_len):
			testFeatureSet.append(testDataNormalized[j - seq_len:j, 0])
			testLables.append(testDataNormalized[j, 0])


		testFeatureSet = np.array(testFeatureSet)
		testFeatureSet = testFeatureSet.reshape(testFeatureSet.shape[0], testFeatureSet.shape[1], 1)
		testLables = np.array(testLables)
		testLables = testLables.reshape(testLables.shape[0], 1)



		trainFeatureSet = np.array(trainFeatureSet)
		trainLables = np.array(trainLables)

		trainFeatureSet = trainFeatureSet.reshape(trainFeatureSet.shape[0], trainFeatureSet.shape[1], 1)
		trainLables = trainLables.reshape(trainLables.shape[0], 1)

		print(testFeatureSet.shape, testLables.shape)
		pickle.dump([trainFeatureSet, testFeatureSet, trainLables, testLables, scaler],
					open(os.path.join(outputPath,f"{modelSelection}/{modelSelection}_constants_{i}.pkl"), 'wb'))


		## Using the features and labels type of data for GRU or LSTM to train the model for every altitude level
		# using a sequential model
		model = Sequential()
		if modelSelection == "GRU":
			model.add(GRU(units=20, return_sequences=True, input_shape=(trainFeatureSet.shape[1], 1))) #using hidden state 20 seemed to be optimum
			model.add(Dropout(0.2))
			model.add(GRU(units=20, return_sequences=False))
		if modelSelection == "LSTM":
			model.add(LSTM(units=20, return_sequences=True, input_shape=(trainFeatureSet.shape[1], 1)))
			model.add(Dropout(0.2))
			model.add(LSTM(units=20, return_sequences=False))

		model.add(Dropout(0.2))  # drop out is used to avoid overfitting
		model.add(Dense(units=1)) # dense layer takes the output of the hidden state and brings it to the required shape of the univariate time series

		model.compile(optimizer='adam', loss='mean_squared_error')
		model.fit(trainFeatureSet, trainLables, epochs=epochs)

		model.save(os.path.join(outputPath,f"{modelSelection}/{modelSelection}_model_{i}_.h5"))


	## using the trained model for predictions for every altitude level
	for i in range(100):
		trainFeatureSet, testFeatureSet, trainLables, testLables, scaler = pickle.load(
			open(os.path.join(outputPath,f"{modelSelection}/{modelSelection}_constants_{i}.pkl"), "rb"))
		trainedModel = load_model(os.path.join(outputPath,f"{modelSelection}/{modelSelection}_model_{i}_.h5"))

		pred1 = trainedModel.predict(trainFeatureSet)
		train_error = (np.divide((np.abs(np.array(trainLables) - np.array(pred1))), np.array(trainLables))) * 100

		pred2 = trainedModel.predict(testFeatureSet)
		test_error = (np.divide((np.abs(np.array(testLables) - np.array(pred2))), np.array(testLables))) * 100

		pickle.dump([pred1, train_error, pred2, test_error], open(os.path.join(outputPath,f"{modelSelection}/{modelSelection}_preds_{i}.pkl"), "wb"))

		if i==altitudeInput:
			import pandas as pd
			pred2 = scaler.inverse_transform(pred2)
			pred2 = pred2.flatten()
			testLables = scaler.inverse_transform(testLables)
			testLables = testLables.flatten()
			pred2 = pd.Series(pred2, index=data_frame.index[-500:])
			testLables = pd.Series(testLables, index=data_frame.index[-500:])

			pred1 = scaler.inverse_transform(pred1)
			pred1 = pred1.flatten()
			pred1 = pd.Series(list(pred1), index=data_frame.index[seq_len:-500])
			print(pred1)
			print(pred2)


			trainLables = scaler.inverse_transform(trainLables)
			trainLables = trainLables.flatten()
			trainLables = pd.Series(list(trainLables), index=data_frame.index[seq_len:-500])

			print(trainLables)
			print(testLables)
			print(f"test R2 score:{r2_score(pred2, testLables)}")
			print(f"train R2 score:{r2_score(pred1, trainLables)}")
			print(f"test MAPE:{np.mean(test_error)}")
			print(f"train MAPE:{np.mean(train_error)}")
			pickle.dump([pred1, trainLables, pred2, testLables], open(os.path.join(outputPath,f"{modelSelection}/pred.pkl"), "wb"))
