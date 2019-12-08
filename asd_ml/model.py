import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from time import time
from tensorflow.keras.utils import plot_model
import pydoc
from ann_visualizer.visualize import ann_viz
from tensorflow.keras.models import model_from_json,Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from sklearn import model_selection

def createModel(id, trainsize, maxepoch, neuronarr):
    trainsize_f = int(trainsize) / 100
    # #Read Data
    # data = pd.read_csv("D:\Life\DEGREE\SEM 4\ISP560\PROJECT\Example\data\ASD.csv")
    # print(data.head())
    #
    # #Description
    # n_records = len(data.index)
    # n_asd_yes = len(data[data["Class/ASD"] == "YES"])
    # n_asd_no = len(data[data["Class/ASD"] == "NO"])
    # yes_percent = float(n_asd_yes / n_records) * 100
    # no_percent = float(n_asd_no / n_records) * 100
    #
    # print(f"Total Number of Records {n_records}")
    # print(f"Individuals Diagonised with ASD: {n_asd_yes}")
    # print(f"Individuals Not Diagonised with ASD: {n_asd_no}")
    # print(f"Percentage of individuals Diagonised with ASD: {yes_percent}")
    # print(f"Percentage of individuals not Diagonised with ASD: {no_percent}")

    #Prepare Data

    data = pd.read_csv("ml_web/static/file/"+str(id)+".csv", na_values=['?'])
    # print(data.head())
    # print(data.describe())

    #Clean Dataset

    # drop unwanted columns
    data = data.drop(['relation','used_app_before','ethnicity', 'age_desc','contry_of_res'], axis=1)
    # print(data.head())
    # print(data.loc[(data['result'].isnull()) | (data['age'].isnull()) |(data['gender'].isnull()) |(data['jundice'].isnull())|(data['austim'].isnull())])
    # data.dropna(inplace=True)

    # print(data.dtypes)
    asd_output = data["Class/ASD"]

    features_raw = data[['A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score',
                          'A9_Score','A10_Score','age', 'gender', 'jundice', 'austim', 'result',
                          ]]
    # print(asd_output.head())
    # print(features_raw.head())

    #One-Hot Encode (Categorical Data)
    features_final = pd.get_dummies(features_raw)
    # print(features_final)
    asd_class = asd_output.apply(lambda x: 1 if x == "YES" else 0)
    # print(f"Total Features after one-hot encoding :  {len(list(features_final.columns))}")
    # print(f"Encoded : {list(features_final.columns)}")

    #Split and Shuffle

    np.random.seed(1234)

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features_final,asd_class,train_size=trainsize_f,random_state=1)
    # print(f"X_Train : {X_train.shape}")
    # print(f"X_test : {X_test.shape}")
    # print(f"Y_train : {Y_train.shape}")
    # print(f"Y_test : {Y_test.shape}")
    # print(f"Training set has {X_train.shape[0]} samples")
    # print(f"Testing set has {X_test.shape[0]} samples")
    # print(f"X_Train : {X_train}")
    # print(f"X_test : {X_test['age']}")
    # print(f"X_test : {X_test['result']}")
    # print(f"X_test : {X_test['gender_f']}")
    # print(f"X_test : {X_test['age']}")
    # # print(f"Y_train : {Y_train}")
    # # print(f"Y_test : {Y_test}")

    #Models


    np.random.seed(42)

    model = Sequential()

    # print(neuronarr[0])
    model.add(Dense(int(neuronarr[0]), input_dim=18, kernel_initializer='normal', activation='relu'))
    # print(x)
    for x in neuronarr[1:]:
        model.add(Dense(int(x), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()


    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    #Running and evaluation the model
    hist = model.fit(X_train, Y_train,
              batch_size=16,
              epochs=int(maxepoch),
              validation_data=(X_test, Y_test),
              verbose=2)

    # # Evaluating the model on the training and testing set
    # score = model.evaluate(X_train, Y_train)
    # print("\n Training Accuracy:", score)
    #
    # score = model.evaluate(X_test, Y_test, verbose=0)
    # print("\n Testing accuracy: ", score)

    # #-----MODEL TO SAVE------
    model_json = model.to_json()
    with open("ml_web/static/file/"+str(id)+".json", "w") as json_file:
        json_file.write(model_json)
    #WEIGHTS
    model.save_weights("ml_web/static/file/"+str(id)+".h5")

def loadModel(id):
    #----MODEL TO LOAD--------
    json_file = open("ml_web/static/file/"+str(id)+".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #WEIGHTS
    loaded_model.load_weights("ml_web/static/file/"+str(id)+".h5")




#Generate Image Neural Network
# extra step to allow graphviz to be found
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
# # C:\Program Files (x86)\Graphviz2.38\bin
# # C:/Users/Msi/Miniconda3/envs/tensorflow/Library/bin/graphviz
# ann_viz(model, title="My first neural network")
# plot_model(model,to_file="model.png")
# generate classification report using predictions for categorical model
# from sklearn.metrics import classification_report, accuracy_score
# print(X_test)
# newData = pd.read_csv("D:\Life\DEGREE\SEM 4\ISP560\PROJECT\Example\input_data.csv")
# print(newData.shape)
# predictions = model.predict_classes(newData)
# # pd.DataFrame(X_test).to_csv("test_data.csv", index = None)
# print(predictions)
# score = model.evaluate(newData, predictions, verbose=0)
# print("\n Testing accuracy: ", score)
