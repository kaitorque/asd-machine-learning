import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from time import time
#from keras.utils import plot_model

from ann_visualizer.visualize import ann_viz
import tensorflow as tf
from sklearn import model_selection

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

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
    data.dropna(inplace=True)

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

    model = tf.keras.Sequential()

    # print(neuronarr[0])
    model.add(tf.keras.layers.Dense(int(neuronarr[0]), input_dim=18, kernel_initializer='normal', activation='relu'))
    # print(x)
    for x in neuronarr[1:]:
        model.add(tf.keras.layers.Dense(int(x), kernel_initializer='normal', activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.summary()


    adam = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    #Running and evaluation the model
    hist = model.fit(X_train, Y_train,
              batch_size=16,
              epochs=int(maxepoch),
              validation_data=(X_test, Y_test),
              verbose=2)

    # # Evaluating the model on the training and testing set
    score1 = model.evaluate(X_train, Y_train)
    print("\n Training Accuracy:", score1)

    score2 = model.evaluate(X_test, Y_test, verbose=0)
    print("\n Testing accuracy: ", score2)

    # #-----MODEL TO SAVE------
    model_json = model.to_json()
    with open("ml_web/static/file/"+str(id)+".json", "w") as json_file:
        json_file.write(model_json)
    #WEIGHTS
    model.save_weights("ml_web/static/file/"+str(id)+".h5")

    # #Generate Model
    # import os
    # os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
    # # # C:\Program Files (x86)\Graphviz2.38\bin
    # # C:/Users/Msi/Miniconda3/envs/tensorflow/Library/bin/graphviz
    # ann_viz(model, title="My first neural network",view=True)
    return {"training": score1[1], "testing": score2[1]}

def loadModel(id, csv_file):
    #----MODEL TO LOAD--------
    json_file = open("ml_web/static/file/"+str(id)+".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    #WEIGHTS
    loaded_model.load_weights("ml_web/static/file/"+str(id)+".h5")

    newData = pd.read_csv(csv_file)
    # print(newData.shape)
    predictions = loaded_model.predict_classes(newData)
    # # pd.DataFrame(X_test).to_csv("test_data.csv", index = None)
    print(predictions)
    return predictions.tolist()

def loadModel2(id, input):
    #----MODEL TO LOAD--------
    json_file = open("ml_web/static/file/"+str(id)+".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    #WEIGHTS
    loaded_model.load_weights("ml_web/static/file/"+str(id)+".h5")
    print(input["q1"])
    result = int(input["q1"]) + int(input["q2"]) + int(input["q3"]) + int(input["q4"]) + int(input["q5"]) + int(input["q6"]) + int(input["q7"]) + int(input["q8"]) + int(input["q9"]) + int(input["q10"])

    if input["gender"] == 1:
        gFem = 0
        gMal = 1
    else:
        gFem = 1
        gMal = 0

    if input["jaundice"] == 1:
        gJaunYes = 1
        gJaunNo  = 0
    else:
        gJaunYes = 0
        gJaunNo = 1

    if input["autism"] == 1:
        gautismy = 1
        gautismn  = 0
    else:
        gautismy = 0
        gautismn = 1

    newData = {"1": [input["q1"]],
               "2" : [input["q2"]],
               "3" : [input["q3"]],
               "4" : [input["q4"]],
               "5" : [input["q5"]],
               "6" : [input["q6"]],
               "7" : [input["q7"]],
               "8" : [input["q8"]],
               "9" : [input["q9"]],
               "10" : [input["q10"]],
               "age" : [input["age"]],
               "result" : [result],
               "gender_f" : [gFem],
               "gender_m" : [gMal],
               "jaundice_n" : [gJaunNo],
               "jaundine_y" : [gJaunYes],
               "autism_no" : [gautismn],
               "autism_yes" : [gautismy]}
    # print(newData.shape)
    newData = pd.DataFrame(newData)
    predictions = loaded_model.predict_classes(newData)
    # # pd.DataFrame(X_test).to_csv("test_data.csv", index = None)
    print(predictions)
    return predictions.tolist()


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
#
# score = model.evaluate(newData, predictions, verbose=0)
# print("\n Testing accuracy: ", score)
