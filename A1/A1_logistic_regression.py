import pandas as pd
import cv2
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve
from PIL import Image
import numpy as np
from timeit import default_timer as timer
from sklearn.preprocessing import StandardScaler


# resize and load image to array
def loadImage(path):
    img = Image.open(path)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    median = cv2.resize(img, (45, 55))
    img_resize = Image.fromarray(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
    img = img_resize.convert("L")
    data = img.getdata()
    # data = np.array(data).reshape(img.size[1], img.size[0])/100
    return data


# logistic regression model
# sigmoid function
def sigmoid(z):
    sig = 1. / (1. + np.exp(-z))
    return sig


# obtain logistic regression parameter
def logRegParamEstimates(xTrain, yTrain):
    intercept = np.ones((np.array(xTrain).shape[0], 1))
    xTrain = np.concatenate((intercept, xTrain), axis=1)
    theta = np.zeros(np.array(xTrain).shape[1])
    for i in range(2200):  # maximum iteration
        z = np.dot(xTrain, theta)
        h = sigmoid(z)
        lr = 0.01
        gradient = np.dot(xTrain.T, (h - yTrain)) / (yTrain.shape[0] * np.log(2))
        theta -= lr * gradient

    return theta


# fit the model and make prediction
def logRegrNEWRegrPredict(xTrain, yTrain, xTest):
    theta = logRegParamEstimates(xTrain, yTrain)
    intercept = np.ones((np.array(xTest).shape[0], 1))
    xTest = np.concatenate((intercept, xTest), axis=1)
    sig = sigmoid(np.dot(xTest, theta))
    y_pred1 = sig >= 0.5  # true or false assignment
    return y_pred1


def getK(y):  # intermediate function
    x = 0
    k = np.ones((np.array(y).shape[0]))
    for i in y:
        if i == True:
            k[x] = 1
        elif i == False:
            k[x] = 0
        x = x + 1
    return k


def getMSE(k, y):  # Mean square error

    p = 0
    for j in range(len(k)):
        a = np.square(k[j] - y[j])
        p = p + a

    MSE = p / len(k)

    return MSE


def A1_logisticregression():
    start = timer()
    
    # Load training data
    dataset = pd.read_csv('./Datasets/celeba/labels.csv', sep="\t")  # read csv file
    dataset.loc[dataset['gender'] == -1, 'gender'] = 0  # convert -1 to 0
    y_train = dataset['gender']
    img_name1 = dataset['img_name']
    x_train = []
    for name in img_name1:
        img_path = './Datasets/celeba/img/' + name  # get path based on image name
        img = loadImage(img_path)
        x_train.append(img)  # add pic to x_train

    # Load test data
    data_test = pd.read_csv('./Datasets/celeba_test/labels.csv',
                            sep="\t")  # read csv file
    data_test.loc[data_test['gender'] == -1, 'gender'] = 0
    y_test = data_test['gender']
    img_name2 = data_test['img_name']

    x_test = []
    for name in img_name2:
        img_path = './Datasets/celeba_test/img/' + name  # get path based on image name
        img = loadImage(img_path)
        x_test.append(img)  # add pic to x_train

    # apply standard scalar to x_train and x_test
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    y_pred1 = logRegrNEWRegrPredict(x_train, y_train, x_test)  # train the logistic regression model and test it
    k_test = getK(y_pred1)
    y_pred2 = logRegrNEWRegrPredict(x_train, y_train, x_train)  
    k_train = getK(y_pred2)

    end = timer()
    print('Computation time: ', end - start)

    # obtain accuracy metric
    print('Accuracy on train set: ' + str(accuracy_score(y_train, k_train)))
    print('Accuracy on test set: ' + str(accuracy_score(y_test, k_test)))
    print('Confusion matrix: ', '\n', confusion_matrix(y_test, k_test))
    print('MSE: ' + str(getMSE(k_test, y_test)))
    print('Classification report: ', '\n',
          classification_report(y_test, k_test))  # text report showing the main classification metrics
