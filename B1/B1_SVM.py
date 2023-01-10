import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import cv2
from PIL import Image
from timeit import default_timer as timer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC



# Resize and load image
def loadImage(path):
    img = Image.open(path)
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    median = cv2.resize(img, (50, 50))
    img_resize = Image.fromarray(cv2.cvtColor(median,cv2.COLOR_BGR2RGB))
    img = img_resize.convert("L")
    data = img.getdata()
    #data = np.array(data).reshape(img.size[1], img.size[0])/100
    return data


# code for grid search
'''
grid = {'C': [0.2, 0.5, 1, 5, 10]}
search = GridSearchCV(svc, grid,
                        scoring='accuracy',
                        cv=5,
                        )
search.fit(x_train, y_train)
model = search.best_estimator_
y_pred=model.predict(x_test)
print("SVM test Accuracy:", accuracy_score(y_test, y_pred))

print(search.best_params_)
'''

def B1_SVM():

    # Load training data
    dataset = pd.read_csv('./Datasets/cartoon_set/labels.csv', sep="\t")  # read csv file
    y_train = dataset['face_shape']
    img_name1 = dataset['file_name']

    x_train = []
    for name in img_name1:
        img_path = './Datasets/cartoon_set/img/' + name  # get path based on image name
        img = loadImage(img_path)
        x_train.append(img)  # add pic to x_train

    # Load test data
    data_test = pd.read_csv('./Datasets/cartoon_set_test/labels.csv', sep="\t")  # read csv file
    y_test = data_test['face_shape']
    img_name2 = data_test['file_name']

    x_test = []
    for name in img_name2:
        img_path = './Datasets/cartoon_set_test/img/' + name  # get path based on image name
        img = loadImage(img_path)
        x_test.append(img)  # add pic to x_train
        
    # apply standard scaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    svc = SVC(kernel='rbf', C=10)  # these parameters are obtained by grid search

    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    
    print("SVM train Accuracy:", accuracy_score(y_train, y_pred=svc.predict(x_train)))
    print("SVM test Accuracy:", accuracy_score(y_test, y_pred=y_pred))
    print('Confusion matrix: ', '\n', confusion_matrix(y_test, y_pred))
    print('Classification report: ', '\n',
          classification_report(y_test, y_pred))
