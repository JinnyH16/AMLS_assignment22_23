import pandas as pd
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image
import numpy as np
from timeit import default_timer as timer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# resize and load image 
def loadImage(path):
    img = Image.open(path)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    median = cv2.resize(img, (45, 55))
    img_resize = Image.fromarray(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
    img = img_resize.convert("L")
    data = img.getdata()
    # data = np.array(data).reshape(img.size[1], img.size[0])/100
    return data

# code for grid search
'''
grid = {'C': [0.2, 0.5, 1, 5, 10], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
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

def A2_SVM():
    start = timer()

    # Load training data
    dataset = pd.read_csv('./Datasets/celeba/labels.csv', sep="\t")  # read csv file
    dataset.loc[dataset['smiling'] == -1, 'smiling'] = 0
    y_train = dataset['smiling']
    img_name1 = dataset['img_name']

    x_train = []
    for name in img_name1:
        img_path = './Datasets/celeba/img/' + name  # get path based on image name
        img = loadImage(img_path)
        x_train.append(img)  # add pic to x_train


    # Load test data
    data_test = pd.read_csv('./Datasets/celeba_test/labels.csv',
                            sep="\t")  # read csv file
    data_test.loc[data_test['smiling'] == -1, 'smiling'] = 0
    y_test = data_test['smiling']
    img_name2 = data_test['img_name']

    x_test = []
    for name in img_name2:
        img_path = './Datasets/celeba_test/img/' + name  # get path based on image name
        img = loadImage(img_path)
        x_test.append(img)  # add pic to x_train

    # apply standard scalar
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # apply SVM
    svc = SVC(kernel='rbf', C=1)

    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    print("SVM train Accuracy:", accuracy_score(y_train, y_pred=svc.predict(x_train)))
    print("SVM test Accuracy:", accuracy_score(y_test, y_pred))
    print('Confusion matrix: ', '\n', confusion_matrix(y_test, y_pred))
    print('Classification report: ', '\n',
          classification_report(y_test, y_pred))
