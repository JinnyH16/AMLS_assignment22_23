import os
from tqdm import tqdm
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import cv2
from timeit import default_timer as timer
from sklearn.preprocessing import StandardScaler



def prepare_cartoon_data(images_dir, labels_path, filename_index=3, label_index=1, img_size=50, train=True):  # Remove images with glasses
    file = open(labels_path, 'r')
    lines = file.readlines()
    image_label = {line.split('\t')[filename_index].rstrip(): int(line.split('\t')[label_index]) for line in
                   lines[1:]}
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)] # get image paths

    if train:
        print('start find image without glasses in {}....'.format(images_dir))
        without_glasses = []
        # find images without glasses
        for i in tqdm(range(len(image_paths))):
            # extract area of glasses
            image_path = image_paths[i]
            img = cv2.imread(image_path)
            x, y, w, h = 240, 180, 50, 50
            img = img[x:x + w, y:y + h]
            x, y, w, h = 16, 26, 20, 20
            # convert RGB to HSV
            hsv = cv2.cvtColor(img[x:x + w, y:y + h], cv2.COLOR_RGB2HSV)
            mean = hsv[:, :, 2].mean()  # to determine if the image contains glasses
            if mean > 50:
                without_glasses.append(image_path)
        print('total image without glasses: {} total:{} rate:{}'.format(len(without_glasses), len(image_paths),
                                                                        len(without_glasses) / len(image_paths)))
        image_paths = without_glasses

    all_imgs = []
    all_labels = []
    print('start extracting feature of images in ', images_dir)
    i = 0
    for image_path in tqdm(image_paths):
        i = i + 1
        img = cv2.imread(image_path)
        img = cv2.resize(src=img, dsize=(img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
        img_name = image_path.split('/')[-1]

        all_imgs.append(img)
        all_labels.append(image_label[img_name])
    total_len = len(all_imgs)
    all_imgs = np.array(all_imgs).reshape((total_len, -1))
    all_labels = np.array(all_labels)
    return all_imgs, all_labels



def B2_decisiontree():
    start = timer()

    x_train, y_train = prepare_cartoon_data('./Datasets/cartoon_set/img/', './Datasets/cartoon_set/labels.csv')
    x_test, y_test = prepare_cartoon_data('./Datasets/cartoon_set_test/img/', './Datasets/cartoon_set_test/labels.csv')

    # apply standard scalar
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # Importing the Decision tree classifier from the sklearn library.
    tree_params = {
        'criterion': 'entropy'
    }
    clf = tree.DecisionTreeClassifier(**tree_params)

    # Training the decision tree classifier on training set.
    clf.fit(x_train, y_train)

    # Predicting labels on the test set.
    y_pred = clf.predict(x_test)
    
    end = timer()
    print('Computation cost: ', end - start)
    
    # Use accuracy metric from sklearn.metrics library
    print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(x_train)))
    print('Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))
    print('Confusion matrix: ', '\n', confusion_matrix(y_test, y_pred))
    print('Classification report: ', '\n',
          classification_report(y_test, y_pred))
    
    
