import cv2
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl


if(not os.environ.get('PYTHONHTTPSVERIFY','') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


x,y = fetch_openml('letter', version=1, return_X_y=True)
print(pd.Series(y).value_counts())

classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','W','X','Y','Z']
n_classes = len(classes)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=9, train_size=16000, test_size=4000)

x_train_scaled = x_train/255
x_test_scaled = x_test/255

clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(x_train_scaled, y_train)
y_pred = clf.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :- ", accuracy)

i = "C:/Users/DELL/Downloads/Python/ModelViewController/al.png"

def get_pred(image):

    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)

    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
    pixel_filter = 20

    min_pixel = np.percentile(image_bw_resized, pixel_filter)

    image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted - min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized_inverted)

    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel

    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(49,16)
    test_pred = clf.predict(test_sample)

    return(test_pred[0])