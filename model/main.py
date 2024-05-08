#Librerias del MultiLayer Perceptron

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import skimage
import os

from Image import Image


#Cargamos el dataset

'''
Hay 2 carpetas principales V1 y V2, dentro de cada una de ellas existe output, que es otra carpeta en la que se encuentras las imágenes de cada clase

Dentro de output hay 3 carpetas "A" para ausencia de personas, "N" para animales y "P" para personas

Cada imagen tiene un nombre que sigue el siguiente formato: "grid_VX_numImagen_numGrid_clase.png"

Por ejemplo: grid_V1_1_1_P.png, grid_V1_1_2_P.png, etc.
'''

path_V1 = "dataset/V1/output" #Dentro de esa carpeta se encuentran las imágenes de cada clase, guardadas en carpetas "A", "N" y "P"
path_V2 = "dataset/V2/output" 

#Cargamos las imágenes de cada clase (Se guardan junto con las clases para tener un mejor manejo de los datos)
#, se cargan las imágenes como Image de nuestra implementación:

def loadImagesWithClasses(path):
    data = []
    for class_name in os.listdir(path):
        if class_name != "Noise":
            #Que la extensión no sea csv o .DS_Store
            if not class_name.endswith(".csv") and not class_name.endswith(".DS_Store"):
                for image_name in os.listdir(os.path.join(path, class_name)):
                    image = skimage.io.imread(os.path.join(path, class_name, image_name))
                    imgObj = Image(image, image_name)

                    instancia = [imgObj, class_name]

                    data.append(instancia)

    return data

data_V1 = loadImagesWithClasses(path_V1)
data_V2 = loadImagesWithClasses(path_V2)

#Se concatenan los datasets para que esten de la forma vector, clase 
data = data_V1 + data_V2

feature_vectors = []
classes = []
for instance in data:
    img, class_name = instance
    feature_vector = np.array(img.generateFeatureVector())
    feature_vectors.append(feature_vector)
    classes.append(class_name)

df = pd.DataFrame({'FeatureVector': feature_vectors, 'Class': classes})

#Se guardan los datos en un archivo csv
df.to_csv('dataset.csv', index=False)




X = df['FeatureVector'].values
X = [np.array(x) for x in X]



y = df['Class'].values

#Imprimir el número de instancias por clase
print(df['Class'].value_counts())


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#Se muestran las etiquetas decodificadas para efectos de visualización
print(le.inverse_transform([0, 1, 2]))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
print(X_train)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


mlp = MLPClassifier(hidden_layer_sizes=(16, 32, 64, 32, ), max_iter=200, alpha=0.01,
                    solver='lbfgs', verbose=10, random_state=42, learning_rate_init=.1)

mlp.fit(X_train, y_train)

print(classification_report(y_test, mlp.predict(X_test)))
print(confusion_matrix(y_test, mlp.predict(X_test)))

