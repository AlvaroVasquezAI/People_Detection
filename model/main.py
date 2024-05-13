import os
import cv2
import joblib
import skimage
import numpy as np
import pandas as pd
from Image import Image
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate


# Define constants for data paths and model file
DATA_PATH_V1 = "dataset/V1/output"
DATA_PATH_V2 = "dataset/V2/output"
MODEL_FILE = 'model.pkl'

FINAL_HYPERPARAMETERS = {
    'hidden_layer_sizes': (128, 128, 128),
    'max_iter': 100,
    'alpha': 0.01,
    'solver': 'adam',
    'verbose': False,
    'random_state': 42,
    'learning_rate_init': .001,
    'tol': 1e-4
}

def load_images_with_classes(path):
    data = []
    for class_name in os.listdir(path):
        if class_name != "Noise" and not class_name.endswith((".csv", ".DS_Store")):
            for image_name in os.listdir(os.path.join(path, class_name)):
                image = skimage.io.imread(os.path.join(path, class_name, image_name))
                img_obj = Image(image, image_name)
                data.append((img_obj, class_name))
    return data

def prepare_dataset(data_paths):
    data = []
    for path in data_paths:
        data.extend(load_images_with_classes(path))

    # Filter out some instances of class A to address class imbalance
    data = [(img, cls) for img, cls in data if cls != "A" or np.random.rand() < 0.33]

    print("Calculating feature vectors...")
    feature_vectors, classes = zip(*data)
    feature_vectors = [np.array(img.generateFeatureVector()) for img in feature_vectors]

    df = pd.DataFrame({'FeatureVector': feature_vectors, 'Class': classes})

    X = df['FeatureVector'].values
    X = [np.array(x) for x in X]
    y = df['Class'].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(df['Class'].value_counts())

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, le


def train_model(X_train, y_train, use_grid_search=False):
    mlp = MLPClassifier(**FINAL_HYPERPARAMETERS)

    print(mlp)
    
    if use_grid_search:
        param_grid = {
            #Se prueba con muchas arquitecturas, diferentes, desde muy simples hasta muy complejas desde 2 neruonas hasta 128
            'hidden_layer_sizes':  [ (2, 2, 2), (4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64), (128, 128)],
            'max_iter': [100, 200, 300],
            'alpha': [0.1, 0.01, 0.001],
            'solver': ['sgd', 'adam'],
            'learning_rate_init': [0.1, 0.01, 0.001],
            'tol': [1e-8, 1e-6, 1e-4]
        }
        grid_search = GridSearchCV(mlp, param_grid, n_jobs=-1, cv=3)
        grid_search.fit(X_train, y_train)
        print("Best parameters:", grid_search.best_params_)
        return grid_search.best_estimator_
    else:
        mlp.fit(X_train, y_train)
        return mlp


def evaluate_model(model, X_test, y_test, le):
    """Evaluates the model using classification report and confusion matrix."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.inverse_transform([0, 1, 2])))
    print(confusion_matrix(y_test, y_pred))


def perform_cross_validation(model, X, y, cv=5):
    """Performs cross-validation and prints the average accuracy. Returns the best model."""
    results = cross_validate(model, X, y, cv=cv, return_train_score=True)
    print("Cross-validation results:")
    print("Average test accuracy:", np.mean(results['test_score']))
    print("Average train accuracy:", np.mean(results['train_score']))
    
    return model 


def predict_and_visualize(model, image_path, grid_size=128):
    """Predicts on a given image and visualizes the results."""
    #Se carga una imagen completa del conjunto de datos 
    image = skimage.io.imread(image_path)
    image = cv2.resize(image, (1024, 1024))


    grid_size = 128
    num_grids = 1024 // grid_size
    images = [image]
    grids = [[img[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size] for i in range(num_grids) for j in range(num_grids)] for img in images]




    #Se extraen las características de cada grid
    feature_vectors = []
    for grid in grids[0]:
        imgObj = Image(grid, "grid_V1_4_40_P.png")
        feature_vector = np.array(imgObj.generateFeatureVector())
        feature_vectors.append(feature_vector)

    #Se normalizan los datos
    scaler = StandardScaler()
    feature_vectors = scaler.fit_transform(feature_vectors)

    #Se realiza la predicción de cada grid
    predictions = model.predict(feature_vectors)

    #Se crea una imagen con las predicciones
    image_pred = np.zeros((1024,1024,3), dtype=np.uint8)

    for i in range(num_grids):
        for j in range(num_grids):
            grid_index = i*num_grids + j
            if predictions[grid_index] == 0:
                image_pred[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size] = [255,0,0] 
            elif predictions[grid_index] == 1:
                image_pred[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size] = [0,255,0]
            elif predictions[grid_index] == 2:
                image_pred[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size] = [0,0,255]
            else:
                image_pred[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size] = [128, 128, 128]


    #Se muestra la imagen original y la imagen con las predicciones
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.imshow(image)
    plt.title("Original Image")
    plt.subplot(132)
    plt.imshow(image_pred)
    plt.title("Predicted Image")
    image_fused = cv2.addWeighted(image, 0.5, image_pred, 0.5, 0)
    plt.subplot(133)
    plt.imshow(image_fused)
    plt.title("Fused Image")
    plt.show()



if __name__ == "__main__":
    cross_val = False
    if not os.path.exists(MODEL_FILE):
        X_train, X_test, y_train, y_test, le = prepare_dataset([DATA_PATH_V1, DATA_PATH_V2])

        if cross_val:
            model = MLPClassifier(**FINAL_HYPERPARAMETERS)

            perform_cross_validation(model, np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)))
        
        model = train_model(X_train, y_train, use_grid_search=False)

        evaluate_model(model, X_test, y_test, le)
        joblib.dump(model, MODEL_FILE)
    else:
        model = joblib.load(MODEL_FILE)
        # ... (use the model for prediction and visualization)


    predict_and_visualize(model, "dataset/V1/1.png")

    # Perform cross-validation and print the average accuracy
    # cross_val = True