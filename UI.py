import os
import tkinter as tk
from tkinter import filedialog, Label, Frame, Button
from PIL import Image as PilImage, ImageTk
import numpy as np
import joblib
import skimage.io
from skimage.transform import resize
import cv2
from model.Image import Image 
from sklearn.preprocessing import StandardScaler

class App:
    def __init__(self, master):
        self.master = master
        master.title("People Detection")

        self.screen_width = master.winfo_screenwidth()
        self.screen_height = master.winfo_screenheight()

        self.title = Frame(master, bg='white', height=self.screen_height * 0.1)
        self.title.pack(side=tk.TOP, fill=tk.X, expand=False)
        self.title_label = Label(self.title, text="People detection with an artificial neural network", bg='white', font=("Arial", 18))   
        self.title_label.pack(pady=20)

        self.image_frame = Frame(master, bg='white', height= self.screen_height * 0.5)  
        self.image_frame.pack(side=tk.TOP, fill=tk.X, expand=False)

        self.legend_frame = Frame(master, bg='white', height=self.screen_height * 0.2)
        self.legend_frame.pack(side=tk.TOP, padx= self.screen_width * 0.35,expand=False)

        self.control_frame = Frame(master, bg='white', height=self.screen_height * 0.2) 
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

        self.model = joblib.load('model.pkl')  # Carga el modelo entrenado

        self.images = []
        self.current_image_index = 0

        # Botones en el panel de control
        self.btn_selectImage = Button(self.control_frame, text="Select Image", command=self.open_image)
        self.btn_selectImage.config(height=4, width=40)
        self.btn_selectImage.pack(pady=40, side=tk.TOP)



    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_image(file_path)

    def process_image(self, image_path):

        image = skimage.io.imread(image_path)
        image = cv2.resize(image, (1024, 1024))

        grid_size = 128
        num_grids = 1024 // grid_size
        images = [image]
        grids = [[img[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size] for i in range(num_grids) for j in range(num_grids)] for img in images]

        feature_vectors = []
        for grid in grids[0]:
            imgObj = Image(grid, "grid_V1_4_40_P.png")
            feature_vector = np.array(imgObj.generateFeatureVector())
            feature_vectors.append(feature_vector)

        scaler = StandardScaler()
        feature_vectors = scaler.fit_transform(feature_vectors)
        predictions = self.model.predict(feature_vectors)

        image_pred = self.create_prediction_image(predictions, grid_size, num_grids)

        image_person = np.copy(image)
        
        for i in range(image_pred.shape[0]):
            for j in range(image_pred.shape[1]):
                if np.array_equal(image_pred[i, j], [76, 175, 80]):
                    pass
                else:
                    image_person[i, j] = [0, 0, 0]

        gray = cv2.cvtColor(image_person, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_person, contours, -1, (76, 175, 80), 5)

        for i in range(image_person.shape[0]):
            for j in range(image_person.shape[1]):
                if np.array_equal(image_person[i, j], [0, 0, 0]):
                    image_person[i, j] = image[i, j]

        size_display = (int(self.screen_width * 0.248), int(self.screen_width * 0.248))
        image = cv2.resize(image, (size_display))
        image_pred = cv2.resize(image_pred, (size_display))
        image_fused = cv2.addWeighted(image, 0.5, image_pred, 0.5, 0) 
        image_person = cv2.resize(image_person, (size_display))

        self.display_results(image, image_pred, image_fused, image_person)

    def create_prediction_image(self, predictions, grid_size, num_grids):
        image_pred = np.zeros((1024, 1024, 3), dtype=np.uint8)
        for i in range(num_grids):
            for j in range(num_grids):
                index = i * num_grids + j
                color = [255, 152, 0] if predictions[index] == 0 else [33, 150, 243] if predictions[index] == 1 else [76, 175, 80]
                image_pred[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size] = color
        return image_pred

    def display_results(self, image, image_pred, image_fused, image_person):
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        image = PilImage.fromarray(image)
        image = ImageTk.PhotoImage(image)
        label = Label(self.image_frame, image=image)
        label.image = image
        label.pack(side=tk.LEFT, pady=30)

        image_pred = PilImage.fromarray(image_pred)
        image_pred = ImageTk.PhotoImage(image_pred)
        label_pred = Label(self.image_frame, image=image_pred)
        label_pred.image = image_pred
        label_pred.pack(side=tk.LEFT, pady=30)

        image_fused = PilImage.fromarray(image_fused)
        image_fused = ImageTk.PhotoImage(image_fused)
        label_fused = Label(self.image_frame, image=image_fused)
        label_fused.image = image_fused
        label_fused.pack(side=tk.LEFT, pady=30)

        image_person = PilImage.fromarray(image_person)
        image_person = ImageTk.PhotoImage(image_person)
        label_person = Label(self.image_frame, image=image_person)
        label_person.image = image_person
        label_person.pack(side=tk.LEFT, pady=30)

        self.display_legend()  

    def display_legend(self):
        for widget in self.legend_frame.winfo_children():
            widget.destroy()

        A = PilImage.new('RGB', (50, 50), color=(255, 152, 0))
        A = ImageTk.PhotoImage(A)
        label_A = Label(self.legend_frame, image=A)
        label_A.image = A
        label_A.pack(side=tk.LEFT, padx=10, pady=50)
        label_A_text = Label(self.legend_frame, text="No person", bg='white')
        label_A_text.pack(side=tk.LEFT, padx=10, pady=50)

        animal = PilImage.new('RGB', (50, 50), color=(33, 150, 243))
        animal = ImageTk.PhotoImage(animal)
        label_animal = Label(self.legend_frame, image=animal)
        label_animal.image = animal
        label_animal.pack(side=tk.LEFT, padx=10, pady=50)
        label_animal_text = Label(self.legend_frame, text="Animal", bg='white')
        label_animal_text.pack(side=tk.LEFT, padx=10, pady=50)

        person = PilImage.new('RGB', (50, 50), color=(76, 175, 80))
        person = ImageTk.PhotoImage(person)
        label_person = Label(self.legend_frame, image=person)
        label_person.image = person
        label_person.pack(side=tk.LEFT, padx=10, pady=50)
        label_person_text = Label(self.legend_frame, text="Person", bg='white')
        label_person_text.pack(side=tk.LEFT, padx=10, pady=50)
        

if __name__ == '__main__':
    root = tk.Tk()
    root.state('zoomed')
    root.configure(bg='white')
    app = App(root)
    root.mainloop()
