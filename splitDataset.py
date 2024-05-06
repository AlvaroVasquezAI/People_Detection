import os
import cv2
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import csv

def create_grid_interface(image_dir, output_dir, version):
    window = Tk()
    window.title("Image Grid Labeling Tool")
    window.state("zoomed")
    window.config(bg="#212121")

    title = Label(window, text="Image Grid Labeling Tool", font=("Arial", 24), bg="#212121", fg="white")
    title.pack(pady=40)

    grid_size = 128
    num_grids = 1024 // grid_size
    current_grid = 0
    current_image_index = 0
    class_label = StringVar()
    class_label.set("Select the class")

    images = [cv2.cvtColor(cv2.imread(os.path.join(image_dir, f)), cv2.COLOR_BGR2RGB)
              for f in sorted(os.listdir(image_dir)) if f.endswith('.png')]
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    grids = [[img[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size] for i in range(num_grids) for j in range(num_grids)] for img in images]

    # Prepare folders
    for cls in ['P', 'A', 'N', 'Noise']:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

     # CSV file setup
    csv_path = os.path.join(output_dir, f"{version}.csv")
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["NumberOfImage", "NumberOfGrid", "Class", "TypeOfFile"])

    colors = {
        "P": (76, 175, 80),   
        "A": (255, 152, 0), 
        "N": (33, 150, 243),    
        "Noise": (128, 128, 128) 
    }

    def update_label(choice):
        class_label.set(f"Class {choice} selected")
        grid_name = f"grid_{version}_{current_image_index + 1}_{current_grid + 1}_{choice}.png"
        grid_label.config(text=grid_name)
        display_image(current_image_index, choice)  

    def save_and_next():
        nonlocal current_grid, current_image_index
        if class_label.get() == "Select the class":
            messagebox.showerror("Error", "Select a class please")
            return
        choice = class_label.get().split()[-2]

        grid = grids[current_image_index][current_grid]
        grid_name = f"grid_{version}_{current_image_index + 1}_{current_grid + 1}_{choice}.png"
        cv2.imwrite(os.path.join(output_dir, choice, grid_name), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

        # Write to CSV
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_image_index + 1, current_grid + 1, choice, ".png"])

        print(f"Grid {current_grid + 1} of image {current_image_index + 1} saved as {choice}")
        
        class_label.set("Select the class")
        current_grid += 1
        if current_grid < len(grids[current_image_index]):
            display_grid(current_grid)
        else:
            current_grid = 0
            current_image_index += 1
            if current_image_index < len(images):
                display_image(current_image_index)
            else:
                window.destroy()

    def display_image(index, selected_class=None):
        image = images[index].copy()
        grid_length = 1024 // num_grids
        row = current_grid // num_grids
        col = current_grid % num_grids
        start_x = col * grid_length
        start_y = row * grid_length
        color = colors.get(selected_class if selected_class else class_label.get().split()[-1], (255, 255, 255))
        cv2.rectangle(image, (start_x, start_y), (start_x + grid_length, start_y + grid_length), color, 5)
        image = Image.fromarray(image)
        image = image.resize((384, 384), Image.NEAREST)
        photo = ImageTk.PhotoImage(image)
        full_image_display.config(image=photo)
        full_image_display.image = photo
        full_image_label.config(text=image_files[index])

    def display_grid(index):
        grid_image = Image.fromarray(grids[current_image_index][index])
        grid_image = grid_image.resize((256, 256), Image.NEAREST)
        grid_photo = ImageTk.PhotoImage(grid_image)
        grid_display.config(image=grid_photo)
        grid_display.image = grid_photo
        grid_label.config(text=f"grid_{version}_{current_image_index + 1}_{index + 1}")
        display_image(current_image_index)

    def key_handler(event):
        if event.char == '1':
            update_label("P")
        elif event.char == '2':
            update_label("A")
        elif event.char == '3':
            update_label("N")
        elif event.char == '4':
            update_label("Noise")
        elif event.char == '0':  
            update_label("Noise")
        elif event.keysym == 'Return':
            save_and_next()

    window.bind('<Key>', key_handler)

    main_frame = Frame(window, bg="#212121")
    main_frame.pack(side="left", fill="both", expand=True, padx=50)

    right_frame = Frame(window, bg="#212121")
    right_frame.pack(side="right", padx=100, fill="both", expand=True)

    full_image_display = Label(right_frame, bg="#212121")
    full_image_display.pack(pady=20)

    full_image_label = Label(right_frame, text="", font=("Arial", 18), bg="#212121", fg="white")
    full_image_label.pack(pady=10)

    grid_display = Label(main_frame, bg="#212121")
    grid_display.pack(pady=20)

    grid_label = Label(main_frame, text="", font=("Arial", 18), bg="#212121", fg="white")
    grid_label.pack(pady=10)

    info_label = Label(main_frame, textvariable=class_label, font=("Arial", 18), bg="#212121", fg="white")
    info_label.pack(pady=10)

    button_frame = Frame(main_frame, bg="#212121")
    button_frame.pack(pady=20)

    P_button = Button(button_frame, text="Person (P)", command=lambda: update_label("P"), height=2, width=20, bg="#4CAF50")
    P_button.pack(side="left", padx=20, expand=True)

    A_button = Button(button_frame, text="Absent (A)", command=lambda: update_label("A"), height=2, width=20, bg="#FF9800")
    A_button.pack(side="left", padx=20, expand=True)

    N_button = Button(button_frame, text="Animal (N)", command=lambda: update_label("N"), height=2, width=20, bg="#2196F3")
    N_button.pack(side="left", padx=20, expand=True)

    Noise_button = Button(button_frame, text="Noise", command=lambda: update_label("Noise"), height=2, width=20, bg="#808080")
    Noise_button.pack(side="left", padx=20, expand=True)

    next_button = Button(main_frame, text="Next", command=save_and_next, height=2, width=20, bg="#F44336")
    next_button.pack(pady=20)

    display_grid(current_grid)
    window.mainloop()

if __name__ == '__main__':
    image_directory = 'dataset/V2'
    output_directory = 'dataset/V2/output'
    version = 'V2'
    create_grid_interface(image_directory, output_directory, version)
