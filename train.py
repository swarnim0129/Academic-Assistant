from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from tkinter import messagebox
import os
import cv2

class Train:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1710x1150+0+0")
        self.root.title("Training Panel")

        title_lb1 = Label(self.root, text="Welcome to Training Window", font=("verdana", 38, "bold"), bg="black", fg="white")
        title_lb1.place(x=0, y=0, width=1710, height=58)

        img3 = Image.open("./college_images/trainBGBG.png")
        img3 = img3.resize((1710, 1050))
        self.photoimg3 = ImageTk.PhotoImage(img3)
        bg_img = Label(self.root, image=self.photoimg3)
        bg_img.place(x=0, y=58, width=1710, height=1050)

        b1_1 = Button(bg_img, text="Train Images", cursor="hand2", command=self.train_classifier,
                      font=("veranda", 45, "bold"), bg="white", fg='black')
        b1_1.place(x=660, y=400, width=420, height=62)

    def train_classifier(self):
        data_dir = "data"
        path = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if
                file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

        faces = []
        ids = []

        progress_frame = Frame(self.root, width=800, height=30)
        progress_frame.place(relx=0.5, rely=0.5, anchor=CENTER)

        self.progress_bar = ttk.Progressbar(progress_frame, orient=HORIZONTAL, length=800, mode='determinate')
        self.progress_bar.pack(fill='both', expand=True,pady=30)

        total_images = len(path)
        self.current_idx = 0

        self.train_images(path, faces, ids, total_images)

    def train_images(self, path, faces, ids, total_images):
        if self.current_idx < total_images:
            image_path = path[self.current_idx]
            img = Image.open(image_path).convert('L')  # gray scale
            image_np = np.array(img, 'uint8')
            img_id = int(os.path.split(image_path)[1].split('.')[1])

            faces.append(image_np)
            ids.append(img_id)

            # Update progress bar
            progress_value = int((self.current_idx + 1) / total_images * 100)
            self.progress_bar['value'] = progress_value
            self.root.update_idletasks()

            # Schedule the next update affter processing the current image
            self.root.after(1, self.train_images, path, faces, ids, total_images)
            self.current_idx += 1
        else:
            self.finish_training(faces, ids)
            self.progress_bar.destroy()  # Remove progress bar after training

    def finish_training(self, faces, ids):
        ids = np.array(ids)

        # Train the classifier
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces, ids)
        clf.write("clifi.xml")

        cv2.destroyAllWindows()
        messagebox.showinfo("Result", "Training dataset completed!!", parent=self.root)


if __name__ == "__main__":
    root = Tk()
    obj = Train(root)
    root.mainloop()
