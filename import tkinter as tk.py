import cv2
import numpy as np
from keras.models import load_model
from tkinter import Tk, Button, filedialog, Label
from PIL import Image, ImageTk
import cvzone

np.set_printoptions(suppress=True)

model = load_model("C:/Users/thi_s/Downloads/archive/chest_xray/keras_Model.h5", compile=False)


class_names = ['pneumonia', 'normal']


def perform_prediction(image_path):
    
    img = cv2.imread(image_path)
    image = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

   
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

   
    texto1 = f"Class: {class_name}"
    texto2 = f"Confidence Score: {str(np.round(confidence_score * 100))[:-2]} %"
    label1.config(text=texto1)
    label2.config(text=texto2)

   
    cvzone.putTextRect(img, texto1, (50, 50), scale=4)
    cvzone.putTextRect(img, texto2, (50, 100), scale=4)

    altura, largura = img.shape[:2]
    img = cv2.resize(img, (700, 500))
    cv2.imshow('IMG', img)
    cv2.waitKey(0)


def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        perform_prediction(file_path)


root = Tk()
root.title("Pneumonia Detection")


label1 = Label(root, text="Class: ", font=("Arial", 14))
label1.pack()

label2 = Label(root, text="Confidence Score: ", font=("Arial", 14))
label2.pack()


button = Button(root, text="Choose Image", command=open_file_dialog, font=("Arial", 14))
button.pack(pady=10)


root.mainloop()
