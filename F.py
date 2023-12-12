import json
from urllib.parse import urlparse, urljoin, quote
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pocsuite3.api import Output, POCBase, POC_CATEGORY, register_poc, requests, VUL_TYPE
from pocsuite3.lib.core.interpreter_option import OptDict
from pocsuite3.modules.listener import REVERSE_PAYLOAD
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tensorflow as tf


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        return x

def create_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


    def _options(self):
        o = OptDict()
        payload = {
            "nc": REVERSE_PAYLOAD.NC,
            "bash": REVERSE_PAYLOAD.BASH,
            "powershell": REVERSE_PAYLOAD.POWERSHELL,
        }
        o["command"] = OptDict(selected="bash", default=payload)
        o["ml_task"] = OptDict(selected="image_classification", default={
            "selected": "image_classification",
            "image_path": "path/to/image.jpg"
        })
        return o

    def _verify(self):
        result = {}
        ml_task = self.get_option("ml_task")["selected"]
        if ml_task == "image_classification":
            image_path = self.get_option("ml_task")["image_path"]
            prediction = self.classify_image(image_path)
            result['VerifyInfo'] = {'Image Classification Prediction': prediction}
        else:
           
            p = self._check(self.url)
            if p:
                result['VerifyInfo'] = {
                    'URL': p[0],
                    'Command executed successfully': p[1]
                }
        return self.parse_output(result)

    def classify_image(self, image_path):
       
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        transform = transforms.Compose([transforms.Resize((28, 28)),
                                        transforms.ToTensor()])
        image_tensor = transform(image).unsqueeze(0)
        model = SimpleModel()
        output = model(image_tensor)
        _, prediction = torch.max(output, 1)
        pytorch_result = prediction.item()

       
        keras_model = create_keras_model()
        keras_model.load_weights('path/to/keras_model_weights.h5')  # Adjust the path
        keras_result = keras_model.predict(image_tensor.view(1, -1))
        keras_result = tf.argmax(keras_result, axis=1).numpy().item()

        return f'PyTorch: {pytorch_result}, Keras: {keras_result}'


register_poc(WeaverOASqlInjectionWithML)
