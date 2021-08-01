import skimage.io
import random
import os
import sys

ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)  # To find local version of the library


from Food_Mask_RCNN.samples.food_mask import food
import mrcnn.model as modellib
from mrcnn import utils
from mrcnn import visualize
from mrcnn.model import log
#from flask_ngrok import run_with_ngrok
from flask import Flask
import numpy as np
import io
import json
from flask import Flask, jsonify, request
from PIL import Image
import time
import cv2
from tensorflow.python.keras.backend import set_session
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

ROOT_DIR = os.path.join(ROOT_DIR, "Food_Mask_RCNN")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
config = food.FoodConfig()
FOOD_DIR = os.path.join(ROOT_DIR, "datasets/food")
dataset_val = food.FoodDataset()
dataset_val.load_food(FOOD_DIR,"val")
dataset_val.prepare()
#%cd /content/Mask_RCNN
#!mkdir logs
#%cd logs
# !wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

    #uncomment below code to use the pre-trained weights
#!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ovmfOESjJkpYvPwL4KNnat4i9HA9ftqp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ovmfOESjJkpYvPwL4KNnat4i9HA9ftqp" -O model_044.h5 && rm -rf /tmp/cookies.txt

sess = tf.Session()
graph = tf.get_default_graph()
# initialize our Flask application and the Keras model
app = Flask(__name__)
model = None

def load_model():
    global model
    global sess
    set_session(sess)

    class InferenceConfig(food.FoodConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
    model_path= os.path.join(ROOT_DIR, "logs/model_044.h5")
    model.load_weights(model_path, by_name=True)
    model.keras_model._make_predict_function()

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
    
        while img_bytes == None :
           time.sleep(1)
        stream = io.BytesIO(img_bytes)
        imageFile = Image.open(stream)
        #imageFile = imageFile.save("geeks.jpg")
        image = np.array(imageFile)
        #image = skimage.io.imread("/content/Mask_RCNN/datasets/food/val/20151127_115951.jpg")
        if image.ndim != 3:
          image = skimage.color.gray2rgb(image)
        if image.shape[-1] == 4:
          image = image[..., :3]
        masked_plate_pixels=1130972
        real_plate_size=12
        real_plate_area=113.04
        pixels_per_inch_sq=masked_plate_pixels/real_plate_area
        calories=[]
        items=[]
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            results = model.detect([image], verbose=0)
        r = results[0]
        for i in range(r['masks'].shape[-1]):
            masked_food_pixels=r['masks'][:,:,i].sum()
            class_name=dataset_val.class_names[r['class_ids'][i]]
            real_food_area=masked_food_pixels/pixels_per_inch_sq
            calorie=food.get_calorie(class_name,real_food_area)
            calories.append(int(calorie))
            items.append(class_name)
            print (int(calorie) , "  ", class_name)
        return jsonify({"calorie": calories, "class_name": items})


if __name__ == "__main__":
    load_model()
    #run_with_ngrok(app)   #starts ngrok when the app is run
    app.run()