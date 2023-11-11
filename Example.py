import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

filepath = 'C:/Users/kavey/OneDrive/Desktop/plant and medicine/Plant-Leaf-Disease-Prediction-main/model.h5'
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")

tomato_plant = cv2.imread('C:/Users/kavey/OneDrive/Desktop/plant and medicine/Plant-Leaf-Disease-Prediction-main/Dataset/test/Syzygium Jambos (Rose Apple).jpg')
test_image = cv2.resize(tomato_plant, (128,128)) # load image 
  
test_image = img_to_array(test_image)/255 # convert image to np array and normalize
test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
result = model.predict(test_image) # predict diseased palnt or not

print(result)
  
pred = np.argmax(result, axis=1)
print(pred)
if pred==0:
    print( "Amaranthus Viridis (Arive-Dantu)")
       
elif pred==1:
    print("Azadirachta Indica (Neem)")
        
elif pred==2:
    print("Citrus Limon (Lemon)")
        
elif pred==3:
    print("Mangifera Indica (Mango)")
       
elif pred==4:
    print("Mentha (Mint)")
        
elif pred==5:
    print("Moringa Oleifera (Drumstick)")
        
elif pred==6:
    print("Murraya Koenigii (Curry)")
    
elif pred==7:
      print("Ocimum Tenuiflorum (Tulsi)")
      
elif pred==8:
      print("Santalum Album (Sandalwood)")
        
elif pred==9:
      print("Syzygium Jambos (Rose Apple)")
