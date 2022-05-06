import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
#Load the saved model
model = tf.keras.models.load_model('firemodel.h5')
video = cv2.VideoCapture(0)
while True:
        _, frame = video.read()

        im = Image.fromarray(frame, 'RGB')

        im = im.resize((224,224))
        img_array = image.img_to_array(im)
        img_array = np.expand_dims(img_array, axis=0) / 255
        probabilities = model.predict(img_array)[0]
        #Calling the predict method on model to predict 'fire' on the image
        prediction = np.argmax(probabilities)
        #if fire is detected in the frame we turn it gray.
        if prediction == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                print(probabilities[prediction])
        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
        		break
        
video.release()
cv2.destroyAllWindows()
