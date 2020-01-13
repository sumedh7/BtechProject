import time
import picamera
import numpy as np
import keras
#from keras import models
from keras.preprocessing.image import array_to_img, img_to_array, load_img

with picamera.PiCamera() as camera:
	camera.resolution = (240, 640)
	camera.framerate = 10
	time.sleep(2)
	images = np.empty([100,240,640,3], dtype=np.uint8)
	#model = load_model('model.h5')

	for i in range(100):
		output = np.empty((240 * 640 * 3,), dtype=np.uint8)
		camera.capture(output, 'rgb')
		output = output.reshape((240, 640, 3))
		x = array_to_img(output)
		#x.save('{}.bmp'.format(i))
		image = np.array([output])       # the model expects 4D array
		images[i] = output
		print('Captured', i)