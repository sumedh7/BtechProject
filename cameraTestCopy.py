import time
import picamera
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Input
from keras.layers import Conv2D, MaxPooling2D, Activation 
from keras.layers.convolutional import Convolution2D
from keras.models import Model
import RPi.GPIO as GPIO
from time import sleep
import csv
GPIO.setmode(GPIO.BCM)
GPIO.setup(12, GPIO.OUT)
GPIO.setup(16, GPIO.OUT)
GPIO.setup(20, GPIO.OUT)
GPIO.setup(21, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)
GPIO.setup(19, GPIO.OUT)
en1 = GPIO.PWM(18,50)
en1.start(0)	
en2 = GPIO.PWM(19,50)
en2.start(0)	
def constrain(val, min_val, max_val):
	return min(max_val, max(min_val, val))
def MoveRobot(spdL,spdR):
	if (spdL>0):
		GPIO.output(12, 1)
		GPIO.output(16, 0)
	else:
		GPIO.output(12, 0)
		GPIO.output(16, 1)
	if (spdR>0):
		GPIO.output(20, 1)
		GPIO.output(21, 0)
	else:
		GPIO.output(20, 0)
		GPIO.output(21, 1)
	en1.ChangeDutyCycle(abs(spdL)*100/255)
	en2.ChangeDutyCycle(abs(spdR)*100/255)
	return 0
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 240, 640, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
main_input = Input(shape=INPUT_SHAPE, dtype='float32', name='main_input')
lambda1 = Lambda(lambda x: x/127.5 - 1.,
            input_shape=INPUT_SHAPE,
            output_shape=INPUT_SHAPE)(main_input)
convShared1 = Convolution2D(16, 17, 17, subsample=(4, 4), border_mode="same")(lambda1)
eluShared1 = ELU()(convShared1)
convShared2 = Convolution2D(32, 11, 11, subsample=(2, 2), border_mode="same")(eluShared1)
eluShared2 = ELU()(convShared2)
SteeringConv = Convolution2D(64, 11, 11, subsample=(2, 2), border_mode="same")(eluShared2)
SteeringFlat = Flatten()(SteeringConv)
SteeringDrop1 = Dropout(.2)(SteeringFlat)
SteeringElu1 = ELU()(SteeringDrop1)
SteeringDense = Dense(512)(SteeringElu1)
SteeringDrop2 = Dropout(.5)(SteeringDense)
SteeringElu2 = ELU()(SteeringDrop2)
steering = Dense(1)(SteeringElu2)
model = Model(inputs=main_input, outputs= steering)
model.summary()
with picamera.PiCamera() as camera:
	#model.load_weights('steering.h5')
	print('Loaded model')
	#	images = np.empty([100,480,640,3], dtype=np.uint8)
	camera.resolution = (640, 480)
	camera.framerate = 24
	time.sleep(2)
	counter = 0
	actual = []
	with open('/home/pi/my-awesome-project/Run16.csv','rt')as f:
		data = csv.reader(f)
		for row in data:
		#print(row[1])
			actual.append(float(row[1]))
	while(1):
		output = np.empty((480, 640, 3), dtype=np.uint8)
		camera.capture(output, 'rgb')
		print(counter)
		data = output[240:480,:,:]
		data = np.array([data])
		#x = float(model.predict(data, batch_size=1))
		x = actual[counter]
		print(x)
		y = 100
		speedL = y + x
		speedR = y - x
		speedL = constrain(speedL,-255,255)
		speedR = constrain(speedR,-255,255)
		a = MoveRobot(speedL,speedR)
		#images[counter] = output
		#x = array_to_img(output)
		counter = counter + 1
		if (counter==len(actual)):
			x = array_to_img(output)
			x.save('{}.bmp'.format(counter))
			break