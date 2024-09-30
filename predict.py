import numpy as np
import cv2
# import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model
import os
import pickle
import imutils
# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a numpy array (not numpy matrix or scipy matrix) and a list of strings.
# Make sure that the length of the array and the list is the same as the number of filenames that 
# were given. The evaluation code may give unexpected results if this convention is not followed.
save_path = "write.txt"

def load_image(image_path):
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	newimg =  np.zeros((image.shape), np.uint8)
	newimg[:] = image[0][0]
	image = cv2.subtract(image, newimg)
	image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	kernel = np.ones((3,3), np.uint8) 
	image = cv2.erode(image, kernel, iterations=8)
	gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.copyMakeBorder(gray2, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	return thresh

def resize_to_fit(image, width, height):
	"""
	A helper function to resize an image to fit within a given size
	:param image: image to resize
	:param width: desired width in pixels
	:param height: desired height in pixels
	:return: the resized image
	"""

	# grab the dimensions of the image, then initialize
	# the padding values
	(h, w) = image.shape[:2]

	# if the width is greater than the height then resize along
	# the width
	if w > h:
		image = imutils.resize(image, width=width)

	# otherwise, the height is greater than the width so resize
	# along the height
	else:
		image = imutils.resize(image, height=height)

	# determine the padding values for the width and height to
	# obtain the target dimensions
	padW = int((width - image.shape[1]) / 2.0)
	padH = int((height - image.shape[0]) / 2.0)

	# pad the image then apply one more resizing to handle any
	# rounding issues
	image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
		cv2.BORDER_REPLICATE)
	image = cv2.resize(image, (width, height))

	# return the pre-processed image
	return image

def decaptcha( filenames ):
	k = 0
	# numChars = 3 * np.ones( (len( filenames ),) )

	model = load_model("captcha_model.hdf5")
	# if not os.path.exists(save_path):
	# 	os.makedirs(save_path)
	if os.path.exists("model.txt"):
		os.system("rm model.txt")
	file = open("model.txt", "a")

	with open("model_labels.dat", "rb") as f:
		lb = pickle.load(f)
	i = 0
	for image_file in filenames:
		thresh  = load_image(image_file)
		# if(k == 0):
		# 	cv2.imshow("new image", thresh)
		# 	cv2.waitKey(0); 
		# 	k = k+1
		# 	exit()
		yLim = thresh.shape[0]
		xLim = thresh.shape[1]
		#print(xLim,yLim)
		yLim_Sum = 0
		x = 0
		flag = 0

		x_coord = 0    
		y_coord = 0
		w_coord = 0
		h_coord = yLim-1

		HEIGHT_THRESH = 10
		THRESH_MINVALUE = yLim*255
		WIDTH_THRESH = 10
		letter=0
		predictions = ""
		count = 0
		while(xLim > x):
			flag=0
			yLim_Sum=0
			for y in range(0,yLim):
				yLim_Sum = yLim_Sum + thresh[y][x]
			# print(yLim_Sum,"  ", THRESH_MINVALUE)
			if(yLim_Sum < THRESH_MINVALUE):
				x_coord=x
				while(yLim_Sum < THRESH_MINVALUE and x < xLim):
					yLim_Sum = 0
					for y1 in range(0,yLim):
						yLim_Sum = yLim_Sum + thresh[y1][x]
					x = x + 1
					flag = 1
				w_coord = x - x_coord
			if(w_coord > WIDTH_THRESH and flag==1):
				letter_image = thresh[y_coord+20 : y_coord + h_coord - 20, max(0,x_coord-10) :x_coord + w_coord + 10]
				count = count+1
				letter_image = resize_to_fit(letter_image, 20, 20)
				
				
				# Turn the single image into a 4d list of images to make Keras happy
				letter_image = np.expand_dims(letter_image, axis=2)
				letter_image = np.expand_dims(letter_image, axis=0)

				# Ask the neural network to make a prediction
				prediction = model.predict(letter_image/255)
				# if x%100 == 0:
					# print(prediction)
				# Convert the one-hot-encoded prediction back to a normal letter
				letter = lb.inverse_transform(prediction)[0]
				if(count==1):
					predictions += letter
				else:
					predictions += ','+letter	

							
			if(flag==0):
				x = x+1

		# numChars[i] = count
		i = i+1
		file.write(str(predictions)+ "\n")

	# file.close()

	# The use of a model file is just for sake of illustration
	file = open( "model.txt", "r" )
	codes = file.read().splitlines()
	# file.close()
	return (codes)