from PIL import Image
import numpy as np
import os 

import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Conv2D, add, UpSampling2D, Conv2DTranspose, merge
from keras.models import Model, Sequential
from keras import optimizers
from keras.optimizers import Adam

import options as opt
DIV2K_HR_path = opt.DIV2K_HR_path

################ data parameter ##################
data_size = opt.data_size
step = data_size

#features = opt.features
features = opt.features
channels = opt.channels
##################################################

################ leraning paramter ###############
learn_rate = 1e-5
epochs = 1
batch_size = 128
##################################################

def abs_loss(y_true, y_pred):
	return K.mean(K.sum(K.abs(y_true - y_pred), axis = (-1,-2,-3)))

def encoderModel(input_size):
	x = Input(shape = input_size)
	t = Conv2D(features, 5, padding = 'same', activation = 'relu')(x)
	y = Conv2D(features, 3, padding = 'same', activation = 'relu')(t)
	model  = Model(x, y)
	return model

def decoderModel(input_size):
	x = Input(shape = input_size)
	t = Conv2D(features, 3, padding = 'same', activation = 'relu')(x)
	y = Conv2D(channels, 5, padding = 'same', activation = 'relu')(t)
	model  = Model(x, y)
	return model


def train_HR(skip):
	data_list = []
	count = 0

	input_size = (data_size, data_size, channels) 
	x = Input(shape = input_size)
	encoder = encoderModel(input_size)
	encoded = encoder(x)
	decoder = decoderModel((data_size, data_size, features))
	y = decoder(encoded)
	aotu_encoder_model = Model(x, y)
	encoder.load_weights('model/encoder.h5')
	decoder.load_weights('model/decoder.h5')

	file_list = os.listdir(DIV2K_HR_path)
	for i, file in enumerate(file_list):
		s = str.split(file,'.')
		if len(s) == 2 and s[1] == 'png':
			count += 1
			if count <= skip:
				continue
			image = Image.open(DIV2K_HR_path + file)
			array = np.array(image)/255.0
			for y in range(0,array.shape[0], step):
				for x in range(0, array.shape[1], step):
					if  (y+data_size) <= array.shape[0] \
					and (x+data_size) <= array.shape[1]:
						_data = array[y:y+data_size, x:x+data_size]
						data_list.append(_data)

			if count%100 == 0:
				data_array = np.array(data_list)
				print data_array.shape

				aotu_encoder_model.compile(loss = abs_loss,
					optimizer = Adam(lr = learn_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8))

				aotu_encoder_model.fit(data_array, data_array,
							batch_size = batch_size,
							epochs = epochs,
							verbose = 1)
				encoder.save('model/encoder.h5')
				decoder.save('model/decoder.h5')
				data_list = []

def train_jpg(QF, skip=0):
	DIV2K_QF_path = '/media/scs4450/hard/zbl/srcnn/src/train/DIV2K/DIV2K_QF_' + str(QF) + '/'
	save_path = 'model/QF_' + str(QF) + '/'

	data_list = []
	count = 0

	input_size = (data_size, data_size, channels) 
	x = Input(shape = input_size)
	encoder = encoderModel(input_size)
	encoded = encoder(x)
	decoder = decoderModel((data_size, data_size, features))
	y = decoder(encoded)
	aotu_encoder_model = Model(x, y)
	encoder.load_weights('model/encoder.h5')
	decoder.load_weights('model/decoder.h5')

	file_list = os.listdir(DIV2K_QF_path)
	for i, file in enumerate(file_list):
		s = str.split(file,'.')
		if len(s) == 2 and s[1] == 'jpg':
			count += 1
			if count <= skip:
				continue
			if count > 800:
				break;
			image = Image.open(DIV2K_QF_path + file)
			array = np.array(image)/255.0
			for y in range(0,array.shape[0], step):
				for x in range(0, array.shape[1], step):
					if  (y+data_size) <= array.shape[0] \
					and (x+data_size) <= array.shape[1]:
						_data = array[y:y+data_size, x:x+data_size]
						data_list.append(_data)

			if count%100 == 0:
				data_array = np.array(data_list)
				print data_array.shape

				aotu_encoder_model.compile(loss = abs_loss,
					optimizer = Adam(lr = learn_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8))

				aotu_encoder_model.fit(data_array, data_array,
							batch_size = batch_size,
							epochs = epochs,
							verbose = 1)
				encoder.save(save_path + 'encoder.h5')
				decoder.save(save_path + 'decoder.h5')
				data_list = []

if __name__ == '__main__':		
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	train_jpg(10)
