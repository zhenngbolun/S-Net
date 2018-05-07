from keras import backend as K
from keras.layers import Input, Conv2D, add, UpSampling2D, Conv2DTranspose, Concatenate, merge, Lambda, PReLU, BatchNormalization
from keras.models import Model, Sequential
from keras import optimizers
from keras.optimizers import Adam

import autoEncoder as ae
import options as opt
from layers import PixelShuffle

features = opt.features
data_size = opt.data_size
channels = opt.channels
res_blocks = opt.res_blocks
batch_size = opt.batch_size

compare_features = 64
def res_block(x):
	br_conv_1 = Conv2D(features, 3, activation = 'relu', padding = 'same')(x)
	br_conv_2 = Conv2D(features, 3, padding = 'same')(br_conv_1)
	br_conv_2 = Lambda(lambda x: x*0.1)(br_conv_2)
	br_out = add([x, br_conv_2])

	return br_out

def classic_res_block(x):
	br_conv_1 = Conv2D(features, 3, activation = 'relu', padding = 'same')(x)
	br_out = add([x, br_conv_1])
	return br_out

def sab_loss(vects):
	gt, hr = vects
	return K.sum(K.abs(gt - hr), axis = (-1,-2,-3))

def sse_loss(vects):
	gt, hr = vects
	return K.sum(K.square(gt - hr), axis = (-1,-2,-3))

def mab_loss(vects):
	gt, hr = vects
	return K.mean(K.abs(gt - hr), axis = (-1,-2,-3))

def mse_loss(vects):
	gt, hr = vects
	return K.mean(K.square(gt - hr), axis = (-1,-2,-3))

def identity_loss(y_true, y_pred):
	return K.mean(y_pred)

def multiTaskResNetModel(QF, input_size=(), decode_size=(), mode='train'):
	if mode == 'train':
		input_size = (data_size, data_size, channels)
		decode_size = (data_size, data_size, features)
	png = Input(shape = input_size, name = 'png')
	jpg = Input(shape = input_size, name = 'jpg')
	#hr_encoder = ae.encoderModel(input_size)
	jpg_encoder = ae.encoderModel(input_size)
	decoder = ae.decoderModel(decode_size)

	#png_encoded = hr_encoder(png)
	#hr = decoder(png_encoded)
	jpg_encoded = jpg_encoder(jpg)	
	jpg_encoded_1 = res_block(jpg_encoded)
	lr_1 = decoder(jpg_encoded_1)
	jpg_encoded_2 = res_block(jpg_encoded_1)
	lr_2 = decoder(jpg_encoded_2)
	jpg_encoded_3 = res_block(jpg_encoded_2)
	lr_3 = decoder(jpg_encoded_3)
	jpg_encoded_4 = res_block(jpg_encoded_3)
	lr_4 = decoder(jpg_encoded_4)
	jpg_encoded_5 = res_block(jpg_encoded_4)
	lr_5 = decoder(jpg_encoded_5)
	jpg_encoded_6 = res_block(jpg_encoded_5)
	lr_6 = decoder(jpg_encoded_6)
	jpg_encoded_7 = res_block(jpg_encoded_6)
	lr_7 = decoder(jpg_encoded_7)
	jpg_encoded_8 = res_block(jpg_encoded_7)
	lr_8 = decoder(jpg_encoded_8)

	if mode == 'train':
		#hr_loss = merge([png, hr], mode = sse_loss, name = 'hr_loss', output_shape = (1,))
		lr_loss_1 = merge([png, lr_1], mode = mse_loss, name = 'lr_loss_1', output_shape = (1,))
		lr_loss_2 = merge([png, lr_2], mode = mse_loss, name = 'lr_loss_2', output_shape = (1,))
		lr_loss_3 = merge([png, lr_3], mode = mse_loss, name = 'lr_loss_3', output_shape = (1,))
		lr_loss_4 = merge([png, lr_4], mode = mse_loss, name = 'lr_loss_4', output_shape = (1,))
		lr_loss_5 = merge([png, lr_5], mode = mse_loss, name = 'lr_loss_5', output_shape = (1,))
		lr_loss_6 = merge([png, lr_6], mode = mse_loss, name = 'lr_loss_6', output_shape = (1,))
		lr_loss_7 = merge([png, lr_7], mode = mse_loss, name = 'lr_loss_7', output_shape = (1,))
		lr_loss_8 = merge([png, lr_8], mode = mse_loss, name = 'lr_loss_8', output_shape = (1,))
		model = Model([png, jpg], [lr_loss_1, lr_loss_2, lr_loss_3, lr_loss_4, lr_loss_5, lr_loss_6, lr_loss_7, lr_loss_8])
	else:
		model = Model([png, jpg], [lr_1, lr_2, lr_3, lr_4,lr_5,lr_6,lr_7,lr_8])
	return model

def multiTaskClassicResNetModel(QF, input_size=(), decode_size=(), mode='train'):
	if mode == 'train':
		input_size = (data_size, data_size, channels)
		decode_size = (data_size, data_size, features)
	png = Input(shape = input_size, name = 'png')
	jpg = Input(shape = input_size, name = 'jpg')
	#hr_encoder = ae.encoderModel(input_size)
	jpg_encoder = ae.encoderModel(input_size)
	decoder = ae.decoderModel(decode_size)

	#png_encoded = hr_encoder(png)
	#hr = decoder(png_encoded)
	jpg_encoded = jpg_encoder(jpg)	
	jpg_encoded_1 = classic_res_block(jpg_encoded)
	lr_1 = decoder(jpg_encoded_1)
	jpg_encoded_2 = classic_res_block(jpg_encoded_1)
	lr_2 = decoder(jpg_encoded_2)
	jpg_encoded_3 = classic_res_block(jpg_encoded_2)
	lr_3 = decoder(jpg_encoded_3)
	jpg_encoded_4 = classic_res_block(jpg_encoded_3)
	lr_4 = decoder(jpg_encoded_4)
	jpg_encoded_5 = classic_res_block(jpg_encoded_4)
	lr_5 = decoder(jpg_encoded_5)
	jpg_encoded_6 = classic_res_block(jpg_encoded_5)
	lr_6 = decoder(jpg_encoded_6)
	jpg_encoded_7 = classic_res_block(jpg_encoded_6)
	lr_7 = decoder(jpg_encoded_7)
	jpg_encoded_8 = classic_res_block(jpg_encoded_7)
	lr_8 = decoder(jpg_encoded_8)

	if mode == 'train':
		#hr_loss = merge([png, hr], mode = sse_loss, name = 'hr_loss', output_shape = (1,))
		lr_loss_1 = merge([png, lr_1], mode = mse_loss, name = 'lr_loss_1', output_shape = (1,))
		lr_loss_2 = merge([png, lr_2], mode = mse_loss, name = 'lr_loss_2', output_shape = (1,))
		lr_loss_3 = merge([png, lr_3], mode = mse_loss, name = 'lr_loss_3', output_shape = (1,))
		lr_loss_4 = merge([png, lr_4], mode = mse_loss, name = 'lr_loss_4', output_shape = (1,))
		lr_loss_5 = merge([png, lr_5], mode = mse_loss, name = 'lr_loss_5', output_shape = (1,))
		lr_loss_6 = merge([png, lr_6], mode = mse_loss, name = 'lr_loss_6', output_shape = (1,))
		lr_loss_7 = merge([png, lr_7], mode = mse_loss, name = 'lr_loss_7', output_shape = (1,))
		lr_loss_8 = merge([png, lr_8], mode = mse_loss, name = 'lr_loss_8', output_shape = (1,))
		model = Model([png, jpg], [lr_loss_1, lr_loss_2, lr_loss_3, lr_loss_4, lr_loss_5, lr_loss_6, lr_loss_7, lr_loss_8])
	else:
		model = Model([png, jpg], [lr_1, lr_2, lr_3,lr_4, lr_5, lr_6, lr_7, lr_8])
	return model

def multiTaskPReLUModel(QF, input_size=(), decode_size=(), mode='train'):
	def encoderModel(input_size, decode_size):
		x = Input(shape = input_size)
		t = Conv2D(features, 5, padding = 'same')(x)
		t = PReLU(shared_axes=[1, 2])(t)
		y = Conv2D(features, 3, padding = 'same')(t)
		y = PReLU(shared_axes=[1, 2])(y)
		model  = Model(x, y)
		return model

	def decoderModel(input_size, decode_size):
		x = Input(shape = decode_size)
		t = Conv2D(features, 3, padding = 'same', activation = 'relu')(x)
		t = PReLU(shared_axes=[1, 2])(t)
		y = Conv2D(channels, 5, padding = 'same', activation = 'relu')(t)
		y = PReLU(shared_axes=[1, 2])(y)
		model  = Model(x, y)
		return model

	if mode == 'train':
		input_size = (data_size, data_size, channels)
		decode_size = (data_size, data_size, features)
	png = Input(shape = input_size, name = 'png')
	jpg = Input(shape = input_size, name = 'jpg')
	hr_encoder = ae.encoderModel(input_size)
	jpg_encoder = ae.encoderModel(input_size)
	decoder = ae.decoderModel(decode_size)

	png_encoded = hr_encoder(png)
	hr = decoder(png_encoded)
	jpg_encoded = jpg_encoder(jpg)	
	jpg_encoded_1 = Conv2D(features, 3, padding = 'same')(jpg_encoded)
	jpg_encoded_1 = PReLU(shared_axes=[1,2])(jpg_encoded_1)
	lr_1 = decoder(jpg_encoded_1)
	jpg_encoded_2 = Conv2D(features, 3, padding = 'same')(jpg_encoded_1)
	jpg_encoded_2 = PReLU(shared_axes=[1,2])(jpg_encoded_2)
	lr_2 = decoder(jpg_encoded_2)
	jpg_encoded_3 = Conv2D(features, 3, padding = 'same')(jpg_encoded_2)
	jpg_encoded_3= PReLU(shared_axes=[1,2])(jpg_encoded_3)
	lr_3 = decoder(jpg_encoded_3)
	jpg_encoded_4 = Conv2D(features, 3, padding = 'same')(jpg_encoded_3)
	jpg_encoded_4 = PReLU(shared_axes=[1,2])(jpg_encoded_4)
	lr_4 = decoder(jpg_encoded_4)
	jpg_encoded_5 = Conv2D(features, 3, padding = 'same')(jpg_encoded_4)
	jpg_encoded_5 = PReLU(shared_axes=[1,2])(jpg_encoded_5)
	lr_5 = decoder(jpg_encoded_5)
	jpg_encoded_6 = Conv2D(features, 3, padding = 'same')(jpg_encoded_5)
	jpg_encoded_5 = PReLU(shared_axes=[1,2])(jpg_encoded_5)
	lr_6 = decoder(jpg_encoded_6)
	jpg_encoded_7 = Conv2D(features, 3, padding = 'same')(jpg_encoded_6)
	jpg_encoded_7 = PReLU(shared_axes=[1,2])(jpg_encoded_7)
	lr_7 = decoder(jpg_encoded_7)
	jpg_encoded_8 = Conv2D(features, 3, padding = 'same')(jpg_encoded_7)
	jpg_encoded_8 = PReLU(shared_axes=[1,2])(jpg_encoded_8)
	lr_8 = decoder(jpg_encoded_8)

	if mode == 'train':
		hr_loss = merge([png, hr], mode = mse_loss, name = 'hr_loss', output_shape = (1,))
		lr_loss_1 = merge([png, lr_1], mode = mse_loss, name = 'lr_loss_1', output_shape = (1,))
		lr_loss_2 = merge([png, lr_2], mode = mse_loss, name = 'lr_loss_2', output_shape = (1,))
		lr_loss_3 = merge([png, lr_3], mode = mse_loss, name = 'lr_loss_3', output_shape = (1,))
		lr_loss_4 = merge([png, lr_4], mode = mse_loss, name = 'lr_loss_4', output_shape = (1,))
		lr_loss_5 = merge([png, lr_5], mode = mse_loss, name = 'lr_loss_5', output_shape = (1,))
		lr_loss_6 = merge([png, lr_6], mode = mse_loss, name = 'lr_loss_6', output_shape = (1,))
		lr_loss_7 = merge([png, lr_7], mode = mse_loss, name = 'lr_loss_7', output_shape = (1,))
		lr_loss_8 = merge([png, lr_8], mode = mse_loss, name = 'lr_loss_8', output_shape = (1,))
		model = Model([png, jpg], [hr_loss, lr_loss_1, lr_loss_2, lr_loss_3, lr_loss_4, lr_loss_5, lr_loss_6, lr_loss_7, lr_loss_8])
	else:
		model = Model([png, jpg], [hr, lr_5, lr_6, lr_7, lr_8])
	return model

def ClassicResNetModel(QF, input_size=(), decode_size=(), mode='train'):
	if mode == 'train':
		input_size = (data_size, data_size, channels)
		decode_size = (data_size, data_size, features)
	png = Input(shape = input_size, name = 'png')
	jpg = Input(shape = input_size, name = 'jpg')
	#hr_encoder = ae.encoderModel(input_size)
	jpg_encoder = ae.encoderModel(input_size)
	decoder = ae.decoderModel(decode_size)

	#png_encoded = hr_encoder(png)
	#hr = decoder(png_encoded)
	jpg_encoded = jpg_encoder(jpg)	
	jpg_encoded_1 = classic_res_block(jpg_encoded)
	jpg_encoded_2 = classic_res_block(jpg_encoded_1)
	jpg_encoded_3 = classic_res_block(jpg_encoded_2)
	jpg_encoded_4 = classic_res_block(jpg_encoded_3)
	jpg_encoded_5 = classic_res_block(jpg_encoded_4)
	jpg_encoded_6 = classic_res_block(jpg_encoded_5)
	jpg_encoded_7 = classic_res_block(jpg_encoded_6)
	jpg_encoded_8 = classic_res_block(jpg_encoded_7)
	lr_8 = decoder(jpg_encoded_8)

	if mode == 'train':
		lr_loss_8 = merge([png, lr_8], mode = mse_loss, name = 'lr_loss_8', output_shape = (1,))
		model = Model([png, jpg], [lr_loss_8])
	else:
		model = Model([png, jpg], [lr_8])
	return model

def ResNetModel(QF, input_size=(), decode_size=(), mode='train'):
	if mode == 'train':
		input_size = (data_size, data_size, channels)
		decode_size = (data_size, data_size, features)
	png = Input(shape = input_size, name = 'png')
	jpg = Input(shape = input_size, name = 'jpg')
	#hr_encoder = ae.encoderModel(input_size)
	jpg_encoder = ae.encoderModel(input_size)
	decoder = ae.decoderModel(decode_size)

	#png_encoded = hr_encoder(png)
	#hr = decoder(png_encoded)
	jpg_encoded = jpg_encoder(jpg)	
	jpg_encoded_1 = res_block(jpg_encoded)
	jpg_encoded_2 = res_block(jpg_encoded_1)
	jpg_encoded_3 = res_block(jpg_encoded_2)
	jpg_encoded_4 = res_block(jpg_encoded_3)
	jpg_encoded_5 = res_block(jpg_encoded_4)
	jpg_encoded_6 = res_block(jpg_encoded_5)
	jpg_encoded_7 = res_block(jpg_encoded_6)
	jpg_encoded_8 = res_block(jpg_encoded_7)
	lr_8 = decoder(jpg_encoded_8)

	if mode == 'train':
		lr_loss_8 = merge([png, lr_8], mode = mse_loss, name = 'lr_loss_8', output_shape = (1,))
		model = Model([png, jpg], [lr_loss_8])
	else:
		model = Model([png, jpg], [lr_8])
	return model