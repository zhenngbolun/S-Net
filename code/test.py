from PIL import Image
import numpy as np
import os 
import time

import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Conv2D, add, UpSampling2D, Conv2DTranspose, merge
from keras.models import Model, Sequential
from keras import optimizers
from keras.optimizers import Adam

import options as opt
import models

channels = opt.channels
features = opt.features
DIV2K_HR_path = opt.DIV2K_HR_path
ratio = opt.ratio

def get_Y(x):
	r = x[:,:,0]
	g = x[:,:,1]
	b = x[:,:,2]
	y = 0.257*r + 0.504*g + 0.098*b + 0.0627
	return y

def get_YCbCr(x):
	r = x[:,:,0]
	g = x[:,:,1]
	b = x[:,:,2]
	ycbcr = x
	ycbcr[:,:,0] = 0.257*r + 0.504*g + 0.098*b + 0.0627
	ycbcr[:,:,1] = -0.148*r - 0.291*g + 0.439*b + 0.5
	ycbcr[:,:,2] = 0.439*r - 0.368*g - 0.071*b + 0.5
	
	return ycbcr

def get_RGB(x):
	y = x[:,:,0]
	cb = x[:,:,1]
	cr = x[:,:,2]
	rgb = x
	rgb[:,:,0] = 1.164*y + 1.596*cr - 222.9/255
	rgb[:,:,1] = 1.164*y - 0.392*cb - 0.823*cr + 135.6/255
	rgb[:,:,2] = 1.164*y + 2.017*cb - 276.8/255

	return rgb
def get_Image(_jpg):
	r = Image.fromarray(_jpg[0,:,:,0]*255).convert('L')
	g = Image.fromarray(_jpg[0,:,:,1]*255).convert('L')
	b = Image.fromarray(_jpg[0,:,:,2]*255).convert('L')
	z = Image.merge('RGB', (r,g,b))
	return z

def test(QF, dataset, batches, img_name = None):
	if opt.model_type == 'SR':
		dataset = 'Set5'
	if dataset == 'DIV2K':
		hr_path = DIV2K_HR_path
		hr_tail = '.png'
		lr_path = '/media/scs4450/hard/zbl/srcnn/src/train/DIV2K/DIV2K_QF_' + str(QF) + '/'
	if dataset == 'LIVE1':
		hr_path = 'data/LIVE1/'
		if channels == 1:
			hr_tail = '.png'
		if channels == 3:
			hr_tail = '.bmp'			
		lr_path = hr_path + 'QF_' + str(QF) + '/'
		save_path8 = hr_path + '8/QF_' + str(QF) + '_test/'
		save_path7 = hr_path + '7/QF_' + str(QF) + '_test/'
		save_path6 = hr_path + '6/QF_' + str(QF) + '_test/'
		save_path5 = hr_path + '5/QF_' + str(QF) + '_test/'
		save_path4 = hr_path + '4/QF_' + str(QF) + '_test/'
		save_path3 = hr_path + '3/QF_' + str(QF) + '_test/'
		save_path2 = hr_path + '2/QF_' + str(QF) + '_test/'
		save_path1 = hr_path + '1/QF_' + str(QF) + '_test/'
	if dataset == 'BSDS500':
		hr_path = 'data/BSDS500/all_img/'
		hr_tail = '.jpg'
		lr_path = hr_path + 'QF_' + str(QF) + '/'
		save_path8 = hr_path + '8/QF_' + str(QF) + '_test/'
		save_path7 = hr_path + '7/QF_' + str(QF) + '_test/'
		save_path6 = hr_path + '6/QF_' + str(QF) + '_test/'
		save_path5 = hr_path + '5/QF_' + str(QF) + '_test/'
		save_path4 = hr_path + '4/QF_' + str(QF) + '_test/'
		save_path3 = hr_path + '3/QF_' + str(QF) + '_test/'
		save_path2 = hr_path + '2/QF_' + str(QF) + '_test/'
		save_path1 = hr_path + '1/QF_' + str(QF) + '_test/'
	if dataset == 'Set5':
		hr_path = 'data/Set5/'
		hr_tail = '.bmp'
		save_path8 = hr_path + '8/x' + str(opt.ratio) + '/'
		save_path7 = hr_path + '7/x' + str(opt.ratio) + '/'
		save_path6 = hr_path + '6/x' + str(opt.ratio) + '/'
		save_path5 = hr_path + '5/x' + str(opt.ratio) + '/'
	if dataset == 'WIN143':
		hr_path = 'data/WIN143/'
		hr_tail = '.png'
		w,h = 1920,1080
		lr_path = hr_path + 'QF_' + str(QF) + '/'
		save_path8 = hr_path + '8/QF_' + str(QF) + '_test/'
		save_path7 = hr_path + '7/QF_' + str(QF) + '_test/'
		save_path6 = hr_path + '6/QF_' + str(QF) + '_test/'
		save_path5 = hr_path + '5/QF_' + str(QF) + '_test/'
		save_path4 = hr_path + '4/QF_' + str(QF) + '_test/'
		save_path3 = hr_path + '3/QF_' + str(QF) + '_test/'
		save_path2 = hr_path + '2/QF_' + str(QF) + '_test/'
		save_path1 = hr_path + '1/QF_' + str(QF) + '_test/'
		input_size = (h, w, channels)
		decode_size = (h, w, features)
		model = models.multiTaskResNetModel(QF = QF, input_size = input_size, decode_size = decode_size, mode = 'test')
		weights_path = 'model/QF_'+str(QF)+'/MTRN1_k8_f'+str(features)+'_c'+str(channels)+'_QF'+str(QF)+'_'+str(batches)+'.h5'
		model.load_weights(weights_path)

	count = 0
	file_list = os.listdir(hr_path)
	if img_name == None:
		for file in file_list:
			if file.find(hr_tail) >= 0:
				count = count + 1
				print count, file
				if count <= 0:
					continue
				s = str.split(file, '.')
				hr = Image.open(hr_path + s[0] + hr_tail)
				width, height = hr.size
				if dataset == 'WIN143':
					if width != w or height != h:
						print 'Size Error: ', width, height
						continue
				if dataset != 'Set5':
					lr = Image.open(lr_path + s[0] + '.jpg')
					if dataset == 'DIV2K':
						width >>= 1
						height >>= 1
						bbox = (0,0,width,height)
						lr = lr.crop(bbox)
						hr = hr.crop(bbox)
				else:
					width = width - width%ratio
					height = height - height%ratio
					lr_size = (int(width/ratio), int(height/ratio))
					bbox = (0, 0, width, height)
					hr = hr.crop(bbox)
					lr = hr.resize(lr_size, Image.BICUBIC)		
				
				if opt.model_type != 'SR':
					input_size = (height, width, channels)
					decode_size = (height, width, features)
				else:
					input_size = (int(height/ratio), int(width/ratio), channels)
					decode_size = (int(height/ratio), int(width/ratio), features)
					if opt.model_type == 'MultiTaskPReLU':
						model = models.multiTaskPReLUModel(QF = QF, input_size = input_size, decode_size = decode_size, mode = 'test')
						weights_path = 'model/MT_PReLU_k8_f'+str(features)+'_c'+str(channels)+'_QF'+str(QF)+'_'+str(batches)+'.h5'
					if opt.model_type == 'MultiTaskClassicResNet':
						model = models.multiTaskClassicResNetModel(QF = QF, input_size = input_size, decode_size = decode_size, mode = 'test')
						weights_path = 'model/CRN_k8_f'+str(features)+'_c'+str(channels)+'_QF'+str(QF)+'_'+str(batches)+'.h5'
					if opt.model_type == 'ClassicResNet':
						model = models.ClassicResNetModel(QF = QF, input_size = input_size, decode_size = decode_size, mode = 'test')
						weights_path = 'model/CRN_k8_f'+str(features)+'_c'+str(channels)+'_QF'+str(QF)+'_'+str(batches)+'.h5'
					if opt.model_type == 'MultiTaskResNet':
						model = models.multiTaskResNetModel(QF = QF, input_size = input_size, decode_size = decode_size, mode = 'test')
						weights_path = 'model/QF_'+str(QF)+'/MTRN1_k8_f'+str(features)+'_c'+str(channels)+'_QF'+str(QF)+'_'+str(batches)+'.h5'
					if opt.model_type == 'ResNet1':
						model = models.ResNetModel(QF = QF, input_size = input_size, decode_size = decode_size, mode = 'test')
						weights_path = 'model/RN_k3_f'+str(features)+'_c'+str(channels)+'_QF'+str(QF)+'_'+str(batches)+'.h5'
					model.load_weights(weights_path)


				if opt.model_type != 'SR':
					jpg_arr = np.array(lr)/255.0
					png_arr = np.array(hr)/255.0
					if channels == 1 and lr.mode =='RGB':
						jpg_arr = get_Y(jpg_arr)
					if channels == 1 and hr.mode == 'RGB':
						png_arr = get_Y(png_arr)

					jpg_arr = jpg_arr.reshape(1, height, width, channels)									
					png_arr = png_arr.reshape(1, height, width, channels)
				
					[ _jpg1, _jpg2, _jpg3, _jpg4, _jpg5, _jpg6, _jpg7, _jpg8] = model.predict([png_arr, jpg_arr])
					'''for i in range(0, 100):
						time1 = time.time()
						[_jpg8] = model.predict([png_arr, jpg_arr])
						time2 = time.time()
						print time2-time1'''
				else:
					lr_arr = np.array(lr)/255.0
					lr_arr = lr_arr.reshape(1, int(height/ratio), int(width/ratio), channels)
					png_arr = np.array(hr)/255.0
					png_arr = png_arr.reshape(1, height, width, channels)
					#[_png, _jpg5, _jpg6, _jpg7, _jpg8] = model.predict([png_arr, lr_arr])
					for i in range(0, 100):
						time1 = time.time()
						[_jpg8] = model.predict([png_arr, lr_arr])
						time2 = time.time()
						print time2-time1
				if channels == 3:
					z8 = get_Image(_jpg8)
					z7 = get_Image(_jpg7)
					z6 = get_Image(_jpg6)
					z5 = get_Image(_jpg5)
					z4 = get_Image(_jpg4)
					z3 = get_Image(_jpg3)
					z2 = get_Image(_jpg2)
					z1 = get_Image(_jpg1)

				else:
					z8 = Image.fromarray(_jpg8[0,:,:,0]*255).convert('L')
					#z7 = Image.fromarray(_jpg7[0,:,:,0]*255).convert('L')
					#z6 = Image.fromarray(_jpg6[0,:,:,0]*255).convert('L')
					#z5 = Image.fromarray(_jpg5[0,:,:,0]*255).convert('L')
				z8.save(save_path8+s[0]+'.bmp')
				z7.save(save_path7+s[0]+'.bmp')
				z6.save(save_path6+s[0]+'.bmp')
				z5.save(save_path5+s[0]+'.bmp')
				z4.save(save_path4+s[0]+'.bmp')
				z3.save(save_path3+s[0]+'.bmp')
				z2.save(save_path2+s[0]+'.bmp')
				z1.save(save_path1+s[0]+'.bmp')


def get_session():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	return tf.Session(config = config)


if __name__ == '__main__':		
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	K.tensorflow_backend.set_session(get_session())

	test(QF=20, dataset = 'WIN143', batches = 50000)
