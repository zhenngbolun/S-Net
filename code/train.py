from PIL import Image
import numpy as np
import os 
import random
import gc

import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Conv2D, add, UpSampling2D, Conv2DTranspose, merge
from keras.models import Model, Sequential
from keras import optimizers
from keras.optimizers import Adam, SGD

import options as opt
import models


data_size = opt.data_size
channels = opt.channels
features = opt.features
step = opt.step
batch_size = opt.batch_size
epochs = opt.epochs
weights = opt.weights
snapshot = opt.snapshot
if opt.random:
	x_step = int(step*random.uniform(1.5,2.5))
	y_step = int(step*random.uniform(1.5,2.5))
else:
	x_step = step
	y_step = step

def get_Y(x):
	r = x[:,:,0]
	g = x[:,:,1]
	b = x[:,:,2]
	y = (65.481*r + 128.553*g + 24.966*b + 16)/255
	return y

def get_YCbCr(x):
	r = x[:,:,0]
	g = x[:,:,1]
	b = x[:,:,2]
	ycbcr = np.ones(x.shape)
	ycbcr[:,:,0] = 0.257*r + 0.504*g + 0.098*b + 0.0627
	ycbcr[:,:,1] = -0.148*r - 0.291*g + 0.439*b + 128.0/255
	ycbcr[:,:,2] = 0.439*r - 0.368*g - 0.071*b + 128.0/255
	
	return ycbcr

def compile(model, learn_rate):
	model.compile(loss = {'hr_loss':models.identity_loss, 'lr_loss':models.identity_loss},
							optimizer = Adam(lr = learn_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8))

def compileMutiTask(model, learn_rate):
	model.compile(loss = {#'hr_loss':models.identity_loss, 
						  'lr_loss_1':models.identity_loss,
						  'lr_loss_2':models.identity_loss,
						  'lr_loss_3':models.identity_loss,
						  'lr_loss_4':models.identity_loss,
						  'lr_loss_5':models.identity_loss,
						  'lr_loss_6':models.identity_loss,
						  'lr_loss_7':models.identity_loss,
						  'lr_loss_8':models.identity_loss},
				  loss_weights = {#'hr_loss' : 1.0,
				  				  'lr_loss_1' : 0.125,
				  				  'lr_loss_2' : 0.125,
				  				  'lr_loss_3' : 0.125,
				  				  'lr_loss_4' : 0.125,
				  				  'lr_loss_5' : 0.125,
				  				  'lr_loss_6' : 0.125,
				  				  'lr_loss_7' : 0.125,
				  				  'lr_loss_8' : 0.125},
				optimizer = Adam(lr = learn_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8))

def train_MultiTaskResNet_batches(QF, skip = 0):
	learn_rate = opt.learn_rate
	delta = opt.delta
	delta_interception = opt.delta_interception
	train_interception = opt.train_interception
	dataset = opt.dataset
	f_loss = open('loss_mtrn2.txt', 'wb+')

	if opt.res_blocks == 8:
		model = models.multiTaskResNetModel(QF = QF)
	if opt.res_blocks == 16:
		model = models.multiTask16Model(QF = QF)
	compileMutiTask(model, learn_rate)
	if weights != '':
		model.load_weights(weights)
	if dataset == 'DIV2K':
		HR_path = opt.DIV2K_HR_path
		QF_path = opt.DIV2K_QF_path
	else:
		HR_path = 'data/'+dataset+'/'
		QF_path = HR_path+'QF_'+str(opt.QF)+'/'
	file_list = os.listdir(HR_path)
	image_list = []
	for f in file_list:
		s = str.split(f,'.')
		if len(s) == 2 and (s[1] == 'png' or s[1] == 'jpg'):
			image_list.append(f)
			if len(image_list) == 800:
				break
	
	count = 0
	batches = 0
	png_data_list = []
	jpg_data_list = []
	data_list = []

	for k in range(0, epochs):
		random.shuffle(image_list)
		for file in image_list:
			s = str.split(file,'.')
			count += 1
			print count
			if opt.random:
				x_step = int(step*random.uniform(1.5,2.5))
				y_step = int(step*random.uniform(1.5,2.5))
			else:
				x_step = step
				y_step = step
			img_png = Image.open(HR_path + file)
			img_jpg = Image.open(QF_path + s[0] + opt.QF_tail)
			arr_png = np.array(img_png)/255.0
			arr_jpg = np.array(img_jpg)/255.0
			for y in range(0,arr_png.shape[0], y_step):
				for x in range(0, arr_png.shape[1], x_step):
					if  (y+data_size) <= arr_png.shape[0] \
					and (x+data_size) <= arr_png.shape[1]:
						png_data = arr_png[y:y+data_size, x:x+data_size]
						jpg_data = arr_jpg[y:y+data_size, x:x+data_size]
						if channels == 1:
							if img_png.mode == 'RGB':
								png_data = get_Y(png_data)
							if img_jpg.mode == 'RGB':
								jpg_data = get_Y(jpg_data)
							png_data = png_data.reshape(data_size, data_size, 1)
							jpg_data = jpg_data.reshape(data_size, data_size, 1)
						data_list.append((png_data,jpg_data))

			if count % train_interception == 0:
				random.shuffle(data_list)
				for _data in data_list:
					png_data_list.append(_data[0])
					jpg_data_list.append(_data[1])
				data_list = []
				png_data_array = np.array(png_data_list)
				png_data_list = []
				jpg_data_array = np.array(jpg_data_list)
				jpg_data_list = []
				labels = np.ones(png_data_array.shape[0])

				for i in range(0, png_data_array.shape[0], batch_size):
					_png_data_array = png_data_array[i:i+batch_size]
					_jpg_data_array = jpg_data_array[i:i+batch_size]
					_labels = labels[i:i+batch_size]

					batches += 1
					print "%d/%d"%(batches, int(png_data_array.shape[0]/batch_size))
					if batches % delta_interception == 0:
						learn_rate *= delta
						if learn_rate >= 1e-6:
							compileMutiTask(model, learn_rate)
					loss = model.train_on_batch({'png':_png_data_array, 'jpg':_jpg_data_array}, 
												 {'hr_loss':_labels, 'lr_loss_1':_labels,
												  'lr_loss_2':_labels, 'lr_loss_3':_labels,
												  'lr_loss_4':_labels, 'lr_loss_5':_labels,
												  'lr_loss_6':_labels, 'lr_loss_7':_labels,
												  'lr_loss_8':_labels})
					print loss
					ls = ''
					for l in loss:
						ls = ls + str(l) + ' '
					ls = ls + '\r'
					f_loss.writelines(ls)
					if batches % snapshot == 0:
						if opt.res_blocks == 8:
							save_path = 'model/MTRN_k8_f'+str(opt.features)+'_c'+str(channels)+'_QF'+str(QF)+'_'+str(opt.skip+batches)+'.h5'
						else:
							save_path = 'model/MT_k8_f'+str(opt.features)+'_c'+str(channels)+'_16_QF'+str(QF)+'_'+str(opt.skip+batches)+'.h5'
						model.save(save_path)

def train_MultiTaskClassicResNet_batches(QF, skip = 0):
	learn_rate = opt.learn_rate
	delta = opt.delta
	delta_interception = opt.delta_interception
	train_interception = opt.train_interception
	dataset = opt.dataset
	f_loss = open('loss_mtcrn.txt', 'wb+')

	if opt.res_blocks == 8:
		model = models.multiTaskClassicResNetModel(QF = QF)
	if opt.res_blocks == 16:
		model = models.multiTask16Model(QF = QF)
	compileMutiTask(model, learn_rate)
	if weights != '':
		model.load_weights(weights)
	if dataset == 'DIV2K':
		HR_path = opt.DIV2K_HR_path
		QF_path = opt.DIV2K_QF_path
	else:
		HR_path = 'data/'+dataset+'/'
		QF_path = HR_path+'QF_'+str(opt.QF)+'/'
	file_list = os.listdir(HR_path)
	image_list = []
	for f in file_list:
		s = str.split(f,'.')
		if len(s) == 2 and (s[1] == 'png' or s[1] == 'jpg'):
			image_list.append(f)
			if len(image_list) == 800:
				break
	
	count = 0
	batches = 0
	png_data_list = []
	jpg_data_list = []
	data_list = []

	for k in range(0, epochs):
		random.shuffle(image_list)
		for file in image_list:
			s = str.split(file,'.')
			count += 1
			print count

			img_png = Image.open(HR_path + file)
			img_jpg = Image.open(QF_path + s[0] + opt.QF_tail)
			arr_png = np.array(img_png)/255.0
			arr_jpg = np.array(img_jpg)/255.0
			if opt.random:
				x_step = int(step*random.uniform(1.5,2.5))
				y_step = int(step*random.uniform(1.5,2.5))
			else:
				x_step = step
				y_step = step
			for y in range(0,arr_png.shape[0], y_step):
				for x in range(0, arr_png.shape[1], x_step):
					if  (y+data_size) <= arr_png.shape[0] \
					and (x+data_size) <= arr_png.shape[1]:
						png_data = arr_png[y:y+data_size, x:x+data_size]
						jpg_data = arr_jpg[y:y+data_size, x:x+data_size]
						if channels == 1:
							if img_png.mode == 'RGB':
								png_data = get_Y(png_data)
							if img_jpg.mode == 'RGB':
								jpg_data = get_Y(jpg_data)
							png_data = png_data.reshape(data_size, data_size, 1)
							jpg_data = jpg_data.reshape(data_size, data_size, 1)
						data_list.append((png_data,jpg_data))

			if count % train_interception == 0:
				random.shuffle(data_list)
				for _data in data_list:
					png_data_list.append(_data[0])
					jpg_data_list.append(_data[1])
				data_list = []
				png_data_array = np.array(png_data_list)
				png_data_list = []
				jpg_data_array = np.array(jpg_data_list)
				jpg_data_list = []
				labels = np.ones(png_data_array.shape[0])

				for i in range(0, png_data_array.shape[0], batch_size):
					_png_data_array = png_data_array[i:i+batch_size]
					_jpg_data_array = jpg_data_array[i:i+batch_size]
					_labels = labels[i:i+batch_size]

					batches += 1
					print "%d/%d"%(batches, int(png_data_array.shape[0]/batch_size))
					if batches % delta_interception == 0:
						learn_rate *= delta
						if learn_rate >= 1e-6:
							compileMutiTask(model, learn_rate)
					loss = model.train_on_batch({'png':_png_data_array, 'jpg':_jpg_data_array}, 
												 {'hr_loss':_labels, 'lr_loss_1':_labels,
												  'lr_loss_2':_labels, 'lr_loss_3':_labels,
												  'lr_loss_4':_labels, 'lr_loss_5':_labels,
												  'lr_loss_6':_labels, 'lr_loss_7':_labels,
												  'lr_loss_8':_labels})
					print loss
					ls = ''
					for l in loss:
						ls = ls + str(l) + ' '
					ls = ls + '\r'
					f_loss.writelines(ls)
					if batches % snapshot == 0:
						if opt.res_blocks == 8:
							save_path = 'model/MTCRN_k8_f'+str(opt.features)+'_c'+str(channels)+'_QF'+str(QF)+'_'+str(opt.skip+batches)+'.h5'
						else:
							save_path = 'model/MT_k8_f'+str(opt.features)+'_c'+str(channels)+'_16_QF'+str(QF)+'_'+str(opt.skip+batches)+'.h5'
						model.save(save_path)

def train_ClassicResNet_batchs(QF, skip = 0):
	learn_rate = opt.learn_rate
	delta = opt.delta
	delta_interception = opt.delta_interception
	train_interception = opt.train_interception
	dataset = opt.dataset
	f_loss = open('loss_crn.txt', 'wb+')

	if opt.res_blocks == 8:
		model = models.ClassicResNetModel(QF = QF)
	if opt.res_blocks == 16:
		model = models.multiTask16Model(QF = QF)
	model.compile(loss = {'lr_loss_8':models.identity_loss},
							optimizer = Adam(lr = learn_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8))
	if weights != '':
		model.load_weights(weights)
	if dataset == 'DIV2K':
		HR_path = opt.DIV2K_HR_path
		QF_path = opt.DIV2K_QF_path
	else:
		HR_path = 'data/'+dataset+'/'
		QF_path = HR_path+'QF_'+str(opt.QF)+'/'
	file_list = os.listdir(HR_path)
	image_list = []
	for f in file_list:
		s = str.split(f,'.')
		if len(s) == 2 and (s[1] == 'png' or s[1] == 'jpg'):
			image_list.append(f)
			if len(image_list) == 800:
				break
	
	count = 0
	batches = 0
	png_data_list = []
	jpg_data_list = []
	data_list = []

	for k in range(0, epochs):
		random.shuffle(image_list)
		for file in image_list:
			s = str.split(file,'.')
			count += 1
			print count

			img_png = Image.open(HR_path + file)
			img_jpg = Image.open(QF_path + s[0] + opt.QF_tail)
			arr_png = np.array(img_png)/255.0
			arr_jpg = np.array(img_jpg)/255.0

			if opt.random:
				x_step = int(step*random.uniform(1.5,2.5))
				y_step = int(step*random.uniform(1.5,2.5))
			else:
				x_step = step
				y_step = step
			for y in range(0,arr_png.shape[0], y_step):
				for x in range(0, arr_png.shape[1], x_step):
					if  (y+data_size) <= arr_png.shape[0] \
					and (x+data_size) <= arr_png.shape[1]:
						png_data = arr_png[y:y+data_size, x:x+data_size]
						jpg_data = arr_jpg[y:y+data_size, x:x+data_size]
						if channels == 1:
							if img_png.mode == 'RGB':
								png_data = get_Y(png_data)
							if img_jpg.mode == 'RGB':
								jpg_data = get_Y(jpg_data)
							png_data = png_data.reshape(data_size, data_size, 1)
							jpg_data = jpg_data.reshape(data_size, data_size, 1)
						data_list.append((png_data,jpg_data))

			if count % train_interception == 0:
				random.shuffle(data_list)
				for _data in data_list:
					png_data_list.append(_data[0])
					jpg_data_list.append(_data[1])
				data_list = []
				png_data_array = np.array(png_data_list)
				png_data_list = []
				jpg_data_array = np.array(jpg_data_list)
				jpg_data_list = []
				labels = np.ones(png_data_array.shape[0])

				for i in range(0, png_data_array.shape[0], batch_size):
					_png_data_array = png_data_array[i:i+batch_size]
					_jpg_data_array = jpg_data_array[i:i+batch_size]
					_labels = labels[i:i+batch_size]

					batches += 1
					print "%d/%d"%(batches, int(png_data_array.shape[0]/batch_size))
					if batches % delta_interception == 0:
						learn_rate *= delta
						if learn_rate >= 1e-6:
							model.compile(loss = {'lr_loss_8':models.identity_loss},
										  optimizer = Adam(lr = learn_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8))
	
					loss = model.train_on_batch({'png':_png_data_array, 'jpg':_jpg_data_array}, 
												 {'lr_loss_8':_labels})
					print loss
					ls = ''
					ls = str(loss) + '\r'
					f_loss.writelines(ls)
					if batches % snapshot == 0:
						if opt.res_blocks == 8:
							save_path = 'model/CRN_k8_f'+str(opt.features)+'_c'+str(channels)+'_QF'+str(QF)+'_'+str(opt.skip+batches)+'.h5'
						else:
							save_path = 'model/MT_k8_f'+str(opt.features)+'_c'+str(channels)+'_16_QF'+str(QF)+'_'+str(opt.skip+batches)+'.h5'
						model.save(save_path)

def train_ResNet_batchs(QF, skip = 0):
	learn_rate = opt.learn_rate
	delta = opt.delta
	delta_interception = opt.delta_interception
	train_interception = opt.train_interception
	dataset = opt.dataset
	f_loss = open('loss_rn.txt', 'wb+')

	if opt.res_blocks == 8:
		model = models.ResNetModel(QF = QF)
	if opt.res_blocks == 16:
		model = models.multiTask16Model(QF = QF)
	model.compile(loss = {'lr_loss_8':models.identity_loss},
							optimizer = Adam(lr = learn_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8))
	if weights != '':
		model.load_weights(weights)
	if dataset == 'DIV2K':
		HR_path = opt.DIV2K_HR_path
		QF_path = opt.DIV2K_QF_path
	else:
		HR_path = 'data/'+dataset+'/'
		QF_path = HR_path+'QF_'+str(opt.QF)+'/'
	file_list = os.listdir(HR_path)
	image_list = []
	for f in file_list:
		s = str.split(f,'.')
		if len(s) == 2 and (s[1] == 'png' or s[1] == 'jpg'):
			image_list.append(f)
			if len(image_list) == 800:
				break
	
	count = 0
	batches = 0
	png_data_list = []
	jpg_data_list = []
	data_list = []

	for k in range(0, epochs):
		random.shuffle(image_list)
		for file in image_list:
			s = str.split(file,'.')
			count += 1
			print count

			img_png = Image.open(HR_path + file)
			img_jpg = Image.open(QF_path + s[0] + opt.QF_tail)
			arr_png = np.array(img_png)/255.0
			arr_jpg = np.array(img_jpg)/255.0

			if opt.random:
				x_step = int(step*random.uniform(1.5,2.5))
				y_step = int(step*random.uniform(1.5,2.5))
			else:
				x_step = step
				y_step = step
			for y in range(0,arr_png.shape[0], y_step):
				for x in range(0, arr_png.shape[1], x_step):
					if  (y+data_size) <= arr_png.shape[0] \
					and (x+data_size) <= arr_png.shape[1]:
						png_data = arr_png[y:y+data_size, x:x+data_size]
						jpg_data = arr_jpg[y:y+data_size, x:x+data_size]
						if channels == 1:
							if img_png.mode == 'RGB':
								png_data = get_Y(png_data)
							if img_jpg.mode == 'RGB':
								jpg_data = get_Y(jpg_data)
							png_data = png_data.reshape(data_size, data_size, 1)
							jpg_data = jpg_data.reshape(data_size, data_size, 1)
						data_list.append((png_data,jpg_data))

			if count % train_interception == 0:
				random.shuffle(data_list)
				for _data in data_list:
					png_data_list.append(_data[0])
					jpg_data_list.append(_data[1])
				data_list = []
				png_data_array = np.array(png_data_list)
				png_data_list = []
				jpg_data_array = np.array(jpg_data_list)
				jpg_data_list = []
				labels = np.ones(png_data_array.shape[0])

				for i in range(0, png_data_array.shape[0], batch_size):
					_png_data_array = png_data_array[i:i+batch_size]
					_jpg_data_array = jpg_data_array[i:i+batch_size]
					_labels = labels[i:i+batch_size]

					batches += 1
					print "%d/%d"%(batches, int(png_data_array.shape[0]/batch_size))
					if batches % delta_interception == 0:
						learn_rate *= delta
						if learn_rate >= 1e-6:
							model.compile(loss = {'lr_loss_8':models.identity_loss},
										  optimizer = Adam(lr = learn_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8))
	
					loss = model.train_on_batch({'png':_png_data_array, 'jpg':_jpg_data_array}, 
												 {'lr_loss_8':_labels})
					print loss
					ls = ''
					ls = str(loss) + '\r'
					f_loss.writelines(ls)
					if batches % snapshot == 0:
						if opt.res_blocks == 8:
							save_path = 'model/RN_k8_f'+str(opt.features)+'_c'+str(channels)+'_QF'+str(QF)+'_'+str(opt.skip+batches)+'.h5'
						else:
							save_path = 'model/MT_k8_f'+str(opt.features)+'_c'+str(channels)+'_16_QF'+str(QF)+'_'+str(opt.skip+batches)+'.h5'
						model.save(save_path)

def get_session(gpu_fraction = 1.0):
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_fraction)
	return tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

def train_MultiTask_PReLU_batches(QF):
	learn_rate = opt.learn_rate
	delta = opt.delta
	delta_interception = opt.delta_interception
	train_interception = opt.train_interception
	dataset = opt.dataset
	f_loss = open('loss_mt.txt', 'wb+')

	model = models.multiTaskPReLUModel(QF= QF)
	compileMutiTask(model, learn_rate)
	if weights != '':
		model.load_weights(weights)
	if dataset == 'DIV2K':
		HR_path = opt.DIV2K_HR_path
		QF_path = opt.DIV2K_QF_path
	else:
		HR_path = 'data/'+dataset+'/'
		QF_path = HR_path+'QF_'+str(opt.QF)+'/'
	file_list = os.listdir(HR_path)
	image_list = []
	for f in file_list:
		s = str.split(f,'.')
		if len(s) == 2 and (s[1] == 'png' or s[1] == 'jpg'):
			image_list.append(f)
			if len(image_list) == 800:
				break
	
	count = 0
	batches = 0
	png_data_list = []
	jpg_data_list = []
	data_list = []

	for k in range(0, epochs):
		random.shuffle(image_list)
		for file in image_list:
			s = str.split(file,'.')
			count += 1
			print count

			img_png = Image.open(HR_path + file)
			img_jpg = Image.open(QF_path + s[0] + opt.QF_tail)
			arr_png = np.array(img_png)/255.0
			arr_jpg = np.array(img_jpg)/255.0
			for y in range(0,arr_png.shape[0], y_step):
				for x in range(0, arr_png.shape[1], x_step):
					if  (y+data_size) <= arr_png.shape[0] \
					and (x+data_size) <= arr_png.shape[1]:
						png_data = arr_png[y:y+data_size, x:x+data_size]
						jpg_data = arr_jpg[y:y+data_size, x:x+data_size]
						#png_data = get_YCbCr(png_data)
						#jpg_data = get_YCbCr(jpg_data)
						if channels == 1:
							if img_png.mode == 'RGB':
								png_data = get_Y(png_data)
							if img_jpg.mode == 'RGB':
								jpg_data = get_Y(jpg_data)
							png_data = png_data.reshape(data_size, data_size, 1)
							jpg_data = jpg_data.reshape(data_size, data_size, 1)
						data_list.append((png_data,jpg_data))

			if count % train_interception == 0:
				random.shuffle(data_list)
				for _data in data_list:
					png_data_list.append(_data[0])
					jpg_data_list.append(_data[1])
				data_list = []
				png_data_array = np.array(png_data_list)
				png_data_list = []
				jpg_data_array = np.array(jpg_data_list)
				jpg_data_list = []
				labels = np.ones(png_data_array.shape[0])

				for i in range(0, png_data_array.shape[0], batch_size):
					_png_data_array = png_data_array[i:i+batch_size]
					_jpg_data_array = jpg_data_array[i:i+batch_size]
					_labels = labels[i:i+batch_size]

					batches += 1
					print "%d/%d"%(batches, int(png_data_array.shape[0]/batch_size))
					if batches % delta_interception == 0:
						learn_rate *= delta
						#if learn_rate >= 1e-6:
						compileMutiTask(model, learn_rate)
					loss = model.train_on_batch({'png':_png_data_array, 'jpg':_jpg_data_array}, 
												 {'hr_loss':_labels, 'lr_loss_1':_labels,
												  'lr_loss_2':_labels, 'lr_loss_3':_labels,
												  'lr_loss_4':_labels, 'lr_loss_5':_labels,
												  'lr_loss_6':_labels, 'lr_loss_7':_labels,
												  'lr_loss_8':_labels})
					print loss
					ls = ''
					for l in loss:
						ls = ls + str(l) + ' '
					ls = ls + '\r'
					f_loss.writelines(ls)
					if batches % snapshot == 0:
						if opt.res_blocks == 8:
							save_path = 'model/MT_PReLU_k8_f'+str(opt.features)+'_c'+str(channels)+'_QF'+str(QF)+'_'+str(opt.skip+batches)+'.h5'
						else:
							save_path = 'model/MT_PReLU_k8_f'+str(opt.features)+'_c'+str(channels)+'_16_QF'+str(QF)+'_'+str(opt.skip+batches)+'.h5'
						model.save(save_path)

if __name__ == '__main__':
	if opt.gpu == 1:		
		os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	K.tensorflow_backend.set_session(get_session(gpu_fraction = opt.memory))
	if opt.model_type == 'MultiTaskResNet':
		train_MultiTaskResNet_batches(opt.QF, skip = opt.skip)
	if opt.model_type == 'ClassicResNet':
		train_ClassicResNet_batchs(opt.QF)
	if opt.model_type == 'MultiTaskClassicResNet':
		train_MultiTaskClassicResNet_batches(opt.QF, skip = opt.skip)
	if opt.model_type == 'Diff':
		train_Diff_batches(opt.QF)
	if opt.model_type == 'MultiTaskPReLU':
		train_MultiTask_PReLU_batches(opt.QF)
	if opt.model_type == 'ResNet':
		train_ResNet_batchs(opt.QF)