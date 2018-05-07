################ model parameter ##################
data_size = 48
step = data_size
step = 25
features = 256
channels = 3 
QF = 40
res_blocks = 8
model_type = 'MultiTaskResNet'		# MultiTask/Resnet/MultiTaskResNet/MultiTaskDiff/Tiny/Max
##################################################

################ leraning paramter ###############
learn_rate = 1e-4
delta = 1/2
delta_interception = 10000
train_interception = 200
epochs = 100
batch_size = 16
random = True
##################################################

################## train parameter ###############
gpu = 1
memory = 0.3
dataset = 'DIV2K'#'DIV2K'
QF_tail = '.jpg'
DIV2K_ROOT_path = '/media/scs4450/hard/zbl/srcnn/src/train/DIV2K/'
DIV2K_HR_path = DIV2K_ROOT_path + 'DIV2K_train_HR/'
DIV2K_QF_path = DIV2K_ROOT_path + 'DIV2K_QF_' + str(QF) + '/'
skip = 0
weights = ''#'model/QF_40/MT_k8_f256_c3_QF40_avarage.h5'#'model/MT_k8_f256_c3_QF40_avarage.h5'#'model/MT_k8_f256_c3_QF40_800.h5'
snapshot = 10000
meanVec = [0.4488, 0.4371, 0.4040]
ratio = 2 	#for SR
################## compare parameter ############### 