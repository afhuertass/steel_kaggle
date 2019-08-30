import numpy as np 

import tensorflow as tf  
from keras.callbacks import Callback, ModelCheckpoint , EarlyStopping , ReduceLROnPlateau 
import os

"""
class SWA(keras.callbacks.Callback):
    
	def __init__(self, filepath, swa_epoch):
		super(SWA, self).__init__()
		self.filepath = filepath
		self.swa_epoch = swa_epoch 

	def on_train_begin(self, logs=None):
		self.nb_epoch = self.params['epochs']
		print('Stochastic weight averaging selected for last {} epochs.'.format(self.nb_epoch - self.swa_epoch))

	def on_epoch_end(self, epoch, logs=None):

	if epoch == self.swa_epoch:
	    self.swa_weights = self.model.get_weights()
	    
	elif epoch > self.swa_epoch:    
	    for i, layer in enumerate(self.model.layers):
	        self.swa_weights[i] = (self.swa_weights[i] * (epoch - self.swa_epoch) + self.model.get_weights()[i])/((epoch - self.swa_epoch)  + 1)  

	else:
	    pass
	        
	def on_train_end(self, logs=None):
		self.model.set_weights(self.swa_weights)
		print('Final model parameters set to stochastic weight average.')
		self.model.save_weights(self.filepath)
		print('Final stochastic averaged weights saved to file.')


""" 
def get_callbacks( args, fold  ):

	output_path = os.path.join(args.output_dir, args.backbone + "_fold_{}_.h5".format(fold) )
	print(output_path)
	callbacks = [EarlyStopping(monitor='val_loss', patience=10),
	ModelCheckpoint(filepath= output_path , monitor='val_loss', save_best_only=True , save_weights_only=True) , 
	ReduceLROnPlateau(monitor= "val_dice_coef", factor= 0.2 ,
	patience= 3,
	min_lr=1e-7 , verbose=1, mode='max' )

	]
	return callbacks 