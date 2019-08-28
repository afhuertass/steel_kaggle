import numpy as np 

import tensorflow as tf  


def get_callbacks( args ):

		callbacks = [EarlyStopping(monitor='val_loss', patience=10),
		ModelCheckpoint(filepath= drive_path+ 'model_vg19.h5', monitor='val_loss', save_best_only=True , save_weights_only=True) , 
		ReduceLROnPlateau(monitor= "val_dice_coef", factor= 0.2 ,
		patience= 3,
		min_lr=1e-6 , verbose=1, mode='max' )

		]
	return callbacks 