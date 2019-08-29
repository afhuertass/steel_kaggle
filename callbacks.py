import numpy as np 

import tensorflow as tf  
from keras.callbacks import Callback, ModelCheckpoint , EarlyStopping , ReduceLROnPlateau 
import os 
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