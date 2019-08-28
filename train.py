import pandas as pd 
import numpy as np 

import tensorflow as tf 
from keras.optimizers import RMSprop

import model as modelgenerator
import losses
import generator 
import processtrain 
import argparse 
import utils 
import aumentations 

from keras import backend as K
import gc
import os 
parser = argparse.ArgumentParser(description='Segmentation code for the Seversteel competition' )

parser.add_argument( '--train-data' , default = "../data/train.csv"  ) # location of the df holding the data
parser.add_argument('--images-path' , default = "../data/train_images")
parser.add_argument( '--output-dir' , default = "../output/")
parser.add_argument( '--folds' , default = 5 ) # number of folds 
parser.add_argument('--backbone' , default = "resnet50")
parser.add_argument( '--lr' , default = 0.0001)
parser.add_argument( '--batch-size' , default = 8 )
parser.add_argument( '--earlystopping-patience' , default = 20  )
parser.add_argument('--reduce-lr-factor' , default = 0.25)
parser.add_argument('--loss' , default = "bce")
parser.add_argument('--pretrain-weights' , default = None )
parser.add_argument('--epochs' , default = 100 )
results = parser.parse_args()


def main():


	df_train = pd.read_csv( results.train_data )
	df_train = df_train[:1000]
	df_train = processtrain.generate_folds( df_train  , folds = results.folds )

	print( results.output_dir )
	model_path = os.path.join(results.output_dir, results.backbone )

	K.clear_session()


	#
	for fold in range( results.folds ) : 
		# fold is just and interger 

		train = df_train[ df_train["fold"] != fold ].copy()
		valid = df_train[ df_train["fold"] == fold ].copy()
		print( train.shape )
		print( valid.shape )
		#df_train = None

		print( "Training on : {} samples".format( len( train )) )
		print( "validating on: {}  samples".format( len(valid )) )



		model = modelgenerator.get_model_unet(  backbone = results.backbone , freeze_encoder = True )

		opt = RMSprop( lr = results.lr  )

		model.compile( optimizer = opt , loss = losses.pick_loss( results.loss ) , metrics = [ losses.dice_coef ] )

		
		if results.pretrain_weights is None:

			print( "training from scratch ")

		else:

			model.load_weights( results.pretrain_weights )

		au = aumentations.get_augmentations("valid")
		generator_train = generator.DataGenerator( train.index, train , df_train ,  results.batch_size , aumentations = au , base_path =   results.imgs_path )
		generator_valid = generator.DataGenerator( valid.index,  valid , df_train  , results.batch_size, aumentations = au  , base_path = results.imgs_path )

		call_bks = callbacks.get_callbacks( results  ) # get the callbacks 

		model.fit_generator( 

			generator = generator_train , 
			validation_data = generator_valid , 
			epochs = results.epochs , 
			steps_per_epoch= generator_train.steps_generator ,   
			validation_steps = generator_valid.steps_generator , 
			callbacks = call_bks 

			) 

		gc.collect( )

		## TODO FROM HERE 

		
	# train the whole thing 

	# get the folds
	# get the data
	# create generators 
	#build the model
	#train the model
	# save the model 

	#log results

	return ""

if __name__== "__main__":

	main()