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
import callbacks 

from keras import backend as K
import gc
import os 
parser = argparse.ArgumentParser(description='Segmentation code for the Seversteel competition' )

parser.add_argument( '--train-data' , default = "../data/train.csv"  ) # location of the df holding the data
parser.add_argument('--images-path' , default = "../data/train_images")
parser.add_argument( '--output-dir' , default = "../output/")
parser.add_argument( '--folds' , default = 5 , type=int) # number of folds 
parser.add_argument('--backbone' , default = "resnet50")
parser.add_argument( '--lr' , default = 0.000001 , type = float )
parser.add_argument( '--batch-size' , default = 2 , type=int )
parser.add_argument( '--earlystopping-patience' , default = 20 , type=int )
parser.add_argument('--reduce-lr-factor' , default = 0.25 , type=float)
parser.add_argument('--loss' , default = "bce" )
parser.add_argument('--pretrain-weights' , default = None )
parser.add_argument('--epochs' , default = 100 ,type=int)
parser.add_argument('--swa_epoch' , default = 10 , type=int)
results = parser.parse_args()

def prepare_df( df ):

	df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
	df['ClassId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])
	df['hasMask'] = ~ df['EncodedPixels'].isna()
	return df

def main():

	print( results )
	df_train = pd.read_csv( results.train_data )
	df_train = prepare_df( df_train )
	df_train = df_train[:1000]
	df_target = df_train.copy()



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
		generator_train = generator.DataGenerator( list_IDs = train.index, df = train , target_df = df_target ,  batch_size = results.batch_size , aumentations = au , base_path =   results.images_path , mode = "fit" )
		generator_valid = generator.DataGenerator( list_IDs= valid.index,  df = valid , target_df = df_target  , batch_size = results.batch_size, aumentations = au , base_path = results.images_path , mode = "fit" )

		call_bks = callbacks.get_callbacks( results , fold , generator_train.samples    ) # get the callbacks 

		model.fit_generator( 

			generator = generator_train , 
			validation_data = generator_valid , 
			epochs = results.epochs , 
			steps_per_epoch= generator_train.samples // results.batch_size,   
			validation_steps = generator_valid.samples  // results.batch_size , 
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