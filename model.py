

import pandas as pd 
import numpy as np 
import segmentation_models as sm

from segmentation_models.losses import bce_jaccard_loss 
from segmentation_models.metrics import iou_score
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose


def get_model_unet( backbone = "resnet50" , input_shape = ( 256, 1600 , 3 ) , classes = 4  , freeze_encoder = True  ):

	# return a keras model 

	model = sm.Unet( backbone , input_shape=input_shape, encoder_weights='imagenet' , classes=classes , freeze_encoder = freeze_encoder )

	inp = Input(shape=(256, 1600, 1))
	l1 = Conv2D(  3 , (1, 1))(inp) # map N channels data to 3 channels
	out = model(l1)

	model_final = Model(inp, out, name=model.name)

	return model_final  