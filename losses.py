
import numpy as np 
from keras.losses import binary_crossentropy
from keras import backend as K

def pick_loss( loss ):

	if loss == "bce":
		return bce_dice_loss 
	if loss == "focal":
		return focal_loss

def focal_loss( y_true , y_pred ):

	return loss  

def dice_loss(y_true, y_pred):
	smooth = 1.
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = y_true_f * y_pred_f
	score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
	return 1. - score

def bce_dice_loss(y_true, y_pred):
	return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1):

	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

