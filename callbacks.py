import numpy as np 

import tensorflow as tf  
from keras.callbacks import Callback, ModelCheckpoint , EarlyStopping , ReduceLROnPlateau 
import os
from math import pi
from math import cos
from math import floor
from keras import backend as K


class SWA(Callback):

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


# define custom learning rate schedule
class CosineAnnealingLearningRateSchedule(Callback):
	# constructor
	def __init__(self, training_steps , n_cycles = 5, lrate_max = 0.001, verbose=0):
		# training steps number of steps per epoch 
		self.training_steps = training_steps
		self.cycles = 1
		#self.cycles = n_cycles
		self.lr_max = lrate_max
		self.lrates = list()
		self.trainig_count = 0 

	# calculate learning rate for an epoch
	def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
		epochs_per_cycle = floor(n_epochs/n_cycles)
		cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
		return lrate_max/2 * (cos(cos_inner) + 1)

	# calculate and set learning rate at the start of the epoch
	def on_epoch_end( self , epoch, logs=None):

		self.training_count = 0 

	def on_train_batch_begin(self, batch, logs=None ):
		logs = logs or {}

		if self.trainig_count == 0:
			K.set_value(self.model.optimizer.lr, self.lr_max )
			self.training_count += 1 
		else:
			K.set_value(self.model.optimizer.lr, self.cosine_annealing(   self.trainig_count , self.training_steps , self.cycles , self.lr_max    ))
			self.trainig_count += 1 


class CyclicLR(Callback):
	"""This callback implements a cyclical learning rate policy (CLR).
	The method cycles the learning rate between two boundaries with
	some constant frequency.
	# Arguments
	    base_lr: initial learning rate which is the
	        lower boundary in the cycle.
	    max_lr: upper boundary in the cycle. Functionally,
	        it defines the cycle amplitude (max_lr - base_lr).
	        The lr at any cycle is the sum of base_lr
	        and some scaling of the amplitude; therefore
	        max_lr may not actually be reached depending on
	        scaling function.
	    step_size: number of training iterations per
	        half cycle. Authors suggest setting step_size
	        2-8 x training iterations in epoch.
	    mode: one of {triangular, triangular2, exp_range}.
	        Default 'triangular'.
	        Values correspond to policies detailed above.
	        If scale_fn is not None, this argument is ignored.
	    gamma: constant in 'exp_range' scaling function:
	        gamma**(cycle iterations)
	    scale_fn: Custom scaling policy defined by a single
	        argument lambda function, where
	        0 <= scale_fn(x) <= 1 for all x >= 0.
	        mode paramater is ignored
	    scale_mode: {'cycle', 'iterations'}.
	        Defines whether scale_fn is evaluated on
	        cycle number or cycle iterations (training
	        iterations since start of cycle). Default is 'cycle'.
	The amplitude of the cycle can be scaled on a per-iteration or
	per-cycle basis.
	This class has three built-in policies, as put forth in the paper.
	"triangular":
	    A basic triangular cycle w/ no amplitude scaling.
	"triangular2":
	    A basic triangular cycle that scales initial amplitude by half each cycle.
	"exp_range":
	    A cycle that scales initial amplitude by gamma**(cycle iterations) at each
	    cycle iteration.
	For more detail, please see paper.
	# Example for CIFAR-10 w/ batch size 100:
	    ```python
	        clr = CyclicLR(base_lr=0.001, max_lr=0.006,
	                            step_size=2000., mode='triangular')
	        model.fit(X_train, Y_train, callbacks=[clr])
	    ```
	Class also supports custom scaling functions:
	    ```python
	        clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
	        clr = CyclicLR(base_lr=0.001, max_lr=0.006,
	                            step_size=2000., scale_fn=clr_fn,
	                            scale_mode='cycle')
	        model.fit(X_train, Y_train, callbacks=[clr])
	    ```
	# References
	  - [Cyclical Learning Rates for Training Neural Networks](
	  https://arxiv.org/abs/1506.01186)
	"""
	def __init__(self,base_lr=0.001,max_lr=0.006,step_size=2000.,mode='triangular',gamma=1.,scale_fn=None,scale_mode='cycle'):
		super(CyclicLR, self).__init__()
		if mode not in ['triangular', 'triangular2','exp_range']:
			raise KeyError("mode must be one of 'triangular', 'triangular2', or 'exp_range'")
		self.base_lr = base_lr
		self.max_lr = max_lr
		self.step_size = step_size
		self.mode = mode
		self.gamma = gamma
		if scale_fn is None:
			if self.mode == 'triangular':
				self.scale_fn = lambda x: 1.
				self.scale_mode = 'cycle'
			elif self.mode == 'triangular2':
				self.scale_fn = lambda x: 1 / (2.**(x - 1))
				self.scale_mode = 'cycle'
			elif self.mode == 'exp_range':
				self.scale_fn = lambda x: gamma ** x
				self.scale_mode = 'iterations'
		else:
			self.scale_fn = scale_fn
			self.scale_mode = scale_mode
		self.clr_iterations = 0.
		self.trn_iterations = 0.
		self.history = {}

		self._reset()

	def _reset(self, new_base_lr=None, new_max_lr=None,new_step_size=None):
		"""Resets cycle iterations.
		Optional boundary/step size adjustment.
		"""
		if new_base_lr is not None:
			self.base_lr = new_base_lr
		if new_max_lr is not None:
			self.max_lr = new_max_lr
		if new_step_size is not None:
			self.step_size = new_step_size
			self.clr_iterations = 0.

	def clr(self):
		cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
		x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
		if self.scale_mode == 'cycle':
			return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
		else:
			return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)

	def on_train_begin(self, logs={}):
		logs = logs or {}

		if self.clr_iterations == 0:
			K.set_value(self.model.optimizer.lr, self.base_lr)
		else:
			K.set_value(self.model.optimizer.lr, self.clr())

	def on_batch_end(self, epoch, logs=None):

		logs = logs or {}
		self.trn_iterations += 1
		self.clr_iterations += 1
		K.set_value(self.model.optimizer.lr, self.clr())

		self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
		self.history.setdefault('iterations', []).append(self.trn_iterations)

		for k, v in logs.items():
			self.history.setdefault(k, []).append(v)

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}
		logs['lr'] = K.get_value(self.model.optimizer.lr)

def get_callbacks( args, fold, training_steps  ):

	output_path = os.path.join(args.output_dir, args.backbone + "_fold_{}_.h5".format(fold) )
	output_path_swa = os.path.join(args.output_dir, args.backbone + "_swa.h5" )
	print(output_path)
	cosine_callback = CosineAnnealingLearningRateSchedule( training_steps  = training_steps )

	swa_callback = SWA( filepath = output_path_swa , swa_epoch = args.swa_epoch  )
	callbacks = [EarlyStopping(monitor='val_loss', patience=10),
	ModelCheckpoint(filepath= output_path , monitor='val_loss', save_best_only=True , save_weights_only=True) , 
	cosine_callback , 
	swa_callback 
	]
	return callbacks

"""
	ReduceLROnPlateau(monitor= "val_dice_coef", factor= 0.2 ,
	patience= 3,
	min_lr=1e-7 , verbose=1, mode='max' )
""" 
