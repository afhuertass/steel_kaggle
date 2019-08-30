import pandas as pd 


import numpy as np 
import keras 
import cv2


def mask2rle(img):
	pixels= img.T.flatten()
	pixels = np.concatenate([[0], pixels, [0]])
	runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
	runs[1::2] -= runs[::2]
	return ' '.join(str(x) for x in runs)

def rle2mask(mask_rle, shape=(256,1600)):

	s = mask_rle.split()
	starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
	starts -= 1
	ends = starts + lengths
	img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
	for lo, hi in zip(starts, ends):
		img[lo:hi] = 1
	return img.reshape(shape).T

def build_masks(rles, input_shape):
	depth = len(rles)
	height, width = input_shape
	masks = np.zeros((height, width, depth))

	for i, rle in enumerate(rles):
		if type(rle) is str:
			masks[:, :, i] = rle2mask(rle, (width, height))

	return masks

def build_rles(masks):
	width, height, depth = masks.shape

	rles = [mask2rle(masks[:, :, i])  for i in range(depth)]

	return rles

class DataGenerator(keras.utils.Sequence):

	def __init__(self, list_IDs, df, target_df=None, mode='fit',base_path='./', batch_size=32, dim=(256, 1600), n_channels=1,n_classes=4, random_state=2019, shuffle=True , aumentations = None ):

		self.dim = dim
		self.batch_size = batch_size
		self.df = df
		self.mode = mode
		self.base_path = base_path
		self.target_df = target_df
		self.list_IDs = list_IDs
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.random_state = random_state


		self.samples = len( list_IDs )
		self.au = aumentations
		self.on_epoch_end()
		print( target_df.columns )
		print( "*"*10 )


	def _read_image_train(self , image_id , train = True ):

		image_name = self.df['ImageId'].iloc[ image_id ]
		img_path = f"{self.base_path}/{image_name}"
		img = self.__load_grayscale( img_path )
		#img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
		#img = img.astype(np.float32) / 255
		#img = np.expand_dims(img, axis=-1)

		if train:

			mask_pix = self.target_df[self.target_df['ImageId'] == image_name ]

			rles = mask_pix['EncodedPixels'].values
			masks = build_masks(rles, input_shape=self.dim )
			#print( masks.shape )
			#print( "asdasdasdasdasd")

			if self.au:

				data = { "image" : img , "mask" : masks }
				augmented = self.au( **data )
				img , mask = augmented["image"] , augmented["mask"]


				return img , mask
			else:
				# no aumentations in testing time 
				return img , masks 


	def __len__(self):
		
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_batch = [self.list_IDs[k] for k in indexes]
		#print( list_IDs_batch )

		if self.mode == 'fit':
			X , y = self._generate_batch( list_IDs_batch , True )

			return X, y

		elif self.mode == 'predict':
			X = self._generate_batch( list_IDs_batch , False )
			return X

		else:
			raise AttributeError('The mode parameter should be set to "fit" or "predict".')
        
	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.seed(self.random_state)
			np.random.shuffle(self.indexes)

	def _generate_batch( self ,list_IDs_batch , train = True  ):

		X = np.empty((self.batch_size, *self.dim, self.n_channels)) 
		y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

		for i  , ID in enumerate( list_IDs_batch ):

			img , mask = self._read_image_train( ID , train  )

			X[i , :] = img 
			y[i , :] = mask 

		#print( X.shape )
		#print( y.shape )
		return X , y

	def __load_grayscale(self, img_path):

		img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
		img = img.astype(np.float32) / 255.
		img = np.expand_dims(img, axis=-1)

		return img