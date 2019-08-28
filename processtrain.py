import pandas as pd 

import numpy as np 
import utils 

from sklearn.model_selection import KFold

def generate_folds( train_df , folds = 5  ):


	# returns data trafe with an aditional column indicating which fold the image belongs 
	
	train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
	train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])
	train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()

	mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()
	mask_count_df.sort_values('hasMask', ascending=False, inplace=True)
	#print(mask_count_df.shape)
	#mask_count_df.head()

	# mask_count_df 
	mask_count_df = mask_count_df[ mask_count_df["hasMask"] > 0 ]
	mask_count_df["hasMask"] = mask_count_df["hasMask"] - 1 

	kf =  KFold(n_splits= folds , random_state = 666 )

	mask_count_df["fold"] = 0
	f = 0 
	dfs = [ ]
	for train_index ,  test_index in kf.split( mask_count_df ): 

		temp = mask_count_df.iloc[ train_index ]
		temp.loc[ : , "fold"] = f
		dfs.append( temp )
		#print( dd.head() )
		f = f + 1
	df = pd.concat( dfs )
	
	return  df  