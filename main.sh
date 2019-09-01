

## this script it to be run after the Conda installation is working. 
mkdir ~/.kaggle
cp ./kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
mkdir data 
kaggle competitions download -c severstal-steel-defect-detection -p ./data 

cd data 
unzip "train.csv.zip"
mkdir  train_images
cp train_images.zip ./train_images/train_images.zip
cd train_images 
unzip train_images.zip 
cd .. 
mkdir output

# directory structure :
# code
# data
# output
python train.py --train-data ./data/train.csv /
--images-path ./data/train_images --output-dir ./output --folds 5 --backbone resnet50 --lr 0.000001 --batch-size 8 /
--earlystopping-patience 20 --loss "bce" --epochs 100 --swa_epoch 20 



