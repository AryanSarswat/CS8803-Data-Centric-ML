# CS8803-Data-Centric-ML

Selection-via-Pruning

## Datasets
Download datasets with git via the commands below  
Takenote to download into scratch folder, not home folder on PACE-ICE 
`cd $DATA_FOLDER`
### CC595K 
```
git lfs install  
git clone git@hf.co:datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K  
```
### ImageNet
```
mkdir ImageNet  
cd ImageNet  
wget https://github.com/0429charlie/ImageNet_metadata/raw/master/ILSVRC2012_devkit_t12.tar.gz  
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar  
wget https://github.com/raghakot/keras-vis/raw/refs/heads/master/resources/imagenet_class_index.json  
```

# Run
`python main.py --data_name imagenet --data_folder $DATA_FOLDER`
