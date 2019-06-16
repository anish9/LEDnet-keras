
from gen import gen_conf,generate_flow
import cv2
import numpy as np




data_dir =  "/home/anish/datadir/" #Root training and Val data dir 
train_image = "train_images"
val_images = "val_images"
train_masks = "train_masks"
val_masks = "val_masks"

#do not change below directory to avoid directory conflicts 


model_param = {"image_size" : (512,512),
			   "train_batch_size" : 2,
			   "val_batch_size" : 2,
			   "augument":False
			   }


train_gens = generate_flow(data_dir,train_image,train_masks,model_param["train_batch_size"],model_param["image_size"],png=False,augument=model_param["augument"])
train_steps = gen_conf(data_dir,train_image,model_param["train_batch_size"])
val_gens = generate_flow(data_dir,val_images,val_masks,model_param["val_batch_size"],model_param["image_size"],png=False)
val_steps = gen_conf(data_dir,val_images,model_param["val_batch_size"])
