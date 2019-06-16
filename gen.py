
import os
import cv2
from tqdm import tqdm
from random import shuffle
import numpy as np


from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomGamma    
)


def generate_flow(data_dir,Timage,Tmask,batch,size,augument=False,png=False): 
    images_ = os.listdir(data_dir+Timage)
    shuffle(images_)
    ids_int = list(range(len(images_)))
    while True:
        for start in range(0,len(ids_int),batch):
            x_batch = []
            y_batch = []
            end = min(start+batch,len(images_))
            batch_create = ids_int[start:end]
            for loads in batch_create:
                try:
                    img = cv2.imread(os.path.join(data_dir,Timage,images_[loads]))
                    #print((os.path.join(data_dir,Timage,images_[loads])))
                    img = cv2.resize(img,size)
                    if png:
                        try:
                            masks = cv2.imread(os.path.join(data_dir,Tmask,images_[loads]).replace(".jpeg",".png"),0)
                            masks = cv2.resize(masks,size)
                            masks = np.expand_dims(masks,axis=2)
                        except:
                            masks = cv2.imread(os.path.join(data_dir,Tmask,images_[loads]).replace(".jpg",".png"),0)
                            masks = cv2.resize(masks,size)
                            masks = np.expand_dims(masks,axis=2)
                        #print(os.path.join(data_dir,Tmask,images_[loads]).replace(".jpg",".png"))
                    else:
                        masks = cv2.imread(os.path.join(data_dir,Tmask,images_[loads]),0)
                        masks = cv2.resize(masks,size)
                        masks = np.expand_dims(masks,axis=2)
                except:
                    continue
                if augument:
                    aug = Compose([VerticalFlip(p=0.1),Transpose(p=0.01),RandomGamma(p=0.06),OpticalDistortion(p=0.00, distort_limit=0.7, shift_limit=0.3)])
                    augmented = aug(image=img, mask=masks)
                    img = augmented['image']
                    masks = augmented['mask']
                    x_batch.append(img)
                    y_batch.append(masks)
                else:
                    x_batch.append(img)
                    y_batch.append(masks)
            x_batch = np.array(x_batch,np.float32) / 255.
            y_batch = np.array(y_batch,np.float32)
            yield x_batch,y_batch

def gen_conf(data_dir,Timage,batch): 
	im_count = os.listdir(data_dir+Timage)
	id_count = len(im_count)/batch
	return id_count
