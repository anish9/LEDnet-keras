import numpy as np
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import add
import tensorflow.keras.backend as K



def resdiual_block(in_tensor,filters,stage,block):
    conv = "block2_conv"
    bnorm =  "block2_bn"
    filter1,filter2,filter3 = filters
    x = Conv2D(filter1,(1,1),name=conv+stage+block+"a",padding="same")(in_tensor)
    x = BatchNormalization(axis=3,name=bnorm+stage+block+"a")(x)
    x = Activation("relu")(x)
    x = Conv2D(filter2,(3,3),name=conv+stage+block+"b",padding="same")(x)
    x = BatchNormalization(axis=3,name=bnorm+stage+block+"b")(x)
    x = Activation("relu")(x)
    x = Conv2D(filter3,(1,1),name=conv+stage+block+"c",strides=(2,2),padding="same")(x)
    x = BatchNormalization(axis=3,name=bnorm+stage+block+"c")(x)
    x = Activation("relu")(x)
    shortcut = Conv2D(filter3,(1,1),padding="same",strides=(2,2),name="shortcut"+conv+stage+block)(in_tensor)
    shortcut = BatchNormalization(axis=3,name="shortcut_bn"+stage+block)(shortcut)
    shortcut = add([x,shortcut])
    shortcut = Activation("relu")(shortcut)
    return shortcut


def downsampling(in_tensor,filter_,block):
    tname = "downsample_layer"+str(block)
    c1 = Conv2D(filter_,3,strides=(2,2),padding="same",name=tname+"conv_3x3")(in_tensor)
    c1 = BatchNormalization(axis=-1)(c1)
    c1 = Activation("relu")(c1)
    return c1


def resizer(x,size,dataformat = "channels_last"):
    res = tf.image.resize_images(x,size,align_corners=True)
    return res

def resizer_block(tensor,size):
    layer = Lambda(lambda x : resizer(x,size))(tensor)
    return layer


def channel_split(x):
    split1,split2 = tf.split(x,num_or_size_splits=2,axis=-1)
    return split1,split2

def splitter(tensor):
    layer = Lambda(lambda x : channel_split(x))(tensor)
    return layer


def nnbt_conv(in_tensor1,in_tensor2,block,filter_,dilation_rate=None):
    tname = "NNBT_channel_id_"+str(block)
    c1 = Conv2D(filter_,(3,1),padding="same",name=tname+"path1_conv1")(in_tensor1)
    c1 = Activation("relu")(c1)
    c1 = Conv2D(filter_,(1,3),padding="same",name=tname+"path1_conv2")(c1)
    c1 = BatchNormalization(axis=-1,name = tname+"bn_"+"path1_bn1")(c1)
    c1 = Activation("relu")(c1)
    c1 = Conv2D(filter_,(3,1),padding="same",name=tname+"path1_conv3")(c1)
    c1 = Activation("relu")(c1)
    c1 = Conv2D(filter_,(1,3),padding="same",name=tname+"path1_conv4")(c1)
    c1 = BatchNormalization(axis=-1,name = tname+"bn_"+"path1_bn2")(c1)
    c1 = Activation("relu")(c1)
    c2 = Conv2D(filter_,(1,3),padding="same",name=tname+"path2_conv1")(in_tensor2)
    c2 = Activation("relu")(c2)
    c2 = Conv2D(filter_,(3,1),padding="same",name=tname+"path2_conv2")(c2)
    c2 = BatchNormalization(axis=-1,name = tname+"bn_"+"path2_bn1")(c2)
    c2 = Activation("relu")(c2)
    c2 = Conv2D(filter_,(1,3),padding="same",name=tname+"path2_conv3")(c2)
    c2 = Activation("relu")(c2)
    c2 = Conv2D(filter_,(3,1),padding="same",name=tname+"path2_conv4")(c2)
    c2 = BatchNormalization(axis=-1,name = tname+"bn_"+"path2_bn2")(c2)
    c2 = Activation("relu")(c2)
    
    if dilation_rate:
        c1 = Conv2D(filter_,(3,1),padding="same",name=tname+"path1_conv1")(in_tensor1)
        c1 = Activation("relu")(c1)
        c1 = Conv2D(filter_,(1,3),padding="same",name=tname+"path1_conv2")(c1)
        c1 = BatchNormalization(axis=-1,name = tname+"bn_"+"path1_bn1")(c1)
        c1 = Activation("relu")(c1)
        c1 = Conv2D(filter_,(3,1),padding="same",dilation_rate=dilation_rate,name=tname+"path1_dil_conv3")(c1)
        c1 = Activation("relu")(c1)
        c1 = Conv2D(filter_,(1,3),padding="same",dilation_rate=dilation_rate,name=tname+"path1_dil_conv4")(c1)
        c1 = BatchNormalization(axis=-1,name = tname+"bn_"+"path1_bn2")(c1)
        c1 = Activation("relu")(c1)
        c2 = Conv2D(filter_,(1,3),padding="same",name=tname+"path2_conv1")(in_tensor2)
        c2 = Activation("relu")(c2)
        c2 = Conv2D(filter_,(3,1),padding="same",name=tname+"path2_conv2")(c2)
        c2 = BatchNormalization(axis=-1,name = tname+"bn_"+"path2_bn1")(c2)
        c2 = Activation("relu")(c2)
        c2 = Conv2D(filter_,(1,3),padding="same",dilation_rate=dilation_rate,name=tname+"path2_dil_conv3")(c2)
        c2 = Activation("relu")(c2)
        c2 = Conv2D(filter_,(3,1),padding="same",dilation_rate=dilation_rate,name=tname+"path2_dil_conv4")(c2)
        c2 = BatchNormalization(axis=-1,name = tname+"bn_"+"path2_bn2")(c2)
        c2 = Activation("relu")(c2)

    concat_layer = concatenate([c1,c2],axis=-1)
    concat_layer = Conv2D(256,1,padding="same")(concat_layer)
    concat_layer = BatchNormalization(axis=-1)(concat_layer)
    concat_layer = Activation("relu")(concat_layer)
    return concat_layer
    

def nnbt(in_tensor,block,filt_val,dilation_rate):
    ch0,ch1 = splitter(in_tensor)
    nconv = nnbt_conv(ch0,ch1,block,filt_val,dilation_rate)
    skip = Conv2D(K.int_shape(nconv)[3],1,padding="same")(in_tensor)
    skip_layer = add([skip,nconv])
    skip_layer = Conv2D(256,1,padding="same")(skip_layer)
    skip_layer = Activation("relu")(skip_layer)
    skip_layer = BatchNormalization(axis=-1)(skip_layer)
    return skip_layer

def encoder(inpu):
    ds1 = downsampling(inpu,32,1)
    nb1 = nnbt(ds1,1,16,dilation_rate=None)
    nb2 = nnbt(nb1,2,16,dilation_rate=None)
    nb3 = nnbt(nb2,3,16,dilation_rate=None)
    ds2 = downsampling(nb3,64,2)
    nb4 = nnbt(ds2,4,32,dilation_rate=None)
    nb5 = nnbt(nb4,5,32,dilation_rate=None)
    ds3 = downsampling(nb5,128,3)
    nb6 = nnbt(ds3,6,64,dilation_rate=1)
    nb7 = nnbt(nb6,7,64,dilation_rate=2)
    nb8 = nnbt(nb7,8,64,dilation_rate=5)
    nb9 = nnbt(nb8,9,64,dilation_rate=9)
    nb10 = nnbt(nb9,10,64,dilation_rate=2)
    nb11 = nnbt(nb10,11,64,dilation_rate=5)
    nb12 = nnbt(nb11,12,64,dilation_rate=9)
    nb13 = nnbt(nb12,13,64,dilation_rate=17)
    return nb13

def reshape(tensor,shape):
    la = Lambda(lambda x : K.reshape(x,shape))(tensor)
    return la


def expand(tensor,ax):
    la = Lambda(lambda x : K.expand_dims(x,axis=ax))(tensor)
    return la

def decoder(inpu_tensor,classes):
    
    
    mtensor_1,mtensor_2 = K.int_shape(inpu_tensor)[1],K.int_shape(inpu_tensor)[2]
#     print(mtensor_1,mtensor_2)
    pooled_enco = GlobalAveragePooling2D()(inpu_tensor)
    pooled_enco = expand(pooled_enco,ax=1)
    pooled_enco = expand(pooled_enco,ax=1)
    cc = resizer_block(pooled_enco,(mtensor_1,mtensor_2))
    cc = Conv2D(classes,3,padding="same")(cc)
    cc = BatchNormalization(axis=-1)(cc)
    cc = Activation("relu")(cc)

    
    base_patch= Conv2D(classes,3,padding="same",name="base_patch_conv")(inpu_tensor)
    base_patch = Activation("relu")(base_patch)
    
    c3 = Conv2D(128,3,strides=(2,2),padding="same",name="feature_conv3x3")(inpu_tensor)
    c3 = BatchNormalization(axis=-1)(c3)
    c3 = Activation("relu")(c3)
    class_contextc3 = Conv2D(classes,1,padding="same",name="feature3_convclass1x1")(c3)
    class_contextc3 = BatchNormalization(axis=-1)(class_contextc3)
    class_contextc3 = Activation("relu")(class_contextc3)
    
    
    c5 = Conv2D(128,5,strides=(2,2),padding="same",name="feature_conv5x5")(c3)
    c5 = BatchNormalization(axis=-1)(c5)
    c5 = Activation("relu")(c5)
    class_contextc5 = Conv2D(classes,1,padding="same",name="feature5_convclass1x1")(c5)
    class_contextc5 = BatchNormalization(axis=-1)(class_contextc5)
    class_contextc5 = Activation("relu")(class_contextc5)
    
    
    c7 = Conv2D(128,7,strides=(2,2),padding="same",name="feature_conv7x7")(c5)
    c7 = BatchNormalization(axis=-1)(c7)
    c7 = Activation("relu")(c7)
    class_contextc7 = Conv2D(classes,1,padding="same",name="feature7_convclass1x1")(c7)
    class_contextc7 = BatchNormalization(axis=-1)(class_contextc7)
    class_contextc7 = Activation("relu")(class_contextc7)
    
    up_dim1,up_dim2 = 2*K.int_shape(class_contextc7)[1],2*K.int_shape(class_contextc7)[2]
    
    upcon_7 = resizer_block(class_contextc7,(up_dim1,up_dim2))
    up_sum = add([upcon_7,class_contextc5])
    up1_dim1,up2_dim2 = 2*K.int_shape(up_sum)[1],2*K.int_shape(up_sum)[2]
    upcon_8 = resizer_block(up_sum,(up1_dim1,up2_dim2))
    up1_sum = add([upcon_8,class_contextc3])
    up2_dim1,up2_dim2 = 2*K.int_shape(up1_sum)[1],2*K.int_shape(up1_sum)[2]
    upcon_9 = resizer_block(up1_sum,(up2_dim1,up2_dim2))
    
    patch1_merger = multiply([upcon_9,base_patch])
    
    final_merge = add([patch1_merger,cc])
    
    final_merge = resizer_block(final_merge,(512,512))
    final_merge = Conv2D(classes,1,padding="same")(final_merge)
    final_merge = Activation("softmax")(final_merge)
    return final_merge
