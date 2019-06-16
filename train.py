
from model import LEDnet
from config import train_gens,train_steps,val_gens,val_steps
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import *


GPU = 1 #count
CLASSES = 3
Tensorboard_dir = "./logs"
Merge_Flag = True 

build = LEDnet(CLASSES)

multi = build
tb = TensorBoard(Tensorboard_dir)
mc = ModelCheckpoint("best_weight.h5",monitor="val_acc",save_best_only=True,verbose=1)
es = EarlyStopping(monitor="val_acc",patience=10,verbose=1)
callback = [tb,mc,es]

multi.compile(optimizer=Nadam(0.00004),loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("---DATA LOADED SUCCESSFULLY--- ")
print("Param Initalized")
print("Training...")

multi.fit_generator(train_gens,steps_per_epoch=int(train_steps),validation_data=val_gens,validation_steps=val_steps,
                    epochs=50,callbacks=callback)

multi.save("model.h5")
