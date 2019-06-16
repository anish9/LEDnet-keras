from led import *

def LEDnet(classes):
    inpu = Input(shape=(512,512,3))
    encode = encoder(inpu)
    dec = decoder(encode,classes)
    comp = Model(inputs=inpu,outputs=dec)
    return comp
