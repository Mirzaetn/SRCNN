from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
import tensorflow as tf


def SRCNN915():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=64, kernel_size=9, padding='valid', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=32, kernel_size=1, padding='valid', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3,  kernel_size=5, padding='valid',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)
    
    return Model(X_in, X_out, name="SRCNN915")

def SRCNN935():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=64, kernel_size=9, padding='valid', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=32, kernel_size=3, padding='valid', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3,  kernel_size=5, padding='valid',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)

    return Model(X_in, X_out, name="SRCNN935")

def SRCNN955():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=64, kernel_size=9, padding='valid', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=32, kernel_size=5, padding='valid', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3,  kernel_size=5, padding='valid',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)

    return Model(X_in, X_out, name="SRCNN955")

def SRCNN18210():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=128, kernel_size=18, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3,  kernel_size=10, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)
    
    return Model(X_in, X_out, name="SRCNN18210")

def SRCNN18610():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=128, kernel_size=18, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=64, kernel_size=6, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3,  kernel_size=10, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)
    
    return Model(X_in, X_out, name="SRCNN18610")

def SRCNN181010():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=128, kernel_size=18, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=64, kernel_size=10, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3,  kernel_size=10, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)
    
    return Model(X_in, X_out, name="SRCNN181010")

def SRCNN1895210():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=128, kernel_size=18, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=64, kernel_size=9, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=32, kernel_size=5, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=16, kernel_size=2, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3,  kernel_size=10, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)
    
    return Model(X_in, X_out, name="SRCNN1895210")

def SRCNN27315():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=256, kernel_size=27, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3,  kernel_size=15, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)
    
    return Model(X_in, X_out, name="SRCNN27315")

def SRCNN27915():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=256, kernel_size=27, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=128, kernel_size=9, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3,  kernel_size=15, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)
    
    return Model(X_in, X_out, name="SRCNN27915")

def SRCNN271515():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=256, kernel_size=27, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=128, kernel_size=15, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3,  kernel_size=15, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)
    
    return Model(X_in, X_out, name="SRCNN271515")

def SRCNN97315():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=64, kernel_size=9, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=32, kernel_size=7, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=8, kernel_size=1, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3, kernel_size=5, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)
    
    return Model(X_in, X_out, name="SRCNN97315")

def SRCNN361220():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=256, kernel_size=36, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=128, kernel_size=12, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3,  kernel_size=20, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)
    
    return Model(X_in, X_out, name="SRCNN361220")

def SRCNN36420():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=256, kernel_size=36, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=128, kernel_size=4, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3,  kernel_size=20, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)
    
    return Model(X_in, X_out, name="SRCNN36420")

def SRCNN362020():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=256, kernel_size=36, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=128, kernel_size=20, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3,  kernel_size=20, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)
    
    return Model(X_in, X_out, name="SRCNN362020")

def SRCNN9315():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=64, kernel_size=9, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=8, kernel_size=1, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3, kernel_size=5, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)
    
    return Model(X_in, X_out, name="SRCNN9315")

def SRCNN973135():
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=64, kernel_size=9, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X_in)
    X = Conv2D(filters=32, kernel_size=7, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=8, kernel_size=1, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=3, kernel_size=3, padding='same', activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X = Conv2D(filters=1, kernel_size=5, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)
    
    return Model(X_in, X_out, name="SRCNN973135")