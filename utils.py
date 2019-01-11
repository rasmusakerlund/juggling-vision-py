import numpy as np
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

def getDistance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

# Has to be called early otherwise uninitialized errors might occur.
def handleTensorflowSession(memoryLimit):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = memoryLimit
    config.gpu_options.visible_device_list = "0"
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 1
    set_session(tf.Session(config=config))
