import tensorflowjs as tfjs
from keras.models import load_model
from losses import grid_loss_with_hands

model = load_model('../grid_models/grid_model_submovavg_64x64_light.h5', custom_objects={'grid_loss_with_hands': grid_loss_with_hands})
#model = load_model('3b30_demo_pattern_model.h5')
tfjs.converters.save_keras_model(model, "../submovavglight_js")
