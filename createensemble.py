from keras.models import load_model, Model
from keras.layers import Input, Average
from keras.utils.generic_utils import CustomObjectScope
from losses import grid_loss_with_hands

models = []
grids = []

input_tensor = Input(shape=(64,64,3))

for i in range(5):
    models.append(load_model('grid_model_bgr_' + str(i) + '.h5', custom_objects={'grid_loss_with_hands': grid_loss_with_hands}))
    grids.append(models[i](input_tensor))

output_tensor = Average()(grids)
ensemble_model = Model(inputs=input_tensor, outputs=output_tensor)
ensemble_model.save('grid_model_bgr_ensemble.h5')
