from patterndataloader import PatternDataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
import cv2

pdl = PatternDataLoader("3balls_demo", length=60)
saveModelFilename="3b30_demo_pattern_model.h5"



from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, GaussianNoise, LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import regularizers
from keras import metrics

model = Sequential()
model.add(Flatten(input_shape=pdl.trainx.shape[1:3]))
# model.add(GaussianNoise(0.1))
# model.add(Dropout(0.10))
model.add(Dense(units=60, kernel_regularizer=regularizers.l2(0.0001)))
model.add(LeakyReLU())
model.add(Dense(units=60, kernel_regularizer=regularizers.l2(0.0001)))
model.add(LeakyReLU())
model.add(Dense(units=60, kernel_regularizer=regularizers.l2(0.0001)))
model.add(LeakyReLU())


model.add(Dense(units=13, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=[metrics.categorical_accuracy])
checkpoint = ModelCheckpoint(saveModelFilename, verbose=1, save_best_only=True, period=1)
model.fit(x=pdl.trainx,y=pdl.trainy, batch_size=32, validation_data=(pdl.valx, pdl.valy), epochs=50, callbacks=[checkpoint])
# batch_size = 32, epochs = 50


model = load_model(saveModelFilename)

metrics = model.evaluate(pdl.testx, pdl.testy)
print("Testset Accuracy: %.1f" % (metrics[1]*100))
pred = model.predict(pdl.testx)
pred = np.argmax(pred, axis=1)
testy  = np.argmax(pdl.testy, axis=1)

names = pdl.getNames()
testy = [names[i] for i in testy]
pred = [names[i] for i in pred]
skplt.metrics.plot_confusion_matrix(testy, pred, x_tick_rotation=45, title=" ", text_fontsize="large")
plt.show()








# sample = np.random.randint(0,pdl.valx.shape[0])
#
# recording = pdl.valx[sample]
#
# recording = recording - np.min(recording)
# recording = recording * 256 / np.max(recording)
# recording = recording.astype(np.uint8)
#
# for i in range(0,recording.shape[0]):
#     canvas = np.zeros((256,256,3), dtype=np.uint8)
#     cv2.line(canvas, (recording[i,0]-10, recording[i,1]), (recording[i,0]+10, recording[i,1]), (0,255,0), 2)
#     cv2.line(canvas, (recording[i,2]-10, recording[i,3]), (recording[i,2]+10, recording[i,3]), (0,0,255), 2)
#     for j in range(4, recording.shape[1], 2):
#         colorshift = j*50 % 255
#         cv2.circle(canvas, (recording[i,j], recording[i,j+1]), 10, (colorshift,255-colorshift,colorshift), 2)
#     cv2.imshow('PlayPattern', canvas)
#     cv2.waitKey(60)
#
#
#
#
# cv2.waitKey()
# print(pdl.valy[sample])
