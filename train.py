"""Training the Model"""

import numpy as np
import matplotlib.pyplot as plt
import napari
import pandas as pd
from tqdm import tqdm
from utils.utils import load_training
from dnn.models import reconstruct_2d_3d
import tensorflow as tf
from dnn.loss_func import jaccard_index, jaccard_loss, dice_coefficient
from utils.img_processing import bbox_3d

################################# Load training data #################################
file_list=['Th0', 'Th1', 'Th2', 'Treg', 'nTh17', 'pTh17']
bf_cropss = []
seg_cropss = []

for file in tqdm(file_list):
    path = r"\\philliplab-server.wse.jhu.edu\data\Chanhong\Behavior Reconstruction\training\%s\cropped\\"%file
    bf_crops, seg_crops = load_training(path)

    bf_cropss.append(bf_crops)
    seg_cropss.append(seg_crops)
    print( '%s: %s number of data'%(file, bf_crops.shape[0]) )

bf_cropss = np.array(bf_cropss, dtype=object)
seg_cropss = np.array(seg_cropss, dtype=object)

bf_cropss = np.concatenate(bf_cropss, axis=0)
seg_cropss = np.concatenate(seg_cropss, axis=0)

# import random
# i = random.randint(0, len(bf_cropss)-1)
# viewer = napari.view_image(np.squeeze(bf_cropss[i], axis=-1), blending='additive', name='imgs', scale = np.array([1.5, 0.568, 0.568]))
# viewer.add_image(np.squeeze(seg_cropss[i], axis=-1), scale = np.array([1.5, 0.568, 0.568]), name='truth')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(bf_cropss, seg_cropss, test_size=0.1, random_state=0)

print('x_train: ',x_train.dtype, x_train.shape)
print('x_test: ',x_test.dtype, x_test.shape)
print('y_train: ',y_train.dtype, y_train.shape)
print('y_test: ',y_test.dtype, y_test.shape)

################################# Train model #################################
#input_shape=(32,32,32,1)
input_shape=(None,None,None,1)
model = Attention_3DUnet(input_shape)
model_name='Att_3DUnet'

from tensorflow.keras import callbacks
checkpointer = callbacks.ModelCheckpoint('saved_model/%s'%model_name, monitor='val_jaccard_index', verbose=1,
                save_weights_only=False,save_freq='epoch', mode='auto', save_best_only=True)  # Save model every epoch
stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5)

result = model.fit(x_train, y_train, batch_size=3, epochs=10, verbose=1, validation_data = (x_test, y_test),
                   shuffle=True, callbacks=[stop_early, checkpointer])

################################# Save model #################################
#!mkdir -p saved_model -> saved_model이라는 폴더 생성
model.optimizer = None
model.compiled_loss = None
model.compiled_metrics = None
# model의 compile된 loss 와 metric을 모두 초기화 시킨 후 저장해야 에러가 발생안함
# -> 왜냐하면 sm.metrics.iou_score 라는 사용자 정의 metric을 읽지 못하는 에러가 발생하기 때문

model.save('saved_model/%s'%model_name, save_format='tf')
model.save_weights('saved_model/%s_weights'%model_name, save_format='h5')
# save_format은 'tf' = tensorflow saved model 형식, 'h5' = HDF5 형식 2개임
# C:\Users\chanhong\python programming\Deep Learning\saved_model에 생성됨

#ls saved_model

################################# Loss for each epoch #################################
df_history = pd.DataFrame(result.history) # result.history 는 dict 형태

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4))
t = fig.suptitle('Model Performance', fontsize=12)
fig.subplots_adjust(top=0.85,wspace=0.3)

max_epoch = len(result.history['loss']) # 25(epoch 수)
epoch_list = list(range(1,max_epoch+1)) # range(1,26) = 1~25

ax1.plot(epoch_list, result.history['jaccard_index'], label = 'training iou score')
ax1.plot(epoch_list, result.history['val_jaccard_index'], label = 'validation iou score')
ax1.set_xticks(np.arange(1, max_epoch, 100))
ax1.set_xlabel('epoch')
ax1.set_ylabel('iou socre')
ax1.set_title('iou score test')
ax1.legend(loc='best')

ax2.plot(epoch_list, result.history['loss'], label = 'training loss')
ax2.plot(epoch_list, result.history['val_loss'], label = 'validation loss')
ax2.set_xticks(np.arange(1, max_epoch, 100))
ax2.set_xlabel('epoch')
ax2.set_ylabel('loss')
ax2.set_title('loss test')
ax2.legend(loc='best')

plt.savefig('loss_40.png')

################################# Load model and retrain #################################
model_name='Att_3DUnet'

from tensorflow.keras import callbacks
checkpointer = callbacks.ModelCheckpoint('saved_model/%s'%model_name, monitor='val_jaccard_index', verbose=1,
                save_weights_only=False,save_freq='epoch', mode='auto', save_best_only=True)  # Save model every epoch
stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5)

loaded_model = tf.keras.models.load_model('saved_model/%s'%model_name, compile = False)
loaded_model.compile(loss = jaccard_loss, metrics = [jaccard_index, dice_coefficient], optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1))

result = loaded_model.fit(x_train, y_train, batch_size=1, epochs=5, verbose=1, validation_data = (x_test, y_test),
                   shuffle=True, callbacks=[stop_early, checkpointer])

#!mkdir -p saved_model -> saved_model이라는 폴더 생성
loaded_model.optimizer = None
loaded_model.compiled_loss = None
loaded_model.compiled_metrics = None
# model의 compile된 loss 와 metric을 모두 초기화 시킨 후 저장해야 에러가 발생안함
# -> 왜냐하면 sm.metrics.iou_score 라는 사용자 정의 metric을 읽지 못하는 에러가 발생하기 때문

loaded_model.save('saved_model/%s'%model_name, save_format='tf')
loaded_model.save_weights('saved_model/%s_weights'%model_name, save_format='h5')
# save_format은 'tf' = tensorflow saved model 형식, 'h5' = HDF5 형식 2개임
# C:\Users\chanhong\python programming\Deep Learning\saved_model에 생성됨

y_pred = loaded_model.predict(bf_cropss[0:100])
idx=3
viewer = napari.view_image(np.squeeze(bf_cropss[idx], axis=-1), blending='additive', name='imgs', scale = np.array([1.5, 0.568, 0.568]))
viewer.add_image(np.squeeze(y_pred[idx], axis=-1), scale = np.array([1.5, 0.568, 0.568]), name='prediction')
viewer.add_image(np.squeeze(seg_cropss[idx], axis=-1), scale = np.array([1.5, 0.568, 0.568]), name='truth')