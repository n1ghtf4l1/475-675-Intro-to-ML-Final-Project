
"""Inference Step"""

from skimage import io
import cv2
import tensorflow as tf
from dnn.loss_func import jaccard_index, jaccard_loss, dice_coefficient
from utils.img_processing import *
from cellpose import models

save_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Behavior Reconstruction\prediction\Th0\\'
read_path= r"\\philliplab-server.wse.jhu.edu\data\Chanhong\Behavior Reconstruction\training\Th0\\"

files = next(os.walk(read_path))[2]

bf_channel = 'ch01'
bf_bool = [bf_channel in ele for ele in files]
bf_files = np.array(files)[np.array(bf_bool)]
bf_files.sort()

n_zstacks = 26
resize_factor = 1/2
crop_len=5

for idx in tqdm(range(0, len(bf_files), n_zstacks)):
    t = idx // n_zstacks
    bf_Zstack = bf_files[idx:idx + n_zstacks]

    bf_imgs = []
    for bf_eachZ in bf_Zstack:
        # print(file)
        bf_img = io.imread(read_path + '\\' + bf_eachZ)
        y, x = bf_img.shape
        bf_img = cv2.resize(bf_img, (np.int32(y * resize_factor), np.int32(x * resize_factor)))
        y, x = bf_img.shape
        #bf_img = cv2.resize(bf_img, ((y // 16) * 16, (x // 16) * 16))
        bf_imgs.append(bf_img)
    bf_imgs = np.array(bf_imgs)
    bf_imgs = normalize_zstack(bf_imgs)
    interp_bf_imgs = interpolate_zstacks(bf_imgs, add=1, method='linear')
    z, y, x = interp_bf_imgs.shape

    with tf.device('/cpu:0'):
        loaded_model = tf.keras.models.load_model('saved_model/Att_3DUnet')
        loaded_model.compile(loss=jaccard_loss, metrics=[jaccard_index, dice_coefficient],
                             optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1))

    preds = []
    crops = []
    for i in range(crop_len):
        preds_row=[]
        crops_row=[]
        for j in range(crop_len):
            z, y, x = interp_bf_imgs.shape
            crop = interp_bf_imgs[:, i*(y//crop_len):(i+1)*(y//crop_len), j*(x//crop_len):(j+1)*(x//crop_len)]
            #print(crop.shape)
            z, y, x = crop.shape
            from scipy.ndimage import zoom
            crop = zoom(crop, ( ( (z//16) * 16 )/z, ((y//16) * 16 )/y, ((x//16) * 16 )/x) )  # Resize img to be divisible by 16
            z,y,x = crop.shape

            assert (z%16==0) and (y%16==0) and (x%16==0), 'all shapes of z stack should be divisible by 16'

            crop = np.expand_dims(crop, axis=-1)
            crop = np.expand_dims(crop, axis=0)

            pred = loaded_model.predict(crop, verbose=1, batch_size=1)
            pred[pred < 0.6] = 0
            pred[pred >= 0.6] = 1
            pred = np.squeeze(pred, axis=0)
            pred = np.squeeze(pred, axis=-1)
            preds_row.append(pred)

            crop = np.squeeze(crop, axis=0)
            crop = np.squeeze(crop, axis=-1)
            crops_row.append(crop)

        preds_row = np.array(preds_row)  # (5, 48, 272, 272)
        preds_row = np.concatenate(preds_row, axis=2)  # (48, 272, 5*272)
        preds.append(preds_row)

        crops_row = np.array(crops_row)  # (5, 48, 272, 272)
        crops_row = np.concatenate(crops_row, axis=2)  # (48, 272, 5*272)
        crops.append(crops_row)

    preds = np.array(preds)  # (5, 48, 272, 1360)
    preds = np.concatenate(preds, axis=1)  # (48, 5*272, 1360)

    crops = np.array(crops)  # (5, 48, 272, 1360)
    crops = np.concatenate(crops, axis=1)  # (48, 5*272, 1360)

    # viewer = napari.view_image(crops, blending='additive', name='imgs', scale = np.array([1.5, 0.568, 0.568]))
    # viewer.add_image(preds, scale = np.array([1.5, 0.568, 0.568]), name='prediction')

    np.save(save_path + 'pred/pred_t%03d.npy' % (t), preds)
    np.save(save_path + 'brightfield/bf_t%03d.npy' % (t), crops)