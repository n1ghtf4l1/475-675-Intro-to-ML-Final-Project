"""Inference Step"""

from skimage import io
import cv2
import tensorflow as tf
from dnn.loss_func import jaccard_index, jaccard_loss, dice_coefficient
from utils.img_processing import *
from cellpose import models

save_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Behavior Reconstruction\prediction\pTh17\\'
read_path= r"\\philliplab-server.wse.jhu.edu\data\Chanhong\Behavior Reconstruction\training\pTh17\\"
#read_path2 = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Behavior Reconstruction\prediction\Th0\truth\\'

cyto3 = models.Cellpose(gpu=True,model_type='cyto3')
#cyto3_denoise = denoise.DenoiseModel(model_type="denoise_cyto3", gpu=True)


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
        bf_img = io.imread(read_path + '\\' + bf_eachZ)  # (2865, 2865)
        y, x = bf_img.shape
        bf_img = cv2.resize(bf_img, (np.int32(y * resize_factor), np.int32(x * resize_factor)))  # (1432, 1432)
        #bf_img = cv2.resize(bf_img, ((y // 16) * 16, (x // 16) * 16))
        bf_imgs.append(bf_img)
    bf_imgs = np.array(bf_imgs)  # (26, 1432, 1432)
    bf_imgs = normalize_zstack(bf_imgs)  # (26, 1432, 1432)
    interp_bf_imgs = interpolate_zstacks(bf_imgs, add=1, method='linear')  # (51, 1432, 1432)

    interp_bf_img_slice = interp_bf_imgs[26:27, :, :]  # Pick center slice (1, 1432, 1432)

    #interp_bf_img_slice_denoised = cyto3_denoise.eval(interp_bf_img_slice, channels=[0, 0])
    # masks, flows, styles, diams = cyto3.eval(interp_bf_img_slice, channels=[0,0],
    #                                               diameter=35,
    #                                               cellprob_threshold=-0.7,
    #                                               flow_threshold=2.25,
    #                                               do_3D=False, min_size=-1)

    masks_slice, flows, styles, diams = cyto3.eval(interp_bf_img_slice[0], channels=[0,0],
                                                  diameter=35, cellprob_threshold=-2,
                                                  flow_threshold=3, do_3D=False, min_size=-1)
    masks_binary_slice = (masks_slice>=1).astype(np.uint8)  # (1432, 1432)
    masks_binary_slice = np.expand_dims(masks_binary_slice, axis=0)  #(1, 1432, 1432)

    z, y, x = interp_bf_img_slice.shape

    with tf.device('/cpu:0'):
        loaded_model = tf.keras.models.load_model('saved_model/recons_2d_3d_v2', compile=False)
        loaded_model.compile(loss=jaccard_loss, metrics=[jaccard_index, dice_coefficient],
                             optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1))


    # Divide one image into 25 tiles (original image shape is too big)
    preds = []
    crops = []
    for i in range(crop_len):
        preds_row=[]
        crops_row=[]
        for j in range(crop_len):
            z, y, x = interp_bf_img_slice.shape  # (51, 1432, 1432)
            crop = interp_bf_img_slice[:, i*(y//crop_len):(i+1)*(y//crop_len), j*(x//crop_len):(j+1)*(x//crop_len)]  # (1, 286, 286)
            mask_crop = masks_binary_slice[:, i*(y//crop_len):(i+1)*(y//crop_len), j*(x//crop_len):(j+1)*(x//crop_len)]  # (1, 286, 286)
            #print(crop.shape)
            z, y, x = crop.shape  # (1, 286, 286)
            from scipy.ndimage import zoom
            #crop = zoom(crop, ( ( (z//16) * 16 )/z, ((y//16) * 16 )/y, ((x//16) * 16 )/x) )  # Resize img to be divisible by 16  (48, 272, 272)
            crop = zoom(crop, (1, ((y // 16) * 16) / y, ((x // 16) * 16) / x))  # (1, 272, 272)
            mask_crop = zoom(mask_crop, (1, ((y // 16) * 16) / y, ((x // 16) * 16) / x))  # (1, 272, 272)

            z,y,x = crop.shape  # (1, 272, 272)

            #assert (z%16==0) and (y%16==0) and (x%16==0), 'all shapes of z stack should be divisible by 16'
            assert (y % 16 == 0) and (x % 16 == 0), 'all shapes of z stack should be divisible by 16'

            crop = np.expand_dims(crop, axis=-1)  #(1, 272, 272, 1)
            crop = np.expand_dims(crop, axis=0)  #(1, 1, 272, 272, 1)

            mask_crop = np.expand_dims(mask_crop, axis=-1)  # (1, 272, 272, 1)
            mask_crop = np.expand_dims(mask_crop, axis=0)  # (1, 1, 272, 272, 1)

            crop_merge = np.concatenate((crop, mask_crop), axis=-1)
            pred = loaded_model.predict(crop_merge, verbose=1, batch_size=1)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
            pred = np.squeeze(pred, axis=0)
            pred = np.squeeze(pred, axis=-1)
            preds_row.append(pred)

            crop = np.squeeze(crop, axis=0)
            crop = np.squeeze(crop, axis=-1)
            crops_row.append(crop)

        preds_row = np.array(preds_row)  # (5, 1, 272, 272)
        preds_row = np.concatenate(preds_row, axis=2)  # (1, 272, 5*272)
        preds.append(preds_row)

        crops_row = np.array(crops_row)  # (5, 1, 272, 272)
        crops_row = np.concatenate(crops_row, axis=2)  # (1, 272, 5*272)
        crops.append(crops_row)

    preds = np.array(preds)  # (5, 1, 272, 1360)
    preds = np.concatenate(preds, axis=1)  # (1, 5*272, 1360)

    crops = np.array(crops)  # (5, 1, 272, 1360)
    crops = np.concatenate(crops, axis=1)  # (1, 5*272, 1360)

    # viewer = napari.view_image(crops, blending='additive', name='imgs', scale = np.array([1.5, 0.568, 0.568]))
    # viewer.add_image(preds, scale = np.array([1.5, 0.568, 0.568]), name='prediction')

    np.save(save_path + 'pred/pred_t%03d.npy' % (t), preds)
    np.save(save_path + 'brightfield/bf_t%03d.npy' % (t), crops)


# viewer = napari.view_image(pred, blending='additive', name='imgs', scale = np.array([1.5, 0.568, 0.568]))
# viewer.add_image(crop[0, 0, :, :, 0], scale = np.array([0.568, 0.568]), name='denoised')
#
#
# viewer = napari.view_image(interp_bf_img_slice[0], blending='additive', name='imgs', scale = np.array([0.568, 0.568]))
# viewer.add_image(pred, scale = np.array([0.568, 0.568]), name='denoised')
# viewer.add_image(masks, scale = np.array([0.568, 0.568]), name='mask')
# viewer.add_image(masks_binary, scale = np.array([0.568, 0.568]), name='mask1')