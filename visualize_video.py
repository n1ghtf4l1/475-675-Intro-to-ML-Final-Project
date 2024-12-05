"""Visualize Time Series Z-stacks"""

from utils.img_processing import *
import napari
import pyclesperanto_prototype as cle

read_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Behavior Reconstruction\prediction\Th0\\'
bf_files = next(os.walk(read_path+'/brightfield'))[2]
bf_files.sort()

pred_files = next(os.walk(read_path+'/pred'))[2]
pred_files.sort()

truth_files = next(os.walk(read_path+'/truth'))[2]
truth_files.sort()

crop_len=5
i,j = (2,2)


bf_crops = []
pred_crops = []
truth_crops = []

for t, (bf_file, pred_file, truth_file) in tqdm( enumerate( zip(bf_files[:10], pred_files[:10], truth_files[0:10]) ) ):

    bf_file = bf_files[t]
    pred_file = pred_files[t]
    truth_file = truth_files[t]

    bf = np.load(read_path + '/brightfield/' + bf_file)
    pred = np.load(read_path + '/pred/' + pred_file)
    truth = np.load(read_path + '/truth/' + truth_file)

    cle.select_device('RTX')
    img_gpu = cle.push(pred)
    # backgrund_subtracted = cle.top_hat_box(img_gpu, radius_x=5, radius_y=5, radius_z=2)
    pred_gpu = cle.voronoi_otsu_labeling(img_gpu, spot_sigma=1, outline_sigma=1)
    pred = cle.pull(pred_gpu)

    pred = postprocess_segmented_zstack(pred, volume_thresh=200)

    z, y, x = bf.shape  # (48, 1360, 1360)
    bf_crop = bf[:, i * (y // crop_len):(i + 1) * (y // crop_len), j * (x // crop_len):(j + 1) * (x // crop_len)] # (48, 272, 272)
    pred_crop = pred[:, i * (y // crop_len):(i + 1) * (y // crop_len), j * (x // crop_len):(j + 1) * (x // crop_len)] # (48, 272, 272)
    truth_crop = truth[:, i * (y // crop_len):(i + 1) * (y // crop_len), j * (x // crop_len):(j + 1) * (x // crop_len)] # (48, 272, 272)

    bf_crops.append(bf_crop)
    pred_crops.append(pred_crop)
    truth_crops.append(truth_crop)

bf_crops = np.array(bf_crops) # (n_frames, 48, 272, 272)
pred_crops = np.array(pred_crops) # (n_frames, 48, 272, 272)
truth_crops = np.array(truth_crops) # (n_frames, 48, 272, 272)

# viewer = napari.view_image(bf_crops, blending='additive', name='imgs', scale = np.array([1.5, 0.568, 0.568]))
# viewer.add_labels(pred_crops, scale = np.array([1.5, 0.568, 0.568]), name='prediction')
# viewer.add_labels(truth_crops, scale = np.array([1.5, 0.568, 0.568]), name='truth')


um_per_zsice = 1.5
um_per_pixel = 0.568

viewer = napari.Viewer(ndisplay=3)
viewer.axes.visible = True
viewer.scale_bar.visible=True
viewer.scale_bar.unit = 'μm'
t, z, y, x = pred_crops.shape
lines = bbox_3d(z*um_per_zsice, y*um_per_pixel, x*um_per_pixel)
viewer.add_shapes(
        lines,
        shape_type="line",
        edge_width=0.1,
        edge_color="white",
    )
#viewer.add_image(imgs,  scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), colormap='gray', opacity=1)
viewer.add_image(bf_crops, blending='additive', name='imgs', scale = np.array([um_per_zsice, um_per_pixel, um_per_pixel]))
#viewer.add_image(np.squeeze(y_pred[idx], axis=-1), scale = np.array([um_per_zsice, um_per_pixel, um_per_pixel]), name='prediction')
viewer.add_image(pred_crops, scale = np.array([um_per_zsice, um_per_pixel, um_per_pixel]), name='prediction')
viewer.add_image(truth_crops, scale = np.array([um_per_zsice, um_per_pixel, um_per_pixel]), name='truth')

viewer.camera.angles = (-1, 25, 85)
viewer.dims.axis_labels = ['t', 'z', 'y', 'x']