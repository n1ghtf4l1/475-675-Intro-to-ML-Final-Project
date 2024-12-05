"""3D Tracking"""

from utils.img_processing import *
import napari
import pyclesperanto_prototype as cle
import trackpy as tp

read_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Behavior Reconstruction\3d_to_3d\prediction\Th0\\'
#read_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Behavior Reconstruction\prediction\Th0\\'
bf_files = next(os.walk(read_path+'/brightfield'))[2]
bf_files.sort()

pred_files = next(os.walk(read_path+'/pred'))[2]
pred_files.sort()

truth_files = next(os.walk(read_path+'/truth'))[2]
truth_files.sort()

df_pred_all = pd.DataFrame()
df_truth_all = pd.DataFrame()
for t, (bf_file, pred_file, truth_file) in tqdm( enumerate( zip(bf_files[0:1], pred_files[0:1], truth_files[0:1]) ) ):

    bf_file = bf_files[t]
    pred_file = pred_files[t]
    truth_file = truth_files[t]

    bf = np.load(read_path +'/brightfield/' + bf_file)
    pred = np.load(read_path + '/pred/' +pred_file)
    truth = np.load(read_path + '/truth/' +truth_file)

    # labels = skimage.measure.label(pred)
    # labels = postprocess_segmented_zstack(labels, volume_thresh=200)
    # labels = remove_nearby_objects(labels, thresh=30)

    cle.select_device('RTX')
    img_gpu = cle.push(pred)
    # backgrund_subtracted = cle.top_hat_box(img_gpu, radius_x=5, radius_y=5, radius_z=2)
    pred_gpu = cle.voronoi_otsu_labeling(img_gpu, spot_sigma=1, outline_sigma=1)
    pred = cle.pull(pred_gpu)

    pred = postprocess_segmented_zstack(pred, volume_thresh=200)
    #pred = remove_nearby_objects(pred, thresh=30)

    df_pred = pd.DataFrame(skimage.measure.regionprops_table(pred, properties=['label', 'centroid']))
    df_pred.rename(columns={'centroid-0': 'z', 'centroid-1': 'y', 'centroid-2': 'x' }, inplace=True)
    df_pred['frame'] = t
    df_pred_all = pd.concat([df_pred_all, df_pred], axis=0)

    df_truth = pd.DataFrame(skimage.measure.regionprops_table(truth, properties=['label', 'centroid']))
    df_truth.rename(columns={'centroid-0': 'z', 'centroid-1': 'y', 'centroid-2': 'x'}, inplace=True)
    df_truth['frame'] = t
    df_truth_all = pd.concat([df_truth_all, df_truth], axis=0)

df_pred_all.reset_index(drop=True, inplace=True)
df_truth_all.reset_index(drop=True, inplace=True)

nvp = tp.predict.NearestVelocityPredict()
df_pred_linked = nvp.link_df(df_pred_all, search_range=50, memory=0, adaptive_stop=8, adaptive_step=0.9, pos_columns=['x', 'y', 'z'])
df_pred_linked['type'] = 'pred'
df_truth_linked = nvp.link_df(df_truth_all, search_range=50, memory=0, adaptive_stop=8, adaptive_step=0.9, pos_columns=['x', 'y', 'z'])
df_truth_linked['type'] = 'truth'

# SubnetOversizeException: search range too large, so trackpy will become overwhelmed with candidate features, and give error
# Solution: reduce adaptive_stop
# tp.linking.Linker.MAX_SUB_NET_SIZE = 30, tp.linking.Linker.MAX_SUB_NET_SIZE_ADAPTIVE = 15 by default

df_linked = pd.concat([df_pred_linked, df_truth_linked], axis=0).reset_index(drop=True)
df_linked.to_csv('Th0_pos.csv', index=False)


#viewer.add_tracks(data_true, tail_width=5, name='true tracking')


um_per_zsice = 1.5
um_per_pixel = 0.568

viewer = napari.Viewer(ndisplay=3)
viewer.axes.visible = True
viewer.scale_bar.visible=True
viewer.scale_bar.unit = 'μm'
z, y, x = pred.shape
lines = bbox_3d(z*um_per_zsice, y*um_per_pixel, x*um_per_pixel)
viewer.add_shapes(
        lines,
        shape_type="line",
        edge_width=0.1,
        edge_color="white",
    )
#viewer.add_image(imgs,  scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), colormap='gray', opacity=1)
viewer.add_image(bf, blending='additive', name='imgs', scale = np.array([um_per_zsice, um_per_pixel, um_per_pixel]))
#viewer.add_image(np.squeeze(y_pred[idx], axis=-1), scale = np.array([um_per_zsice, um_per_pixel, um_per_pixel]), name='prediction')
viewer.add_image(pred, scale = np.array([um_per_zsice, um_per_pixel, um_per_pixel]), name='prediction')
viewer.add_image(truth, scale = np.array([um_per_zsice, um_per_pixel, um_per_pixel]), name='truth')


viewer.camera.angles = (-1, 25, 85)
viewer.dims.axis_labels = ['t', 'z', 'y', 'x']