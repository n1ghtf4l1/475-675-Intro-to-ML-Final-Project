"""Training dataset preparation"""

from skimage import io
import napari
import pyclesperanto_prototype as cle
import cv2
from utils.img_processing import *

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Behavior Reconstruction\training\Th0\\'
bf_channel = 'ch01'
fluo_channel = 'ch00'

n_zstacks = 26
resize_factor = 1/2
crop_size = (32, 64, 64)  # (n_zstacks, y, x)

#################################################
files = next(os.walk(path))[2]

bf_bool = [bf_channel in ele for ele in files]
bf_files = np.array(files)[np.array(bf_bool)]
bf_files.sort()

fluo_bool = [fluo_channel in ele for ele in files]
fluo_files = np.array(files)[np.array(fluo_bool)]
fluo_files.sort()

cropped_imgs_allframes = []
cropped_segs_allframes = []
z_disps_allframes = []

if not os.path.isdir(path + 'cropped/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'cropped/')

if not os.path.isdir(path + 'segmented/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'segmented/')

for idx in tqdm(range(0, len(bf_files), n_zstacks)):
    t = idx // n_zstacks
    bf_Zstack = bf_files[idx:idx + n_zstacks]
    fluo_Zstack = fluo_files[idx:idx + n_zstacks]

    bf_imgs = []
    fluo_imgs = []
    for bf_eachZ, fluo_eachZ in zip(bf_Zstack, fluo_Zstack):
        # print(file)
        bf_img = io.imread(path + '\\' + bf_eachZ)
        row, col = bf_img.shape
        bf_img = cv2.resize(bf_img, (np.int32(row * resize_factor), np.int32(col * resize_factor)))
        bf_imgs.append(bf_img)

        fluo_img = io.imread(path + '\\' + fluo_eachZ)
        row, col = fluo_img.shape
        fluo_img = cv2.resize(fluo_img, (np.int32(row * resize_factor), np.int32(col * resize_factor)))
        fluo_imgs.append(fluo_img)

    bf_imgs = np.array(bf_imgs)
    fluo_imgs = np.array(fluo_imgs)

    bf_imgs = normalize_zstack(bf_imgs)
    fluo_imgs = normalize_zstack(fluo_imgs)

    interp_bf_imgs = interpolate_zstacks(bf_imgs, add=1, method='linear')
    interp_fluo_imgs = interpolate_zstacks(fluo_imgs, add=1, method='linear')

    cle.select_device('RTX')
    img_gpu = cle.push(interp_fluo_imgs)
    # backgrund_subtracted = cle.top_hat_box(img_gpu, radius_x=5, radius_y=5, radius_z=2)
    segmented_gpu = cle.voronoi_otsu_labeling(img_gpu, spot_sigma=2, outline_sigma=0.1)
    segmented = cle.pull(segmented_gpu)

    preprocess_seg = postprocess_segmented_zstack(segmented, volume_thresh=200)
    #nearby_removed_seg = remove_nearby_objects(preprocess_seg, thresh=30)

    cropped_imgs, cropped_segs, z_disps = croppedstacks_and_zdisps(interp_bf_imgs, preprocess_seg, size=crop_size)

    ########### Save cropped imgs and z displacements, and segmentation ###########
    img_path = path + 'cropped/img_t%03d.npy' % (t)
    seg_path = path + 'cropped/seg_t%03d.npy' % (t)
    zdisp_path = path + 'cropped/zdisp_t%03d.npy' % (t)

    segmented_path = path + 'segmented/segmented_t%03d.npy' % (t)

    np.save(img_path, cropped_imgs)
    np.save(seg_path, cropped_segs)
    np.save(zdisp_path, z_disps)
    np.save(segmented_path, preprocess_seg)

    ################################################################################
    #cropped_imgs_allframes.append(cropped_imgs)
    #cropped_segs_allframes.append(cropped_segs)
    #z_disps_allframes.append(z_disps)



viewer = napari.Viewer(ndisplay=3)
viewer.add_image(interp_bf_imgs, name='bf', colormap='gray', opacity=1)
viewer.add_image(interp_fluo_imgs, name='fluo', colormap='green', opacity=1)
viewer.add_labels(segmented, name='seg', opacity=1)

#cropped_imgs_allframes = np.array(cropped_imgs_allframes)
#cropped_segs_allframes = np.array(cropped_segs_allframes)
#z_disps_allframes = np.array(z_disps_allframes)

#print(cropped_imgs_allframes.shape)
#print(cropped_segs_allframes.shape)
#print(z_disps_allframes.shape)