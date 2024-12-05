"""Reshape the ground truth consistent with prediction"""

from utils.img_processing import *
from scipy.ndimage import zoom

save_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Behavior Reconstruction\prediction\pTh17\truth\\'
read_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Behavior Reconstruction\training\pTh17\segmented\\'

files = next(os.walk(read_path))[2]
seg_channel = 'segmented'
seg_bool = [seg_channel in ele for ele in files]
seg_files = np.array(files)[np.array(seg_bool)]
seg_files.sort()

crop_len = 5
for t, seg_file in tqdm(enumerate(seg_files)):
    seg = np.load(read_path + seg_file)  # (51, 1432, 1432)
    segs = []

    for i in range(crop_len):
        segs_row = []
        for j in range(crop_len):
            z, y, x = seg.shape
            crop = seg[:, i * (y // crop_len):(i + 1) * (y // crop_len),j * (x // crop_len):(j + 1) * (x // crop_len)]  # (51, 286, 286)
            #print(crop.shape)
            z, y, x = crop.shape
            crop = zoom(crop, (32 / z, ((y // 16) * 16) / y, ((x // 16) * 16) / x))  # (32, 272, 272)
            #crop = zoom(crop, (((z // 16) * 16) / z, ((y // 16) * 16) / y, ((x // 16) * 16) / x))
            z, y, x = crop.shape

            assert (z == 32) and (y % 16 == 0) and (x % 16 == 0), 'x and y k should be divisible by 16, and z size is 32'

            segs_row.append(crop)

        segs_row = np.array(segs_row)  # (5, 32, 272, 272)
        segs_row = np.concatenate(segs_row, axis=2)  # (32, 272, 5*272)
        segs.append(segs_row)

    segs = np.array(segs)  # (5, 32, 272, 1360)
    segs = np.concatenate(segs, axis=1)  # (32, 5*272, 1360)

    img_path = save_path + 'truth_t%03d.npy' % (t)
    np.save(img_path, segs)
