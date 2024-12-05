"""Utility functions"""

import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd
from umap import UMAP

def compute_distance_matrix(X, Y):  # Compute the euclidean distance matrix

    ''' Compute euclidean distance matrix
    Parameters:
    ----------
    X: np.array()
        shape of (M = num of data points, D = num of dimensions)
    Y: np.array()
        shape of (N = num of data points, D = num of dimensions)

    Returns:
    -------
    distance_matrix: np.array()
        shape of (M, N)
    '''

    x_2 = np.sum(X ** 2, axis=1, keepdims=True)  # shape = (M, 1)

    y_2 = np.sum(Y ** 2, axis=1)
    y_2 = y_2[np.newaxis, :]  # shape = (1, N)

    xy = X @ Y.T  # shape = (M, N)

    distance_matrix = x_2 - 2 * xy + y_2
    distance_matrix[distance_matrix < 0] = 0
    distance_matrix = np.sqrt(distance_matrix)

    return distance_matrix


def load_training(path):
    files = next(os.walk(path))[2]
    bf_channel = 'img'
    bf_bool = [bf_channel in ele for ele in files]
    bf_files = np.array(files)[np.array(bf_bool)]
    bf_files.sort()

    seg_channel = 'seg'
    seg_bool = [seg_channel in ele for ele in files]
    seg_files = np.array(files)[np.array(seg_bool)]
    seg_files.sort()

    zdisp_channel = 'zdisp'
    zdisp_bool = [zdisp_channel in ele for ele in files]
    zdisp_files = np.array(files)[np.array(zdisp_bool)]
    zdisp_files.sort()

    bf_crops = []
    seg_crops = []
    z_disps = []

    for bf_file, seg_file, zdisp_file in tqdm(zip(bf_files, seg_files, zdisp_files)):
        bf_crop = np.load(path + bf_file)
        seg_crop = np.load(path + seg_file)
        z_disp = np.load(path + zdisp_file)

        bf_crops.append(bf_crop)
        seg_crops.append(seg_crop)
        z_disps.append(z_disp)

    bf_crops = np.array(bf_crops, dtype=object)
    seg_crops = np.array(seg_crops, dtype=object)
    z_disps = np.array(z_disps, dtype=object)

    bf_crops = np.concatenate(bf_crops, axis=0)
    seg_crops = np.concatenate(seg_crops, axis=0)
    z_disps = np.concatenate(z_disps, axis=0)

    seg_crops[seg_crops != 0] = 1  # Change all nonzero elements to 1 (Semantic segmentation)

    bf_crops = bf_crops.astype(np.float32)
    seg_crops = seg_crops.astype(np.float32)

    # bf_crops_concat = np.concatenate(bf_crops, axis=0)  # (8123, 20, 32, 32) -> (8123*20, 32, 32)
    # seg_crops_rep = np.repeat(seg_crops, repeats=20, axis=0)  # Repeats Z-stack 20 times, (8123, 20, 32, 32) -> (8123*20, 20, 32, 32)

    bf_crops = np.expand_dims(bf_crops, axis=-1)
    # bf_crops_concat = np.expand_dims(bf_crops_concat, axis=-1)
    seg_crops = np.expand_dims(seg_crops, axis=-1)
    # seg_crops_rep = np.expand_dims(seg_crops_rep, axis=-1)

    print(bf_crops.shape, bf_crops.dtype)
    # print(bf_crops_concat.shape, bf_crops_concat.dtype)
    print(seg_crops.shape, seg_crops.dtype)
    # print(seg_crops_rep.shape, seg_crops_rep.dtype)

    return bf_crops, seg_crops

def to_trajectory_duration(df, duration=10, condition_name='Type', frame_name='Time', label_name='Label'):
    ''' Generate trajectories that are spliced by specific duration (applied to trajectories with varied duration)
    Parameters:
    ----------
    df: pd.DataFrame()
        raw dataframe
    duration: int
        number of time frames to generate cell trajectories with consistent frames
    condition_name: str
        name of the column to be grouped

    Returns:
    -------
    traj_data: pd.DataFrame()
        generate spliced trajectory dataframe with additional Time_span, pseudo_TrackID, pseudo_Time column
    '''

    label_data = df.groupby([condition_name, label_name]).apply(lambda x: x.name)  # contain (cell type, TrackID) tuple

    traj_data = pd.DataFrame()
    time_list = []

    for traj_idx in tqdm(range(0, label_data.shape[0])):  # For each cell trajectory(time 1~t)

        traj_data_temp = df.groupby([condition_name, label_name]).get_group(label_data.iloc[traj_idx]).copy()
        # traj_data_temp= PC1, PC2, feature1, feature 2, ... data for each cell trajectory(time 1~t)
        traj_data_temp.reset_index(inplace=True, drop=True)
        traj_data_temp['Time_span'] = traj_data_temp.shape[0]
        time_list.append(traj_data_temp.shape[0])

        if traj_data_temp.shape[0] < duration:  # discard cell that are not tracked long enough
            continue
        if traj_data_temp.shape[0] >= duration:
            for j in range(0, df[frame_name].max() // duration):  # 181 // 20 = 9 (quotient) -> j = 0~8,
                if traj_data_temp.shape[0] // duration > j:  # ex) time_span = 31, then 31//10 = 3, so loop only from j = 0~2   time_span = 181, 181//20 = 9, so j = 0~8
                    new_traj_data = traj_data_temp[:][duration * j:duration * (j+1)]  # df[:][0~10] = row 0~9,df[:][10~20] = row 10~19, ...
                    new_traj_data['pseudo_%s' % label_name] = np.array(pd.DataFrame(traj_data_temp[:][duration * j:duration * (j+1)][label_name].values, dtype='string') + '_%s' % j).flatten()
                    new_traj_data['pseudo_%s' % frame_name] = np.array(range(0, duration))
                    traj_data = pd.concat([traj_data, new_traj_data])

    traj_data.reset_index(inplace=True, drop=True)

    plt.figure()
    plt.hist(np.array(time_list), bins=80)
    plt.title("Time Histogram")
    plt.show()

    print('total number of cells trajectories: ', label_data.shape[0])
    print('number of cell trajectories more than %s frames: ' % duration, len(traj_data[traj_data['Time_span'] >= duration].groupby([condition_name, label_name]).apply(lambda x: x.name)))
    print('number of generated cell trajectories by %s frames duration: ' % duration, traj_data.groupby([condition_name, 'pseudo_%s' % label_name]).apply(lambda x: x.name).shape[0])

    return traj_data

def to_timeseries_fast(df, duration, feature_name=['PC1', 'PC2']):
    traj_list = []
    time_series_list = []
    time_series_dict = {}
    for traj_idx in range(int(df.shape[0]/duration)):  # For each cell trajectory
        traj_data_temp = df[duration*traj_idx:duration*(traj_idx+1)]
        traj_list.append(traj_data_temp)
        time_series_list.append(traj_data_temp[feature_name].values)  # np.array with [PC1, PC2] at t=0, [PC1, PC2] at t=1, ... [PC1, PC2] at t= T (shape = (time frames, 2) )
        time_series_dict[traj_idx] = traj_data_temp[feature_name].values
    time_series_array = np.array(time_series_list) # time_series = np.array with shape (number of traj,number of frames, dimension = 2 or 3)
    return traj_list, time_series_array, time_series_dict

def reduced_label_for_overlapped_volume(df, duration):
    """ Generate reduced features where each row is a trajectory for overlapped volume data
        Parameters:
        ----------
        df: pandas dataframe
            raw df where each row is one cell state at time t
        duration: int
            Number of time frames for each cell trajectory (all trajectories should have same duration)
        Returns:
        -------
        other_features_data: pandas dataframe
            dataframe with reduced label features
        """
    other_features_data = {}
    for feature_name in df.columns:
        aa = []
        for traj_idx in range(int(df.shape[0] / duration)):
            traj_data_temp = df[duration * traj_idx:duration * (traj_idx + 1)]
            row_values = pd.unique(traj_data_temp[feature_name])
            if any(txt in feature_name for txt in ('Overlapped', 'Shortest_Distance')):
                aa.append(traj_data_temp[feature_name].values)
            else:
                if row_values.shape[0] == 1:
                    aa.append(row_values[0])
                else:
                    aa.append(traj_data_temp[feature_name].values)
        other_features_data[feature_name] = aa
    return pd.DataFrame(other_features_data)

def dict_to_array(trajectories):
    a = []
    for traj_idx in trajectories:
        traj = trajectories[traj_idx]
        a.append(traj)
    trajectories_array = np.array(a)
    return trajectories_array

def array_to_dict(trajectories):
    trajectories_dict = {}
    for traj_idx in range(trajectories.shape[0]):
        traj = trajectories[traj_idx]
        trajectories_dict[traj_idx]=traj
    return trajectories_dict

def register_traj_disp(trajectories):
    def calc_max_distance(traj):
        all_distance_list = []
        for t in range(1, traj.shape[0]):
            distance = traj[t:] - traj[:-t]
            all_distance_list.append(max(abs(distance)))
        return max(all_distance_list)

    dim = trajectories[0].shape[1]
    registered_trajectories = {}
    for traj_idx in trajectories:
        traj = trajectories[traj_idx]

        max_dist, arg, tlag_max = -1, -1, -1
        for tlag in range(1, traj.shape[0]):
            dxyz = traj[tlag:] - traj[:-tlag]  # Displacement between two nearby points
            avg = np.ones((len(traj[:, 0]), 1)) * np.mean(traj, axis=0)
            xyr = traj - avg

            # determine the rotational matrix
            u, s, rotational_matrix = np.linalg.svd(dxyz)
            rotational_matrix = rotational_matrix.T

            # project major axis of trajectories onto rotational matrix
            xyr_r = xyr @ rotational_matrix
            if dim == 3:
                x = xyr_r[:, 0]
                y = xyr_r[:, 1]
                z = xyr_r[:, 2]
                list_dist = [calc_max_distance(x), calc_max_distance(y), calc_max_distance(z)]
                dist = max(list_dist)
                arg = np.argmax(list_dist)
                # print(tlag, calc_max_distance(x), calc_max_distance(y), calc_max_distance(z))
                if dist > max_dist:
                    max_dist = dist
                    max_arg = arg
                    tlag_max = tlag
                    list_dist[max_arg] = -1
                    max_arg2 = np.argmax(list_dist)
                    list_dist[max_arg2] = -1
                    max_arg3 = np.argmax(list_dist)
            elif dim == 2:
                x = xyr_r[:, 0]
                y = xyr_r[:, 1]
                list_dist = [calc_max_distance(x), calc_max_distance(y)]
                dist = max(list_dist)
                arg = np.argmax(list_dist)
                # print(tlag, calc_max_distance(x), calc_max_distance(y), calc_max_distance(z))
                if dist > max_dist:
                    max_dist = dist
                    max_arg = arg
                    tlag_max = tlag
                    list_dist[max_arg] = -1
                    max_arg2 = np.argmax(list_dist)
            # print(tlag, max_dist, max_arg, max_arg2, max_arg3)

        # print(tlag_max, max_dist, max_arg, max_arg2, max_arg3)

        dxyz = traj[tlag_max:] - traj[:-tlag_max]  # Displacement between two nearby points
        avg = np.ones((len(traj[:, 0]), 1)) * np.mean(traj, axis=0)
        xyr = traj - avg

        # determine the rotational matrix
        u, s, rotational_matrix = np.linalg.svd(dxyz)
        rotational_matrix = rotational_matrix.T

        # project major axis of trajectories onto rotational matrix
        rotated_traj = xyr @ rotational_matrix
        if dim == 3:
            pc1 = rotated_traj[:, max_arg]
            pc2 = rotated_traj[:, max_arg2]
            pc3 = rotated_traj[:, max_arg3]

            rotated_traj = np.vstack((pc1, pc2, pc3)).T

        elif dim == 2:
            pc1 = rotated_traj[:, max_arg]
            pc2 = rotated_traj[:, max_arg2]

            rotated_traj = np.vstack((pc1, pc2)).T

        #rotated_traj_origin = rotated_traj - np.tile(rotated_traj[0], (traj.shape[0], 1))
        registered_trajectories[traj_idx] = rotated_traj

    return registered_trajectories

def get_umap(df_input, n_neighbors, min_dist):

    __umap = UMAP(metric='euclidean', n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=0)
    pcs_array = __umap.fit_transform(df_input)
    df_pcs = pd.DataFrame(pcs_array, columns=['PC1', 'PC2'])

    return df_pcs

def change_dict_order(orig_dict, new_order):
    ''' Change the order of dictionary based on the specified keys
    Parameters:
    ----------
    orig_dict: dict
        original data in dictinary form
    new_order: list
        list of keys of desired order

    Returns:
    -------
    ordered_dict: dict
        new ordered data in dictionary form
    '''
    from collections import OrderedDict
    ordered_dict = OrderedDict(orig_dict)
    ordered_dict = OrderedDict((key, ordered_dict[key]) for key in new_order)

    return ordered_dict