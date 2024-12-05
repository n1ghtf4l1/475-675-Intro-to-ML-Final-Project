"""Compare Trajectories of Prediction with Ground Truth"""

from utils.DrawingGraph import *
from utils.utils import *
from dnn.models import Temporal_Conv1D_2D
from tensorflow.keras import models

dataset = {'2D_to_3D_bf': [0.3454], '3D_to_3D': [0.8332], '2D_to_3D_bf_and_seg':[0.7825]}

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Behavior Reconstruction\results\\'
draw_custom_bar_plot(dataset, directory=path, file_name='accuracy comparison', colors=('#888888', '#888888', '#CC6677'),
                         strip_plot=False, test='mann-whitney', pvalue=False, figsize=(1,2))

########################### Get traj_duration ###########################
excel = pd.read_csv('Th0_pos.csv')

duration = 20
df = to_trajectory_duration(excel, duration=duration, condition_name='type', frame_name='frame', label_name='particle')
traj_list, trajectories_array, trajectories = to_timeseries_fast(df, duration=duration, feature_name=['x', 'y', 'z'])
rotated_trajectories = register_traj_disp(trajectories)
rotated_trajectories = dict_to_array(rotated_trajectories)
df_traj = rotated_trajectories.reshape(rotated_trajectories.shape[0] * rotated_trajectories.shape[1], 3)
df_traj = pd.DataFrame(df_traj, columns=['Rotated_x', 'Rotated_y', 'Rotated_z'])
df = pd.concat([df_traj, df], axis=1)
df.to_csv('traj_duration_%s.csv' % duration, index=False)

########################### Get traj_duration ###########################
duration = 20
df_duration = pd.read_csv('traj_duration_%s.csv' %duration)
traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration, duration=duration, feature_name=['x', 'y', 'z'])
model = Temporal_Conv1D_2D(duration=duration, coor_dim=3, dimension=128)
rotated_trajectories = register_traj_disp(trajectories)
rotated_trajectories = dict_to_array(rotated_trajectories)
X_train = rotated_trajectories

result = model.fit(X_train, X_train, batch_size=256, epochs=10000, verbose=1, validation_split=0.1, shuffle=True)

model_name='Temporal_Conv1D_2D'
model.save('saved_model/%s'%model_name, save_format='tf')
model.save_weights('saved_model/%s_weights'%model_name, save_format='h5')

########################### Plot errors of model  ################################
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4))
t = fig.suptitle('Performance', fontsize=12)
fig.subplots_adjust(top=0.85,wspace=0.3)

max_epoch = len(result.history['accuracy']) # 25(epoch 수)
epoch_list = list(range(1,max_epoch+1)) # range(1,26) = 1~25

ax1.plot(epoch_list, result.history['accuracy'], label = 'training accuracy')
ax1.plot(epoch_list, result.history['val_accuracy'], label = 'validation accuracy')
ax1.set_xticks(np.arange(1, max_epoch, 1000))
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.set_title('accuracy test')
ax1.legend(loc='best')

ax2.plot(epoch_list, result.history['loss'], label = 'training loss')
ax2.plot(epoch_list, result.history['val_loss'], label = 'validation loss')
ax2.set_xticks(np.arange(1, max_epoch, 1000))
ax2.set_xlabel('epoch')
ax2.set_ylabel('loss')
ax2.set_title('loss test')
ax2.legend(loc='best')
# training accuracy(training data) = 매우 높아짐, but validation accuracy(test data) = 높지않음  ------> Overfitting
# 이 때는 epoch number 증가, training images 증가 필요
plt.savefig('traj_recons_loss_10000.png')

########################### Extract latent vectors ################################
bottleneck = models.Model(inputs=model.inputs, outputs=model.layers[64].output)
lvs = bottleneck.predict(X_train)
df_lv = pd.DataFrame(lvs, columns=[ 'lv_%s' % str(i) for i in range(0,lvs.shape[1]) ])
df_labels = reduced_label_for_overlapped_volume(df_duration, duration=duration)

df_motility = pd.concat([df_lv, df_labels], axis=1)
df_motility.to_csv('latent_vector_%s.csv'%duration, index=False)

########################### Extract latent vectors  ################################
from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
df_features = pd.read_csv('latent_vector_20.csv')
df_motility = pd.read_csv('motility_features_20.csv')
motility_data = df_features.iloc[:,:128]

scaler = StandardScaler()  # if not normalize, UMAP space is completely different
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)

umap = get_umap(motility_data_scaled, 20, 0.5)
df = pd.concat([df_features, df_motility, umap], axis=1)

directory = r'C:\Users\ChanhongMin\PycharmProjects\Behavior_Reconstruction\results\\'

df_ = df.replace({'type': {'pred': 'Predicted behavior', 'truth': 'True behavior'}})
draw_umap_space(df_, directory, file_name='space_Type', condition_name='type', label_name='pseudo_particle', colors=('#888888', '#CC6677'), x_name='PC1', y_name='PC2', dot_size=0.07)
draw_contour(df_, directory, file_name='space_contour', condition_name='type', colors=('#888888', '#CC6677'), x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)

df.columns.get_loc('avg_speed')
df.columns.get_loc('displ_autocorr_z_3')

condition_name='type'
replace_keys = {'truth':'truth', 'pred':'prediction'}
feature_list = df.columns[141:].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
if not os.path.isdir(directory + 'feature_violin_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(directory + 'feature_violin_plot_type/')


for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df[condition_name]):
        data = df[df[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    new_order = ['pred', 'truth']
    ordered_dataset = change_dict_order(dataset, new_order)
    dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

    draw_custom_violin_plot(dict_datasets, directory+'feature_violin_plot_type/', file_name=feature_name, colors=('#888888', '#CC6677'),
                            test='mann-whitney', pvalue=True, figsize=(1,2))
