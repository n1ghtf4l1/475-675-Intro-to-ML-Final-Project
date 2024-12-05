# Chanhong Min <cmin11@jhmi.edu>

# Copyright 2023 The Phillip tiME Lab at the Johns Hopkins University
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/Phillip-Lab-JHU/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Extract Features from trajectories"""

from features.Motility import BasicMotility
from features.Anisotropic_Motility import AnisotropicMotility
from utils.utils import *

duration = 20
df_duration = pd.read_csv('traj_duration_%s.csv' %duration)
traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration, duration=duration, feature_name=['x', 'y', 'z'])

###### Calculate basic motility features ######
feature_list = ['avg_speed', 'max_speed', 'min_speed', 'net_distance', 'progressivity', 'alphas',
                'avg_angle', 'max_angle', 'min_angle',
                'displ_variance', 'displ_cov','displ_skewness', 'displ_kurtosis', 'displ_ngaussalpha', 'displ_gini',
                'angle_variance', 'angle_cov', 'angle_skewness', 'angle_kurtosis', 'angle_ngaussalpha', 'angle_gini',
                'msds', 'displ_autocorr', 'displ_partial_autocorr','angle_autocorr', 'angle_partial_autocorr',
                'displ_hurst_RS', 'angle_hurst_RS',
                ]
basic_motil = BasicMotility(trajectories, time_unit=0.5, feature_list=feature_list)
df_basic = basic_motil.extract_features(tau_limit=3)

k = df_basic.isnull().any()
print(np.where(np.isnan(df_basic['alphas'])))

###### Calculate anisotropic motility features ######
ani_motil = AnisotropicMotility(trajectories, time_unit=0.5)
feature_list = ['avg_speed_x', 'max_speed_x', 'min_speed_x', 'net_distance_x', 'progressivity_x',
                'avg_speed_y', 'max_speed_y', 'min_speed_y', 'net_distance_y', 'progressivity_y',
                'avg_speed_z', 'max_speed_z', 'min_speed_z', 'net_distance_z', 'progressivity_z',
                'exy_max', 'eyz_max', 'exz_max', 'phi_max', 'exy_total', 'eyz_total', 'exz_total', 'phi_total',
                'msd_x', 'msd_y', 'msd_z',
                #'alpha_x', 'alpha_y', 'alpha_z',
                'displ_variance_x', 'displ_cov_x', 'displ_skewness_x', 'displ_kurtosis_x', 'displ_ngaussalpha_x',
                'displ_variance_y', 'displ_cov_y', 'displ_skewness_y', 'displ_kurtosis_y', 'displ_ngaussalpha_y',
                'displ_variance_z', 'displ_cov_z', 'displ_skewness_z', 'displ_kurtosis_z', 'displ_ngaussalpha_z',
                'displ_autocorr_x', 'displ_autocorr_y', 'displ_autocorr_z',
                #'displ_partial_autocorr_x', 'displ_partial_autocorr_y', 'displ_partial_autocorr_z',
                #'displ_hurst_RS_x', 'displ_hurst_RS_y', 'displ_hurst_RS_z'
                ]
#df_aniso = ani_motil(feature_list, tau_limit=3)
df_aniso = ani_motil.extract_features(feature_list=feature_list, tau_limit=3)
k = df_aniso.isnull().any()

df_motility = pd.concat([df_basic, df_aniso], axis=1)
df_motility.to_csv('motility_features_%s.csv'%duration, index=False)