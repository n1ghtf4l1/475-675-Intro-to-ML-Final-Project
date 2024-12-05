"""Functions for drawing publication quality figures"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.io as pio
import plotly.express as px
import os
import math
import plotly.graph_objs as go
from matplotlib.colors import ListedColormap
import matplotlib
from statannot import add_stat_annotation
from adjustText import adjust_text

def format_figure(ax,title=None,xlabel=None,ylabel=None,despine=True,detick=False):
    # Nikita's function to format figures visually appealing
    if title != None:
        ax.set_title(title, pad=5)
    if xlabel != None:
        ax.set_xlabel(xlabel, labelpad=5)
    if ylabel != None:
        ax.set_ylabel(ylabel, labelpad=5)
    if despine:
        sns.despine()
    if detick:
        plt.tick_params(left=False, right=False, labelleft=False,
            labelbottom=False, bottom=False)

def draw_custom_bar_plot(dict_datasets, directory, file_name, colors, strip_plot, test, pvalue=True, figsize=(2,2)):
    font = {'family': 'arial',
                'weight': 'normal',
                'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    fig, ax = plt.subplots(figsize=figsize)
    sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())

    ax=sns.barplot(data=sorted_vals, capsize=0.5, edgecolor='0.2', lw=1, errwidth=1, palette=colors)
    if strip_plot == True:
        plot_params={'edgecolor':'0.2', 'linewidth':1, 'fc':'none'}
        ax=sns.stripplot(data=sorted_vals, marker='s', s=1.5, **plot_params)
    # marker='s'(square), s = marker size

    #format_figure(ax, title=None, xlabel=None, ylabel=None, despine=True, detick=True)
    #ax.axhline(max_entropy, linestyle='--', linewidth=1, color='red')
    #plt.xticks(plt.xticks()[0], sorted_keys, fontsize=12, fontdict={'weight': 'bold'})
    #ax.set_xticks(plt.xticks()[0], sorted_keys, fontsize=4)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(width=1, color='0.2')
    plt.xticks(plt.xticks()[0], sorted_keys, fontsize=8, rotation=35, rotation_mode='anchor', ha='right', color='0.2', weight='bold')
    plt.yticks(fontsize=8,  color='0.2', weight='bold')
    #plt.ylabel('%s' % feature_name, fontsize=4)
    # category labels
    plt.grid(False)

    if pvalue == True:
        from scipy import stats
        from itertools import combinations
        p_values = []
        pairs = []
        for pair in combinations(range(0, len(dict_datasets)), 2):  # 2 for pairs, 3 for triplets, etc
            if test == 'mann-whitney':
                stat_test = stats.mannwhitneyu(dict_datasets[sorted_keys[pair[0]]], dict_datasets[sorted_keys[pair[1]]])
            elif test == 't-test':
                stat_test = stats.ttest_ind(dict_datasets[sorted_keys[pair[0]]], dict_datasets[sorted_keys[pair[1]]])
            elif test == 'wilcoxon-ranksum':
                stat_test = stats.ranksums(dict_datasets[sorted_keys[pair[0]]], dict_datasets[sorted_keys[pair[1]]])
            p_values.append(stat_test.pvalue)
            pairs.append(pair)
            #print(pair, stat_test.pvalue)
        plt.title('%s:%s' % (pairs, p_values), fontsize=4)

    elif pvalue == False:
        pass

    plt.savefig(directory + '%s.png' % file_name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(directory + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(directory + 'svg/')
    plt.savefig(directory + 'svg/%s.svg' % file_name, bbox_inches='tight')
    plt.clf()
    plt.close()

def draw_umap_space(df, directory, file_name, condition_name, label_name, colors, dot_size, x_name, y_name):
    ################## Draw interactive version of state space #######################

    #colors = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', '#888888')
    cmap = ListedColormap(colors[:pd.unique(df[condition_name]).shape[0]])

    xmin = math.floor(df[x_name].min()) - 1
    xmax = math.ceil(df[x_name].max()) + 1
    ymin = math.floor(df[y_name].min()) - 1
    ymax = math.ceil(df[y_name].max()) + 1

    # cmap = plt.cm.get_cmap('Set1')
    fig = px.scatter(
        data_frame=df,
        x=x_name,
        y=y_name,
        color=condition_name,
        # color_discrete_sequence = ['red','green','blue','yellow'], # label이 숫자나 bool 형태이면 color 적용이 안되는 버그가 있음
        opacity=0.9,
        template='plotly_white',
        # ggplot2, seaborn, simple_white, plotly, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, none
        # symbol = 'TrackID',
        # symbol_map = {'Control':0,'Clone A':1,'Clone B':2, 'Clone C':3},
        # title='state space',
        labels={x_name: 'UMAP1', y_name: 'UMAP2', },
        hover_data={label_name: True,
                    condition_name:True,
                    },
        hover_name=df.index,

        range_x=[xmin, xmax],
        range_y=[ymin, ymax],

        height=1000,
        width=2000,
    )

    fig.update_traces(marker=dict(size=3),
                      # line = dict(width=1, color='DarkSlateGrey')) ,
                      # selector=dict(mode='markers')
                      )
    fig.write_html(directory + '%s.html' % file_name)

    ################## Draw figure version of state space #######################

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    fig, ax = plt.subplots(figsize=(2, 2))
    #plt.figure(figsize=(15, 10))
    scatter = ax.scatter(df[x_name], df[y_name],
                          c=df[condition_name].replace(list(pd.unique(df[condition_name])),
                                                                [i for i in range(
                                                                    pd.unique(df[condition_name]).shape[0])]),
                          # replace 'wt B-cell', 'mt B-cell', 'T-cell' with 0, 1, 2 respectively
                          s=dot_size, label=df[condition_name],
                          cmap=cmap)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
    handles, labels = scatter.legend_elements(num=None)
    plt.legend(handles=handles, labels=list(pd.unique(df[condition_name])),
               bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
               fontsize=3, frameon=False, markerscale=0.3)

    # bbox_to_anchor is position of labels (x, y) (increasing x moves right, increasing y moves top)
    # frameon=False removes bounding box around label
    # font size adjust size of letter
    # markerscale adjust size of marker

    plt.savefig(directory + '%s.png' % file_name, dpi=300)

    if not os.path.isdir(directory + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(directory + 'svg/')
    plt.savefig(directory + 'svg/%s.svg' % file_name)
    plt.clf()
    plt.close()

def draw_contour(df, directory, file_name, condition_name, colors, x_name='PC1', y_name='PC2', bin_num=50, num_contours=6):
    # color_list = ['Reds', 'Greens', 'Blues', 'Greys', 'Oranges', 'Purples', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
    #               'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'plasma']
    #colors = ('Reds', 'Greys')
    #fig = plt.figure(figsize=(20, 15), dpi=300)
    fig, ax = plt.subplots(figsize=(2, 2))  # 2 inch by 2 inch

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    xmin = math.floor(df[x_name].min()) - 1
    xmax = math.ceil(df[x_name].max()) + 1
    ymin = math.floor(df[y_name].min()) - 1
    ymax = math.ceil(df[y_name].max()) + 1

    #ax = fig.add_subplot(111)
    contours = []
    groups = []

    if condition_name == None:
        x = df[x_name]
        y = df[y_name]
        kde_coordinate = np.vstack([x, y])  # shape = (2(dimension), number of points)
        if kde_coordinate.shape[1] <= 2:  # if there is only few points, it cannot calculate gaussian kde
            raise ValueError('Number of points should be greater than 2 to create contour')
        else:
            kde = scipy.stats.gaussian_kde(kde_coordinate)  # Define kernel (bandwidth by Scott's Rule)

            # evaluate on a regular grid
            xgrid = np.linspace(xmin, xmax, bin_num)  # (100, ) 1d x coordinate
            ygrid = np.linspace(ymin, ymax, bin_num)  # (100, ) 1d y coordinate
            Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
            # Xgrid , Ygrid = (bin_num,bin_num) 2d array
            # Xgrid[i] = xgrid coordinate(divide by 100 from -x ~ +x` and repeat in row direction)
            # Ygrid[:,i] = ygrid coordinate(divide by 100 from -y ~ +y` and repeat in column direction)
            Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
            # Xgrid.ravel() = bin_num^2 shape 1d vector (vector that repeat -x ~ +x`, -x ~ +x` bin_num times)
            # np.vstack() = (2,bin_num^2) 2d array linspaced coordinates
            # Z = (10000,) 1d vector
            pdf = Z.reshape(Xgrid.shape)
            contour = ax.contour(Xgrid, Ygrid, pdf,
                                 # colors='red',
                                 linewidths=1,
                                 linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
                                 # label=group,
                                 cmap='Reds',
                                 origin='lower',
                                 levels=num_contours,
                                 )
            contours.append(contour)

        format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig(directory + '/%s.png' % (file_name), dpi=300)

        if not os.path.isdir(directory + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(directory + 'svg/')
        plt.savefig(directory + 'svg/%s.svg' % (file_name))
        plt.close()
        plt.clf()

    else:
        for i, group in enumerate(list(pd.unique(df[condition_name]))):
            x = df[df[condition_name] == group][x_name]
            y = df[df[condition_name] == group][y_name]

            kde_coordinate = np.vstack([x, y])  # shape = (2(dimension), number of points)
            if kde_coordinate.shape[1] <= 2: # if there is only few points, it cannot calculate gaussian kde
                continue
            else:
                kde = scipy.stats.gaussian_kde(kde_coordinate)  # Define kernel (bandwidth by Scott's Rule)

                # evaluate on a regular grid
                xgrid = np.linspace(xmin, xmax, bin_num)  # (100, ) 1d x coordinate
                ygrid = np.linspace(ymin, ymax, bin_num)  # (100, ) 1d y coordinate
                Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
                # Xgrid , Ygrid = (bin_num,bin_num) 2d array
                # Xgrid[i] = xgrid coordinate(divide by 100 from -x ~ +x` and repeat in row direction)
                # Ygrid[:,i] = ygrid coordinate(divide by 100 from -y ~ +y` and repeat in column direction)
                Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
                # Xgrid.ravel() = bin_num^2 shape 1d vector (vector that repeat -x ~ +x`, -x ~ +x` bin_num times)
                # np.vstack() = (2,bin_num^2) 2d array linspaced coordinates
                # Z = (10000,) 1d vector
                pdf = Z.reshape(Xgrid.shape)
                contour = ax.contour(Xgrid, Ygrid, pdf,
                            # colors='red',
                            linewidths=1,
                            linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
                            #label=group,
                            #cmap=colors[i],
                            colors=colors[i],
                            origin='lower',
                            levels=num_contours,
                            )
                contours.append(contour)
                groups.append(group)

        format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
        ax.legend(handles=[contour.legend_elements()[0][-1] for contour in contours], labels=groups,
                  bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0, fontsize=3, frameon=False, markerscale=0.3)
        #ax.legend(handles=[contour.legend_elements()[0][-1] for contour in contours], labels=groups, fontsize=3, markerscale=0.3)

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig(directory + '/%s.png' % (file_name), dpi=300)

        if not os.path.isdir(directory + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(directory + 'svg/')
        plt.savefig(directory + 'svg/%s.svg' % (file_name))
        plt.close()
        plt.clf()

def draw_custom_violin_plot(dict_datasets, directory, file_name, colors, test, pvalue=True, figsize=(2,2)):

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.violinplot(data=sorted_vals, palette=colors, linewidth=1, linecolor="0.2", inner="box",
                        inner_kws=dict(box_width=10, whis_width=10, color="0.2"))

    # marker='s'(square), s = marker size

    # format_figure(ax, title=None, xlabel=None, ylabel=None, despine=True, detick=True)
    # ax.axhline(max_entropy, linestyle='--', linewidth=1, color='red')
    # plt.xticks(plt.xticks()[0], sorted_keys, fontsize=12, fontdict={'weight': 'bold'})
    # ax.set_xticks(plt.xticks()[0], sorted_keys, fontsize=4)

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(width=1, color='0.2')
    #ax.set_ylabel(feature_name, fontsize=8, weight='bold')
    plt.xticks(plt.xticks()[0], sorted_keys, fontsize=8, rotation=35, rotation_mode='anchor', ha='right', color='0.2',
               weight='bold')
    plt.yticks(fontsize=8, color='0.2', weight='bold')

    if pvalue == True:
        from scipy import stats
        from itertools import combinations
        p_values = []
        pairs = []
        for pair in combinations(range(0, len(dict_datasets)), 2):  # 2 for pairs, 3 for triplets, etc
            if test == 'mann-whitney':
                stat_test = stats.mannwhitneyu(dict_datasets[sorted_keys[pair[0]]], dict_datasets[sorted_keys[pair[1]]])
            elif test == 't-test':
                stat_test = stats.ttest_ind(dict_datasets[sorted_keys[pair[0]]], dict_datasets[sorted_keys[pair[1]]])
            elif test == 'wilcoxon-ranksum':
                stat_test = stats.ranksums(dict_datasets[sorted_keys[pair[0]]], dict_datasets[sorted_keys[pair[1]]])
            p_values.append(stat_test.pvalue)
            pairs.append(pair)
            # print(pair, stat_test.pvalue)
        plt.title('%s:%s' % (pairs, p_values), fontsize=4)

    elif pvalue == False:
        pass

    plt.savefig(directory + '%s.png' % file_name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(directory + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(directory + 'svg/')
    plt.savefig(directory + 'svg/%s.svg' % file_name, bbox_inches='tight')
    plt.clf()
    plt.close()