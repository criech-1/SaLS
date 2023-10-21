import matplotlib.pyplot as plt
import numpy as np

from typing import List, Optional

plt.rcParams.update({'font.size': 14})

def plot_metric(tasks: List,
                metric_avg: np.array,
                metric_label: str,
                fig_size: Optional[int] = 5,
                metric_std: Optional[np.array] = None,):
    num_tasks = len(tasks)        

    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(fig_size,fig_size))
    # fig.suptitle(title, y=0.92)

    axs.set_xticks(np.arange(0.5, num_tasks + 0.5))
    task_label = ['{}'.format(i-1) for i in range(1, num_tasks + 1)]
    axs.set_xticklabels(task_label)

    xs = np.arange(num_tasks) + 0.5
    axs.set_ylabel(metric_label)
    axs.set_xlim(0, num_tasks)
    axs.set_xlabel('Trained task ID')   

    if metric_avg.ndim == 1:
        axs.plot(xs, metric_avg, label=tasks, marker='o', markersize=5, linewidth=2)
        axs.fill_between(xs, metric_avg - metric_std, metric_avg + metric_std, alpha=0.2, label='_nolegend_')
    else:
        for i in range(metric_avg.shape[0]):
            axs.plot(xs, metric_avg[i], label=tasks[i], marker='o', markersize=5, linewidth=2)
            axs.fill_between(xs, metric_avg[i] - metric_std[i], metric_avg[i] + metric_std[i], alpha=0.2, label='_nolegend_')

    for i in range(1,num_tasks):
        axs.axvline(i, linestyle='dotted', color='gray', clip_on=False)

    # plt.show()

    return fig, axs

def plot_metricx_metricy(x_metric: np.array,
                         x_label: str,
                         y_metric: np.array,
                         y_label: str,
                         fig_size: Optional[int] = 5,
                         y_std: Optional[np.array] = None,):
    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(fig_size, fig_size))
    
    if len(y_metric.shape) == 1:
        axs.plot(x_metric, y_metric, marker='o', markersize=5, linewidth=2)
        # axs.plot(x_metric, y_metric, linewidth=2)
        if y_std is not None:
            axs.fill_between(x_metric, y_metric - y_std, y_metric + y_std, alpha=0.2, label='_nolegend_')
    else:
        for i in range(y_metric.shape[0]):
            axs.plot(x_metric, y_metric[i], marker='o', markersize=5, linewidth=2)
            if y_std is not None:
                axs.fill_between(x_metric, y_metric[i] - y_std[i], y_metric[i] + y_std[i], alpha=0.2, label='_nolegend_')
    axs.set_ylabel(y_label)
    axs.set_xlabel(x_label)
    axs.set_ylim(ymin=0)

    # axs.set_xlim(0, np.max(x_metric)*1.05)
    axs.set_xlim(np.min(x_metric), np.max(x_metric))

    plt.grid()
    # plt.show()

    return fig, axs

def plot_metric_task(tasks: List,
                     metric_avg: np.array,
                     metric_std: Optional[np.array] = None,
                     fig_size: Optional[int] = 10,):
    
    num_tasks = len(tasks)

    fig, axs = plt.subplots(num_tasks, 1, sharex=True, sharey=True, figsize=(fig_size, fig_size))
    # fig.suptitle(title, y=0.92)

    axs[-1].set_xticks(np.arange(0.5, num_tasks + 0.5))
    task_label= ['{}'.format(i) for i in range(num_tasks)]
    axs[-1].set_xticklabels(task_label)
    axs[-1].set_xlabel('Trained task ID')

    for i, ax in enumerate(axs):
        if ax != axs[-1]:
            ax.xaxis.set_visible(False)
        ax.set_xlim(0, num_tasks)
        spines = ['right', 'top']
        if i < len(axs) - 1:
            spines.append('bottom')
        for spine in spines:
            ax.spines[spine].set_visible(False)

    for i in range(1,num_tasks):
        axs[0].axvline(i, linestyle='dotted', color='gray', clip_on=False)
        for ax in axs[1:]:
            ax.axvline(i, linestyle='dotted', color='gray', clip_on=False)

    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    xs = np.arange(num_tasks) + 0.5
    for i in range(num_tasks):
        axs[i].plot(xs, metric_avg[:, i], color='C'+str(i), label=tasks[i], marker='o', markersize=5, linewidth=2)   
        axs[i].set_ylabel(tasks[i] + ': {}'.format(i), rotation=0, horizontalalignment = 'right', labelpad=10)

        if metric_std is not None:
            axs[i].fill_between(xs, metric_avg[:, i] - metric_std[:, i], metric_avg[:, i] + metric_std[:, i], color='C'+str(i), alpha=0.2)

    for i in range(1, num_tasks + 1):
        axs[0].axvline(i, linestyle='dotted', color='gray', clip_on=False)
        for ax in axs:
            ax.axvline(i, ymax=1.2, linestyle='dotted', color='gray', clip_on=False)
    # plt.show()

    return fig, axs

def plot_histogram_metric(metric_val: np.array,
                          metric_test: np.array,
                          num_bins: int,
                          x_label: str,
                          title: Optional[str] = None):

        mean_val = np.mean(metric_val)
        std_val = np.std(metric_val)
        mean_test = np.mean(metric_test)
        std_test = np.std(metric_test)

        min_metric = np.min([np.min(metric_val), np.min(metric_test)])
        max_metric = np.max([np.max(metric_val), np.max(metric_test)])

        bins = np.linspace(min_metric, max_metric, num_bins)

        fig = plt.figure(figsize=(5, 5))
        gs = fig.add_gridspec(2, hspace=0)
        axs = gs.subplots(sharex=True, sharey=True)
        if title is not None:
            fig.suptitle(title, y=0.95)

        counts_val = axs[0].hist(metric_val, bins=bins, color='b')
        axs[0].axvline(x=mean_val, color='b', linestyle='--')
        axs[0].fill_betweenx([0, 100], mean_val - 1 * std_val, mean_val + 1 * std_val, color='b', alpha=0.2)
        axs[0].fill_betweenx([0, 100], mean_val - 2 * std_val, mean_val + 2 * std_val, color='b', alpha=0.2)
        # axs[0].fill_betweenx([0, 10], mean_val - 3 * std_val, mean_val + 3 * std_val, color='b', alpha=0.2)

        counts_test = axs[1].hist(metric_test, bins=bins, color='r')    
        axs[1].axvline(x=mean_test, color='r', linestyle='--')
        axs[1].fill_betweenx([0, 100], mean_test - 1 * std_test, mean_test + 1 * std_test, color='r', alpha=0.2)
        axs[1].fill_betweenx([0, 100], mean_test - 2 * std_test, mean_test + 2 * std_test, color='r', alpha=0.2)
        # axs[1].fill_betweenx([0, 10], mean_test - 3 * std_test, mean_test + 3 * std_test, color='r', alpha=0.2)

        axs[1].set_xlabel(x_label)
        max_count = np.max([np.max(counts_val[0]), np.max(counts_test[0])])*1.1
        axs[0].set_ylim(0, max_count)
        axs[0].set_ylabel('Validation samples')
        axs[1].set_ylim(0, max_count)
        axs[1].set_ylabel('Test samples')

        return fig, axs

def plot_full_histogram(metric_val: List, 
                            metric_test: List, 
                            num_bins: int, 
                            x_label: str,
                            tasks: List[str]):

        fig = plt.figure(figsize=(5, 5))
        gs = fig.add_gridspec(2, hspace=0)
        axs = gs.subplots(sharex=True, sharey=True)
        # fig.suptitle(title, y=0.92)

        # Detect min in the whole array
        min_metric_val = np.min([np.min(metric_val[i]) for i in range(len(tasks))])
        min_metric_test = np.min([np.min(metric_test[i]) for i in range(len(tasks))])
        min_metric = np.min([min_metric_val, min_metric_test])

        # Detect max in the whole array
        max_metric_val = np.max([np.max(metric_val[i]) for i in range(len(tasks))])
        max_metric_test = np.max([np.max(metric_test[i]) for i in range(len(tasks))])
        max_metric = np.max([max_metric_val, max_metric_test])

        bins = np.linspace(min_metric, max_metric, num_bins)
        colors = ['C' + str(i) for i in range(len(tasks))]

        counts_val = axs[0].hist(metric_val, bins=bins, color=colors, stacked=True)
        counts_test = axs[1].hist(metric_test, bins=bins, color=colors, stacked=True)   

        axs[1].set_xlabel(x_label)
        max_count = np.max([np.max(counts_val[0]), np.max(counts_test[0])])*1.1
        axs[0].set_ylim(0, max_count)
        axs[0].set_ylabel('Validation samples')
        axs[1].set_ylim(0, max_count)
        axs[1].set_ylabel('Test samples')

        fig.legend(tasks)

        return fig, axs