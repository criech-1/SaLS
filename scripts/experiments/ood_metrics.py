"""Script to evaluate PNN on the CL metrics."""

import argparse
import sklearn.metrics as sk

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision.transforms as transforms

from copy import deepcopy

from src.SaLS.pnn import ProgressiveNeuralNetwork, ClassifierNet
from src.SaLS.utils import fit_pnn, seed_all_rng
from tools.dataloaders import get_split_TOROdataset
from tools.metrics import plot_histogram_metric, plot_full_histogram

# Tasks to evaluate
tasks = ['dark_mat', 'lab_floor', 'stones', 'ramp', 'blue_mattress', 'pedestal']

# OOD methods to test
OOD_method = ['MSP', 'MaxLogit']

def ood_metrics(args):
    print('\x1b[1;37;47m' + '* OOD METRICS *' + '\x1b[0m')
    print('*'*120)

    # Create folders to save results
    os.makedirs('results/exp_ood_metrics', exist_ok=True)

    # Use cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load DINOv2 model from torch hub or local path if net not available
    try:
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
        print('\x1b[1;30;42m' + '>> DINOv2 model loaded from hub' + '\x1b[0m')
    except: 
        dino_model = torch.hub.load('dinov2', 'dinov2_vits14', source='local', pretrained=False)
        dino_model.load_state_dict(torch.load('./Segment-and-Track-Anything/ckpt/dinov2_vits14_pretrain.pth'))
        dino_model.to(device).eval()
        print('\x1b[1;30;42m' + '>> DINOv2 model loaded locally' + '\x1b[0m')
    
    print('*'*120)

    # Initialize PNN
    last_layer_name = 'fc'
    lateral_connections = None

    # PNN parameters
    parameters = {
        'learning_rate': 0.001,
        'weight_decay': 1e-8,
        'patience': 20,
        'num_epochs': 50,
    }

    # Data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(280),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(312),
            transforms.CenterCrop(280),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(312), # 256
            transforms.CenterCrop(280), # 224
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    global_metrics = []

    for iter in range(args.iter):
        # Load base network
        network = ClassifierNet(input_size = list(dino_model.children())[-2].normalized_shape[0],
                                output_size = 2) # 384 if DINOv2 with ViT-Small
        base_network = deepcopy(network).to(device)

        # Create PNN model for each iteration
        pnn_model = ProgressiveNeuralNetwork(base_network = base_network,
                                             backbone = dino_model,
                                             last_layer_name = last_layer_name,
                                             lateral_connections = deepcopy(lateral_connections)).to(device)

        # Choose random sequences for training and testing
        train_dataloaders, val_dataloaders, test_dataloaders = get_split_TOROdataset(dataset_path=args.dataset_path,
                                                                                     tasks=tasks,
                                                                                     train_ratio=args.train_ratio,
                                                                                     data_transforms=data_transforms,
                                                                                     batch_size=1,
                                                                                     num_workers=0,
                                                                                     max_data_seq=20,
                                                                                     randomize=True,)
        # Initialize globel metrics
        metrics = {'MSP_scores': [],
                   'MaxLogit_scores': [],}

        # For every task
        for i, task_i in enumerate(tasks):
            # Train PNN
            train_val_metrics = fit_pnn(model = pnn_model, 
                                        dataloader_train = train_dataloaders[i],
                                        dataloader_val = val_dataloaders[i],
                                        parameters = parameters,
                                        device = device,
                                        name = 'exp_ood_metrics_' + str(args.id),)
            
            MSP_score_task_i = []
            MaxLogit_score_task_i = []

            pnn_model.eval()
            for j, task_j in enumerate(tasks):
                # print('>> Task: {} \t OOD: {}'.format(task_i, task_j))

                MSP_score_task_j = []
                MaxLogit_score_task_j = []

                # Get test maxlogits
                for inputs, labels in test_dataloaders[j]:
                    inputs, labels = inputs.to(device), labels.to(device)

                    with torch.no_grad():
                        logits = pnn_model(inputs)

                        logit = logits[i]
                        maxlogit = torch.max(logit, dim=1)[0]

                        MSP_score_task_j.append(torch.max(torch.softmax(logit, dim=1),dim=1)[0].cpu().numpy()[0])
                        MaxLogit_score_task_j.append(maxlogit.cpu().numpy()[0])

                MSP_score_task_i.append(MSP_score_task_j)
                MaxLogit_score_task_i.append(MaxLogit_score_task_j)

            metrics['MSP_scores'].append(MSP_score_task_i)
            metrics['MaxLogit_scores'].append(MaxLogit_score_task_i)

        global_metrics.append(metrics)

    # Save metrics
    np.save('results/exp_ood_metrics/' + str(args.id) + '.npy', global_metrics)

def get_ood_metrics(out_scores, in_scores, recall_level=0.95):
    pos = np.array(out_scores[:]).reshape((-1, 1))
    neg = np.array(in_scores[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def get_tasks_ood(tasks, anomaly_scores):
    auroc_arr, aupr_arr, fpr_arr = [], [], []
    
    for i, task_i in enumerate(tasks):
        in_scores = -np.array(anomaly_scores[i][i])

        out_scores = [anomaly_scores[i][j] for j in range(len(tasks)) if j != i]
        out_scores = -np.concatenate(out_scores)

        auroc, aupr, fpr = get_ood_metrics(out_scores, in_scores, recall_level=0.95)

        auroc_arr.append(auroc)
        aupr_arr.append(aupr)
        fpr_arr.append(fpr)

    return np.mean(auroc_arr), np.mean(aupr_arr), np.mean(fpr_arr)
    # return auroc_arr, aupr_arr, fpr_arr    

def ood_evaluation(args):
    print('\x1b[1;37;47m' + '* OOD EVALUATION *' + '\x1b[0m')

    metrics_path = 'results/exp_ood_metrics_CE'

    # Create folders to save results
    plots_path = 'results/plots/exp_ood_metrics'
    os.makedirs(plots_path, exist_ok=True)

    # Initialize empty vector with as many positions as OOD methods
    anomaly_results = [[] for i in range(len(OOD_method))]

    for seed in os.listdir(metrics_path):
        metrics = np.load(os.path.join(metrics_path, seed), allow_pickle=True)
        for iter in metrics:
            anomaly_scores_ood = [iter['MSP_scores'], iter['MaxLogit_scores']]
            for i, ood in enumerate(OOD_method):
                auroc, aupr, fpr = get_tasks_ood(tasks, anomaly_scores_ood[i])
                anomaly_results[i].append([auroc, aupr, fpr])
                # print('FPR95 (down): {:.4f} \t AUROC (up): {:.4f} \t AUPR (up): {:.4f}'.format(auroc, aupr, fpr))
    
    mean_anomaly_results = np.mean(np.array(anomaly_results), axis=1)
    std_anomaly_results = np.std(np.array(anomaly_results), axis=1)

    print('\x1b[1;37;42m' + '>> Results:' + '\x1b[0m')
    print('             | MSP              | MaxLogit')

    fpr_str = ' | '.join(['{:.4f} +- {:.4f}'.format(mean_anomaly_results[i,2], std_anomaly_results[i,2]) for i in range(len(OOD_method))])
    print('FPR95 (down) | ' + fpr_str)

    auroc_str = ' | '.join(['{:.4f} +- {:.4f}'.format(mean_anomaly_results[i,0], std_anomaly_results[i,0]) for i in range(len(OOD_method))])
    print('AUROC (up)   | '  + auroc_str)
    
    aupr_str = ' | '.join(['{:.4f} +- {:.4f}'.format(mean_anomaly_results[i,1], std_anomaly_results[i,1]) for i in range(len(OOD_method))])
    print('AUPR (up)    | ' + aupr_str)

    

def ood_values(args):
    print('\x1b[1;37;47m' + '* OOD VALUES *' + '\x1b[0m')
    print('*'*120)

    results_path = 'results/exp_ood_metrics'
    # check if ood_values.npy exists
    if os.path.exists(os.path.join(results_path, 'ood_values_{}.npy'.format(args.id))):
        metrics = np.load(os.path.join(results_path, 'ood_values_{}.npy'.format(args.id)), allow_pickle=True).item()
    else:
        os.makedirs(results_path, exist_ok=True)

        # Use cpu or gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load DINOv2 model from torch hub or local path if net not available
        try:
            dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
            print('\x1b[1;30;42m' + '>> DINOv2 model loaded from hub' + '\x1b[0m')
        except: 
            dino_model = torch.hub.load('dinov2', 'dinov2_vits14', source='local', pretrained=False)
            dino_model.load_state_dict(torch.load('./Segment-and-Track-Anything/ckpt/dinov2_vits14_pretrain.pth'))
            dino_model.to(device).eval()
            print('\x1b[1;30;42m' + '>> DINOv2 model loaded locally' + '\x1b[0m')
        
        print('*'*120)

        # Initialize PNN
        last_layer_name = 'fc'
        lateral_connections = None

        # PNN parameters
        parameters = {
            'learning_rate': 0.001,
            'weight_decay': 1e-8,
            'patience': 20,
            'num_epochs': 50,
        }

        # Data transforms
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(280),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(312),
                transforms.CenterCrop(280),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(312), # 256
                transforms.CenterCrop(280), # 224
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # Load base network
        network = ClassifierNet(input_size = list(dino_model.children())[-2].normalized_shape[0],
                                output_size = 1) # 384 if DINOv2 with ViT-Small
        base_network = deepcopy(network).to(device)

        # Create PNN model for each iteration
        pnn_model = ProgressiveNeuralNetwork(base_network = base_network,
                                                backbone = dino_model,
                                                last_layer_name = last_layer_name,
                                                lateral_connections = deepcopy(lateral_connections)).to(device)

        # Choose random sequences for training and testing
        train_dataloaders, val_dataloaders, test_dataloaders = get_split_TOROdataset(dataset_path=args.dataset_path,
                                                                                        tasks=tasks,
                                                                                        train_ratio=args.train_ratio,
                                                                                        data_transforms=data_transforms,
                                                                                        batch_size=1,
                                                                                        num_workers=0,
                                                                                        max_data_seq=20,
                                                                                        randomize=True,)
        # Initialize globel metrics
        metrics = {'val_scores': [],
                    'test_scores': [],}

        # For every task
        for i, task_i in enumerate(tasks):
            # Train PNN
            train_val_metrics = fit_pnn(model = pnn_model, 
                                        dataloader_train = train_dataloaders[i],
                                        dataloader_val = val_dataloaders[i],
                                        parameters = parameters,
                                        device = device,
                                        name = None)

            metrics['val_scores'].append(train_val_metrics['val'][-1]['logits'])

            test_logits = []
            # Get test maxlogits
            for inputs, labels in test_dataloaders[i]:
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.no_grad():
                    logits = pnn_model(inputs)

                    logit = logits[i]
                    maxlogit = torch.max(logit, dim=1)[0]
                    test_logits.append(maxlogit.cpu().numpy()[0])

            metrics['test_scores'].append(test_logits)

    
        np.save(os.path.join(results_path, 'ood_values_{}.npy'.format(args.id)), metrics)

    metric_val = []
    metric_test = []

    for i,task in enumerate(tasks):
        metric_val.append(np.array(metrics['val_scores'][i]))
        metric_test.append(np.array(metrics['test_scores'][i]))

    # metric_val = np.array(metric_val)
    # metric_test = np.array(metric_test)

    plots_path = 'results/plots/exp_ood_metrics'
    os.makedirs(plots_path, exist_ok=True)

    fig, axs = plot_full_histogram(metric_val=metric_val, 
                                   metric_test=metric_test, 
                                   num_bins=25, 
                                   x_label='MaxLogit values',
                                   tasks=tasks)
    fig.savefig(os.path.join(plots_path, 'ood_values.png'), bbox_inches='tight')

    for i, task in enumerate(tasks):
        val_maxlogits = np.array(metrics['val_scores'][i])
        test_maxlogits = np.array(metrics['test_scores'][i])

        fig, axs = plot_histogram_metric(metric_val=val_maxlogits,
                                         metric_test=test_maxlogits,
                                         num_bins=25,
                                         x_label='MaxLogit values',
                                         title='"{}"'.format(task))
        fig.savefig(os.path.join(plots_path, 'ood_{}.png'.format(task)), bbox_inches='tight')


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(prog='Continual Semantic Segmentation',
                                     description='Description of the program',
                                     epilog='End of the program description')
    parser.add_argument('--id', type=int, required=True, help='experiment id and seed')
    parser.add_argument('--dataset_path', type=str, required=True, help='dataset directory')
    parser.add_argument('--train_ratio', type=float, default=0.7, required=False, help='train ratio')
    parser.add_argument('--iter', type=int, default=2, required=False, help='number of iterations')

    parser.add_argument('--eval', action='store_true', help='evaluate PNN with existing metrics')
    parser.add_argument('--no-eval', dest='eval', action='store_false', help='get PNN metrics')
    parser.set_defaults(eval=False)
    args = parser.parse_args()

    # Set manual seed
    seed_all_rng(args.id)

    if args.eval:
        ood_evaluation(args)
        ood_values(args)
    else:
        ood_metrics(args)