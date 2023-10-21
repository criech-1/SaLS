"""Script to compare different data transforms."""

import argparse
import numpy as np
import os
import torch
import torchvision.transforms as transforms

from copy import deepcopy

from src.SaLS.pnn import ProgressiveNeuralNetwork, ClassifierNet
from src.SaLS.utils import fit_pnn, seed_all_rng

from tools.inference import predict_dataloader
from tools.dataloaders import get_split_TOROdataset
from tools.metrics import plot_metricx_metricy

# Tasks to evaluate
tasks = ['dark_mat', 'lab_floor', 'stones', 'ramp', 'blue_mattress', 'pedestal']
num_tasks = len(tasks)

def transform_metrics(args):
    print('\x1b[1;37;47m' + '* TRANSFORM METRICS *' + '\x1b[0m')
    print('*'*120)

    # Create folders to save results
    os.makedirs('results/exp_transform_metrics', exist_ok=True)

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

    # Load base network
    network = ClassifierNet(input_size = list(dino_model.children())[-2].normalized_shape[0],
                            output_size=1) 
    base_network = deepcopy(network).to(device)

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

    # Transform crop sizes
    crop_sizes = [14*n for n in [1, 10, 20, 30, 40]]
    # crop_sizes = [14*n for n in [1, 20, 40]]

    # Global metrics
    global_metrics = []

    for crop_size in crop_sizes:
        print('\x1b[1;37;43m' + '>> Crop size: {}'.format(crop_size) + '\x1b[0m')
        # Data transforms
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(crop_size+32),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(crop_size+32), # 256
                transforms.CenterCrop(crop_size), # 224
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        for iter in range(args.iter):
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
            metrics = {'accuracy': [], 
                       'train_time': [],
                       'used_GPU': [], 
                       'test_time': [], 
                       'crop_size': []}

            # For every task
            for i, task_i in enumerate(tasks):
                accuracy = []
                test_time = []

                # Measure training time
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                # Train PNN
                fit_pnn(model = pnn_model, 
                        dataloader_train = train_dataloaders[i],
                        dataloader_val = val_dataloaders[i],
                        parameters = parameters,
                        device = device,
                        name = 'exp_transform_metrics_' + str(args.id),)
                
                end.record()
                torch.cuda.synchronize()
                train_time = start.elapsed_time(end)

                metrics['train_time'].append(train_time)

                # Get GPU utilization
                utilization_GPU = torch.cuda.mem_get_info()
                used_GPU = utilization_GPU[1] - utilization_GPU[0]
                metrics['used_GPU'].append(used_GPU)

                # Test with every task
                for j, task_j in enumerate(tasks):
                    # Measure inference time
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()

                    pred_labels = predict_dataloader(model=pnn_model, 
                                                     dataloader=test_dataloaders[j],
                                                     device=device)

                    # Compute accuracy
                    accuracy_j = np.sum(np.array(pred_labels) == task_j) / len(pred_labels)
                    print('>> Accuracy task {}: {}'.format(j, accuracy_j))

                    end.record()
                    torch.cuda.synchronize()
                    test_time_j = start.elapsed_time(end)

                    # Save metrics
                    accuracy.append(accuracy_j)
                    test_time.append(test_time_j)

                metrics['accuracy'].append(accuracy)
                metrics['test_time'].append(test_time)
                metrics['crop_size'].append(crop_size)

            global_metrics.append(metrics)

    # Save metrics
    np.save('results/exp_transform_metrics/' + str(args.id) + '.npy', global_metrics)

def transform_evaluation():
    print('\x1b[1;37;47m' + '* TRANSFORMS EVALUATION *' + '\x1b[0m')
    print('*'*120)

    # Create folders to save results
    plots_path = 'results/plots/exp_transform_metrics'
    os.makedirs(plots_path, exist_ok=True)

    # Load metrics
    list_metrics = os.listdir('results/exp_transform_metrics')

    # Global metric variables
    accuracy = []
    train_time = []
    test_time = []
    used_GPU = []
    crop_size = []

    for metric in list_metrics:
        metrics = np.load('results/exp_transform_metrics/' + metric, allow_pickle=True)

        for iter in metrics:
            # Get experiments metrics
            accuracy.append(iter['accuracy'])
            train_time.append(iter['train_time'])
            test_time.append(iter['test_time'])
            used_GPU.append(iter['used_GPU'])
            crop_size.append(iter['crop_size'])

    accuracy = np.array(accuracy)
    train_time = np.array(train_time)
    test_time = np.array(test_time)
    used_GPU = np.array(used_GPU)
    crop_size = np.array(crop_size)

    # Variables to plot
    avg_acc_cs = []
    std_acc_cs = []
    avg_train_time_cs = []
    std_train_time_cs = []
    avg_test_time_cs = []
    std_test_time_cs = []
    avg_used_GPU_cs = []
    std_used_GPU_cs = []

    # group all metrics according to the crop size
    unique_crop_size = np.unique(crop_size)

    for cs in unique_crop_size:
        # Get index of experiments with the same crop size
        idx = np.where(crop_size == cs)[0]

        # Get metrics for the same crop size
        accuracy_cs = accuracy[idx]
        train_time_cs = train_time[idx]
        test_time_cs = test_time[idx]
        used_GPU_cs = used_GPU[idx]

        # Get global averages and std
        accuracy_avg = np.mean(accuracy_cs, axis=0)
        accuracy_std = np.std(accuracy_cs, axis=0)
        train_time_avg = np.mean(train_time_cs, axis=0)
        train_time_std = np.std(train_time_cs, axis=0)
        test_time_avg = np.mean(test_time_cs, axis=0)
        test_time_std = np.std(test_time_cs, axis=0)
        used_GPU_avg = np.mean(used_GPU_cs, axis=0)
        used_GPU_std = np.std(used_GPU_cs, axis=0)

        # Average accuracy
        std_array_accuracy = []
        average_accuracy = 0
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i >= j:
                    average_accuracy += accuracy_avg[i,j]
                    std_array_accuracy.append(accuracy_avg[i,j])

        average_accuracy /= (num_tasks * (num_tasks + 1) / 2)
        std_accuracy = np.std(std_array_accuracy)

        avg_acc_cs.append(average_accuracy)
        std_acc_cs.append(std_accuracy)

        # Average train time
        avg_train_time_cs.append(np.mean(train_time_avg))
        std_train_time_cs.append(np.std(train_time_avg))

        # Average test time
        avg_test_time_cs.append(np.mean(test_time_avg))
        std_test_time_cs.append(np.std(test_time_avg))

        # Average used GPU
        avg_used_GPU_cs.append(np.mean(used_GPU_avg)) # GB
        std_used_GPU_cs.append(np.std(used_GPU_std)) # GB

    # Convert to numpy arrays
    avg_acc_cs = np.array(avg_acc_cs)*100
    std_acc_cs = np.array(std_acc_cs)*100
    avg_train_time_cs = np.array(avg_train_time_cs)/1000
    std_train_time_cs = np.array(std_train_time_cs)/1000
    avg_test_time_cs = np.array(avg_test_time_cs)/1000
    std_test_time_cs = np.array(std_test_time_cs)/1000
    avg_used_GPU_cs = np.array(avg_used_GPU_cs)/1024/1024/1024
    std_used_GPU_cs = np.array(std_used_GPU_cs)/1024/1024/1024

    # Plot average accuracy per crop size
    fig, ax = plot_metricx_metricy(x_metric=unique_crop_size,
                                   x_label='Crop size (pixels)',
                                   y_metric=avg_acc_cs,
                                   y_label='Average accuracy (%)',
                                   y_std=std_acc_cs,)
    ax.set_ylim([0, 100])
    fig.savefig(os.path.join(plots_path, 'crop_size_accuracy.png'), bbox_inches='tight')

    # Plot average train time per crop size
    fig, ax = plot_metricx_metricy(x_metric=unique_crop_size,
                                   x_label='Crop size (pixels)',
                                   y_metric=avg_train_time_cs,
                                   y_label='Training time (s)',
                                   y_std=std_train_time_cs,)
    fig.savefig(os.path.join(plots_path, 'crop_size_train_time.png'), bbox_inches='tight')

    # Plot average test time per crop size
    fig, ax = plot_metricx_metricy(x_metric=unique_crop_size,
                                   x_label='Crop size (pixels)',
                                   y_metric=avg_test_time_cs,
                                   y_label='Inference time (s)',
                                   y_std=std_test_time_cs,)
    fig.savefig(os.path.join(plots_path, 'crop_size_test_time.png'), bbox_inches='tight')

    # Plot average used GPU per crop size
    fig, ax = plot_metricx_metricy(x_metric=unique_crop_size,
                                   x_label='Crop size (pixels)',
                                   y_metric=avg_used_GPU_cs,
                                   y_label='GPU utilisation (GB)',
                                   y_std=std_used_GPU_cs,)
    fig.savefig(os.path.join(plots_path, 'crop_size_used_GPU.png'), bbox_inches='tight')

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
        transform_evaluation()
    else:
        transform_metrics(args)