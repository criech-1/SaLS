"""Script to evaluate PNN on the CL metrics."""

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
from tools.metrics import plot_metric_task, plot_metric

tasks = ['dark_mat', 'lab_floor', 'stones', 'ramp', 'blue_mattress', 'pedestal']

def cl_metrics(args):
    print('\x1b[1;37;47m' + '* PNN METRICS *' + '\x1b[0m')
    print('*'*120)

    # Create folders to save results
    os.makedirs('results/exp_cl_metrics', exist_ok=True)

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
                            output_size=1) # 384 if DINOv2 with ViT-Small
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

    # Global metrics
    global_metrics = []

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
                   'ood_accuracy': [],
                   'train_time': [], 
                   'test_time': [], 
                   'num_params': [],
                   'used_GPU': []}

        ood_acc = 0
        # For every task
        for i, task_i in enumerate(tasks):
            accuracy = []
            ood_accuracy = []
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
                    name = 'exp_cl_metrics_' + str(args.id),
                    retrain=True)
            
            end.record()
            torch.cuda.synchronize()
            train_time = start.elapsed_time(end)

             # Get GPU utilization
            utilization_GPU = torch.cuda.mem_get_info()
            used_GPU = utilization_GPU[1] - utilization_GPU[0]
            metrics['used_GPU'].append(used_GPU)

            # Get number of parameters of model
            num_params = sum(p.numel() for p in pnn_model.parameters())

            metrics['train_time'].append(train_time)
            metrics['num_params'].append(num_params)

            # Test with every task
            for j, task_j in enumerate(tasks):
                # Measure inference time
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                pred_labels = predict_dataloader(model=pnn_model, 
                                                 dataloader=test_dataloaders[j],
                                                 device=device)
                
                end.record()
                torch.cuda.synchronize()
                test_time_j = start.elapsed_time(end)

                # Compute accuracy
                accuracy_j = np.sum(np.array(pred_labels) == task_j) / len(pred_labels)
                print('>> Accuracy task {}: {}'.format(j, accuracy_j))

                if j > i:
                    ood_accuracy_j = np.sum(np.array(pred_labels) == "Unknown") / len(pred_labels)
                    print('>> OOD Accuracy task {}: {}'.format(j, ood_accuracy_j))
                else:
                    ood_accuracy_j = 0

                # Save metrics
                accuracy.append(accuracy_j)
                ood_accuracy.append(ood_accuracy_j)
                test_time.append(test_time_j)

            metrics['accuracy'].append(accuracy)
            metrics['ood_accuracy'].append(ood_accuracy)
            metrics['test_time'].append(test_time)

        print('>> Acurracies:')
        print(np.array(metrics['accuracy']))
        print('>> Train time [s]:')
        print(np.array(metrics['train_time'])/1000)
        print('>> Test time [s]:')
        print(np.array(metrics['test_time'])/1000)
        print('>> Number of parameters:')
        print(np.array(metrics['num_params']))

        num_tasks = len(tasks)

        average_accuracy = 0
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i >= j:
                    average_accuracy += np.array(metrics['accuracy'])[i,j]
        average_accuracy /= (num_tasks * (num_tasks + 1) / 2)

        print('Avg accuracy: {}'.format(average_accuracy))
        print(pnn_model.networks_maxlogits)

        global_metrics.append(metrics)

    # Save metrics
    np.save('results/exp_cl_metrics/' + str(args.id) + '.npy', global_metrics)

def cl_evaluation(args):
    print('\x1b[1;37;47m' + '* PNN EVALUATION *' + '\x1b[0m')
    print('*'*120)

    # Create folders to save results
    plots_path = 'results/plots/exp_cl_metrics'
    os.makedirs(plots_path, exist_ok=True)

    # Load metrics
    metrics_path = 'results/exp_cl_metrics'
    list_metrics = os.listdir(metrics_path)

    # Global metric variables
    accuracy = []
    ood_accuracy = []
    train_time = []
    test_time = []
    num_params = []
    used_GPU = []

    for metric in list_metrics:
        metrics = np.load(os.path.join(metrics_path, metric), allow_pickle=True)

        for iter in metrics:
            # Get experiments metrics
            accuracy.append(iter['accuracy'])
            ood_accuracy.append(iter['ood_accuracy'])
            train_time.append(iter['train_time'])
            test_time.append(iter['test_time'])
            num_params.append(iter['num_params'])
            used_GPU.append(iter['used_GPU'])

    accuracy = np.array(accuracy)
    ood_accuracy = np.array(ood_accuracy)
    train_time = np.array(train_time)
    test_time = np.array(test_time)
    num_params = np.array(num_params)
    used_GPU = np.array(used_GPU)

    # Get global averages and std
    accuracy_avg = np.mean(accuracy, axis=0)
    accuracy_std = np.std(accuracy, axis=0)
    ood_accuracy_avg = np.mean(ood_accuracy, axis=0)
    ood_accuracy_std = np.std(ood_accuracy, axis=0)
    train_time_avg = np.mean(train_time, axis=0)/1000
    train_time_std = np.std(train_time, axis=0)/1000
    test_time_avg = np.mean(np.mean(test_time, axis=0), axis=0)/1000
    test_time_std = np.mean(np.std(test_time, axis=0), axis=0)/1000
    num_params_avg = np.mean(num_params, axis=0)
    num_params_std = np.std(num_params, axis=0)
    used_GPU_avg = np.mean(used_GPU, axis=0)
    used_GPU_std = np.std(used_GPU, axis=0)

    # Get number of tasks
    num_tasks = len(tasks)

    # Plot accuracy per task
    fig, axs = plot_metric_task(tasks=tasks,
                                metric_avg=accuracy_avg*100,
                                metric_std=accuracy_std*100,)
    fig.savefig(os.path.join(plots_path, 'accuracy.png'), bbox_inches='tight')

    # Plot train time per task
    fig, axs = plot_metric(tasks=tasks,
                           metric_label='Training time (s)',
                           metric_avg=train_time_avg,
                           metric_std=train_time_std)
    fig.savefig(os.path.join(plots_path, 'train_time.png'), bbox_inches='tight')

    # Plot inference time per task
    fig, axs = plot_metric(tasks=tasks,
                           metric_label='Inference time (s)',
                           metric_avg=test_time_avg,
                           metric_std=test_time_std,)
    fig.savefig(os.path.join(plots_path, 'test_time.png'), bbox_inches='tight')

    # Plot number of parameters per task
    fig, axs = plot_metric(tasks=tasks,
                           metric_label='Number of parameters',
                           metric_avg=num_params_avg,
                           metric_std=num_params_std,)
    fig.savefig(os.path.join(plots_path, 'num_parameters.png'), bbox_inches='tight')

     # Plot used GPU per task
    fig, axs = plot_metric(tasks=tasks,
                           metric_label='GPU',
                           metric_avg=used_GPU_avg,
                           metric_std=used_GPU_std,)
    fig.savefig(os.path.join(plots_path, 'used_GPU.png'), bbox_inches='tight')
    
    # Compute other CL metrics:
    # - Average accuracy
    average_accuracy = 0
    for i in range(num_tasks):
        for j in range(num_tasks):
            if i >= j:
                average_accuracy += accuracy_avg[i,j]
    average_accuracy /= (num_tasks * (num_tasks + 1) / 2)

    average_ood_accuracy = 0
    for i in range(num_tasks):
        for j in range(num_tasks):
            if i < j:
                average_ood_accuracy += ood_accuracy_avg[i,j]
    average_ood_accuracy /= (num_tasks * (num_tasks - 1) / 2)

    # - BWT
    bwt = 0
    for i in range(1,num_tasks):
        for j in range(1, i-1):
            bwt += accuracy_avg[i,j] - accuracy_avg[j,j]
    bwt /= (num_tasks * (num_tasks - 1) / 2)
    bwt_pos = np.max([bwt,0])
    rem = 1 - np.abs(min(bwt,0))

    # - FWT
    fwt = 0
    for i in range(num_tasks):
        for j in range(num_tasks):
            if i < j:
                fwt += accuracy_avg[i,j]
    fwt /= (num_tasks * (num_tasks - 1) / 2)

    # - MS
    ms = 0
    for i in range(num_tasks):
        ms += num_params_avg[0]/num_params_avg[i]
    ms = np.min([1,ms/num_tasks])

    # - Size
    # List all available experiments
    list_models = os.listdir('models')
    list_experiments = []
    for model in list_models:
        if 'exp_cl_metrics_' in model:
            list_experiments.append(model)

    # Get size of models for each task and experiment in bytes
    models_size = []
    for exp in list_experiments:
        exp_dir = os.path.join('models', exp)
        models_exp_size = []
        for model in os.listdir(exp_dir):
            if '-1' not in model:
                models_exp_size.append(os.path.getsize(os.path.join(exp_dir, model)))
        models_size.append(models_exp_size)
    models_size = np.array(models_size)
    models_size_avg = np.mean(models_size, axis=0)/1024/1024 # MB
    models_size_std = np.std(models_size, axis=0)/1024/1024 

    # Plot models size per task
    fig, axs = plot_metric(tasks=tasks,
                           metric_avg=models_size_avg,
                           metric_label='Model size (MB)',
                           metric_std=models_size_std,)
    fig.savefig('results/plots/exp_cl_metrics/model_size.png', bbox_inches='tight')

    # Print CL metrics in table
    print('\x1b[1;30;42m' + '* CL metrics *' + '\x1b[0m')
    print('METRIC\t\tVALUE')
    print('Avg acc\t\t{}'.format(np.round(average_accuracy, 4)))
    print('Avg ood acc\t{}'.format(np.round(average_ood_accuracy, 4)))
    print('BWT\t\t{}'.format(np.round(bwt, 4)))
    print('BWT+\t\t{}'.format(np.round(bwt_pos, 4)))
    print('REM\t\t{}'.format(np.round(rem, 4)))
    print('FWT\t\t{}'.format(np.round(fwt, 4)))
    print('MS\t\t{}'.format(np.round(ms, 4)))
    print('Avg used GPU\t{}'.format(np.round(np.mean(used_GPU_avg)/1024/1024, 4)))
    print('# params\t{}'.format(np.round(num_params_avg, 4)))
    print('Model size\t{}'.format(np.round(models_size_avg, 4)))
    print('\nAccuracy matrix:')
    print(np.round(accuracy_avg, 4))
    print('\nOOD accuracy matrix:')
    print(np.round(ood_accuracy_avg, 4))
        
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
        cl_evaluation(args)
    else:
        cl_metrics(args)