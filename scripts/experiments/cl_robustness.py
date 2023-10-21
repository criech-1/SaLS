"""Script to evaluate PNN robustness."""

import argparse
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from copy import deepcopy
from torch.utils.data import DataLoader

from src.SaLS.pnn import ProgressiveNeuralNetwork, ClassifierNet
from src.SaLS.utils import fit_pnn, seed_all_rng
from tools.inference import predict_dataloader
from tools.metrics import plot_metricx_metricy
from tools.dataloaders import ImageDataset

def cl_robustness_metrics(args):
    print('\x1b[1;37;47m' + '* PNN ROBUSTNESS METRICS *' + '\x1b[0m')
    print('*'*120)

    # Create folders to save results
    os.makedirs('results/exp_cl_robustness', exist_ok=True)

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

    pnn_model = ProgressiveNeuralNetwork(base_network = base_network,
                                         backbone = dino_model,
                                         last_layer_name = last_layer_name,
                                         lateral_connections = deepcopy(lateral_connections))

    print('\x1b[1;30;42m' + '>> PNN model loaded' + '\x1b[0m')
    print('*'*120)

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

    # Initialize global metrics
    metrics = {'accuracy': [],
               'train_time': [],
               'test_time': [],
               'num_params': [],
               'model_size': [],
               'used_GPU': []}
    
    tasks = []
    for i in range(args.num_tasks):
        # Artificial image to train PNN
        art_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # Dataset from artificial image
        dataset_train = ImageDataset(img=[art_img], 
                                     category=[str(i)],
                                     transform=data_transforms['train'])
        # Dataloader from artificial image
        tasks.append(DataLoader(dataset=dataset_train,
                                batch_size=1,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True))
        
    # For every task
    for i, task_i in enumerate(tasks):
        accuracy = []
        test_time = []

        # Measure inference time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # Train PNN
        fit_pnn(model = pnn_model, 
                dataloader_train = task_i,
                dataloader_val = task_i,
                parameters = parameters,
                device = device,
                name = 'exp_cl_robustness_' + str(args.id),)
        
        end.record()
        torch.cuda.synchronize()
        metrics['train_time'].append(start.elapsed_time(end))

         # Get GPU utilization
        utilization_GPU = torch.cuda.mem_get_info()
        used_GPU = utilization_GPU[1] - utilization_GPU[0]
        metrics['used_GPU'].append(used_GPU)

        # Get saved model size in B
        model_size = os.path.getsize('models/exp_cl_robustness_' + str(args.id) + '/full_state_dict_{}.pt'.format(i+1))
        metrics['model_size'].append(model_size)

        # Remove all saved models
        os.system('rm -rf models/exp_cl_robustness_' + str(args.id) + '/*')
        
        # Get number of parameters of model
        num_params = sum(p.numel() for p in pnn_model.parameters())
        metrics['num_params'].append(num_params)

        # Test with every task
        for j, task_j in enumerate(tasks):
            # Measure inference time
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            pred_labels = predict_dataloader(model = pnn_model,
                                             dataloader = task_j,
                                             device = device)
            # Compute accuracy
            task_j_label = task_j.dataset.classes[0]
            accuracy_j = np.sum(np.array(pred_labels) == task_j_label) / len(pred_labels)
            # print('>> Accuracy task {}: {}'.format(j, accuracy_j))

            end.record()
            torch.cuda.synchronize()
            test_time_j = start.elapsed_time(end)

            # Save metrics
            accuracy.append(accuracy_j)
            test_time.append(test_time_j)

        metrics['accuracy'].append(accuracy)
        metrics['test_time'].append(test_time)

    # Save metrics
    np.save('results/exp_cl_robustness/' + str(args.id), metrics)

def cl_robustness_evaluation(args):
    print('\x1b[1;37;47m' + '* PNN ROBUSTNESS EVALUATION *' + '\x1b[0m')
    print('*'*120)

    # Create folders to save results
    plots_path = 'results/plots/exp_cl_robustness'
    os.makedirs(plots_path, exist_ok=True)

    # Load metrics
    list_metrics = os.listdir('results/exp_cl_robustness')

    # Task is str list from 1 to num_tasks+1
    tasks = ['{}'.format(i) for i in range(1, args.num_tasks + 1)]
    num_tasks = len(tasks)
    tasks_array = np.array(range(num_tasks))

    # Global metric variables
    accuracy = []
    train_time = []
    test_time = []
    num_params = []
    model_size = []
    used_GPU = []

    for metric in list_metrics:
        metrics = np.load('results/exp_cl_robustness/' + metric, allow_pickle=True).item()

        # Get experiments metrics
        accuracy.append(metrics['accuracy'])
        train_time.append(metrics['train_time'])
        test_time.append(metrics['test_time'])
        num_params.append(metrics['num_params'])
        model_size.append(metrics['model_size'])
        used_GPU.append(metrics['used_GPU'])

    # Average accuracy
    avg_acc_task = []
    std_acc_task = []

    for i in range(len(accuracy)):
        avg_acc_task.append(np.mean(accuracy[i],axis=0))
        std_acc_task.append(np.std(accuracy[i],axis=0))

    # Average test time
    avg_test_time_task = []
    std_test_time_task = []

    for i in range(len(test_time)):
        avg_test_time_task.append(np.mean(test_time[i],axis=0))
        std_test_time_task.append(np.std(test_time[i],axis=0))
    
    # Convert to numpy array
    avg_acc_task = np.mean(avg_acc_task, axis=0)*100
    std_acc_task = np.std(std_acc_task, axis=0)*100
    avg_test_time_task = np.mean(avg_test_time_task, axis=0)
    std_test_time_task = np.std(std_test_time_task, axis=0)
    avg_train_time = np.mean(train_time, axis=0)/1000
    std_train_time = np.std(train_time, axis=0)/1000
    avg_num_params = np.mean(num_params, axis=0)
    std_num_params = np.std(num_params, axis=0)
    avg_model_size = np.mean(model_size, axis=0)/1024/1024
    std_model_size = np.std(model_size, axis=0)/1024/1024
    avg_used_GPU = np.mean(used_GPU, axis=0)/1024/1024/1024
    std_used_GPU = np.std(used_GPU, axis=0)/1024/1024/1024

    # Plot accuracy   
    fig, ax = plot_metricx_metricy(x_metric=tasks_array,
                                   x_label='Tasks',
                                   y_metric=avg_acc_task,
                                   y_label='Average accuracy (%)',
                                   y_std=std_acc_task,)
    fig.savefig(os.path.join(plots_path, 'robustness_accuracy.png'), bbox_inches='tight')

    # Plot training time
    fig, ax = plot_metricx_metricy(x_metric=tasks_array,
                                   x_label='Tasks',
                                   y_metric=avg_train_time,
                                   y_label='Training time (s)',
                                   y_std=std_train_time,)
    ax.set_ylim([0, 100])
    # ax.set_yscale('log')
    fig.savefig(os.path.join(plots_path, 'robustness_train_time.png'), bbox_inches='tight')

    # Plot test time
    fig, ax = plot_metricx_metricy(x_metric=tasks_array,
                                   x_label='Tasks',
                                   y_metric=avg_test_time_task,
                                   y_label='Inference time (ms)',
                                   y_std=std_test_time_task,)
    ax.set_ylim([0, 2*np.max(avg_test_time_task)])
    ax.set_xlim([0, num_tasks])
    fig.savefig(os.path.join(plots_path, 'robustness_test_time.png'), bbox_inches='tight')

    # Plot number of parameters
    fig, ax = plot_metricx_metricy(x_metric=tasks_array,
                                   x_label='Tasks',
                                   y_metric=avg_num_params,
                                   y_label='Number of parameters',
                                   y_std=std_num_params,)
    fig.savefig(os.path.join(plots_path, 'robustness_num_params.png'), bbox_inches='tight')

    # Plot model size
    fig, ax = plot_metricx_metricy(x_metric=tasks_array,
                                   x_label='Tasks',
                                   y_metric=avg_model_size,
                                   y_label='Model size (MB)',
                                   y_std=std_model_size,)
    fig.savefig(os.path.join(plots_path, 'robustness_model_size.png'), bbox_inches='tight')

    # Plot GPU usage
    fig, ax = plot_metricx_metricy(x_metric=tasks_array,
                                   x_label='Tasks',
                                   y_metric=avg_used_GPU,
                                   y_label='GPU utilisation (GB)',
                                   y_std=std_used_GPU,)
    fig.savefig(os.path.join(plots_path, 'robustness_used_GPU.png'), bbox_inches='tight')

    print(avg_used_GPU)
    print(avg_model_size)

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(prog='Continual Semantic Segmentation',
                                     description='Description of the program',
                                     epilog='End of the program description')
    parser.add_argument('--id', type=int, required=True, help='experiment id and seed')
    parser.add_argument('--num_tasks', type=int, required=True, help='number of tasks to perform')
    parser.add_argument('--eval', action='store_true', help='evaluate PNN with existing metrics')
    parser.add_argument('--no-eval', dest='eval', action='store_false', help='get PNN metrics')
    parser.set_defaults(eval=False)
    args = parser.parse_args()

    # Set manual seed
    seed_all_rng(args.id)

    if args.eval:
        cl_robustness_evaluation(args)
    else:
        cl_robustness_metrics(args)
