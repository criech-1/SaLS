"""Script to evaluate different backbones for classification."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision.transforms as transforms

import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy

from src.SaLS.pnn import ProgressiveNeuralNetwork
from src.SaLS.utils import fit_pnn, seed_all_rng
from tools.dataloaders import get_split_TOROdataset
from tools.inference import predict_dataloader
from tools.metrics import plot_metric

# Tasks to evaluate
tasks = ['dark_mat', 'lab_floor', 'stones', 'ramp', 'blue_mattress', 'pedestal']
num_tasks = len(tasks)

class MultiLayer(nn.Module): # input size: 384, output size: 1 -> self.fc = nn.Linear(384, 1)
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input = nn.Linear(in_features=input_size, out_features=288)
        self.hidden_1 = nn.Linear(in_features=288, out_features=192)
        self.hidden_2 = nn.Linear(in_features=192, out_features=144)
        self.hidden_3 = nn.Linear(in_features=144, out_features=96)
        self.hidden_4 = nn.Linear(in_features=96, out_features=64)
        self.hidden_5 = nn.Linear(in_features=64, out_features=32)
        self.hidden_6 = nn.Linear(in_features=32, out_features=16)
        self.hidden_7 = nn.Linear(in_features=16, out_features=8)
        self.fc = nn.Linear(in_features=8, out_features=output_size)
        
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = F.relu(self.hidden_4(x))
        x = F.relu(self.hidden_5(x))
        x = F.relu(self.hidden_6(x))
        x = F.relu(self.hidden_7(x))
        return self.fc(x)
    
class SingleLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(in_features=input_size, out_features=output_size)
        
    def forward(self, x):
        return self.fc(x) 

def backbone_metrics(args):
    print('\x1b[1;37;47m' + '* BACKBONE METRICS *' + '\x1b[0m')
    print('*'*120)

    # Create folders to save results
    os.makedirs('results/exp_backbone_metrics', exist_ok=True)

    # Use cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # Num outputs of NN
    num_classes = 1 #len(tasks)

    # Load backbone and/or classifier and train
    if args.backbone == 'DINOv2_singlelayer' or args.backbone == 'DINOv2_multilayer':
        # Load DINOv2 model from torch hub or local path if net not available
        try:
            backbone_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
            print('\x1b[1;30;42m' + '>> DINOv2 model loaded from hub' + '\x1b[0m')
        except: 
            backbone_model = torch.hub.load('dinov2', 'dinov2_vits14', source='local', pretrained=False)
            backbone_model.load_state_dict(torch.load('./Segment-and-Track-Anything/ckpt/dinov2_vits14_pretrain.pth'))
            backbone_model.to(device).eval()
            print('\x1b[1;30;42m' + '>> DINOv2 model loaded locally' + '\x1b[0m')

        if args.backbone == 'DINOv2_singlelayer':
            # Load_simple network
            network = SingleLayer(input_size=2,
                                  output_size=num_classes) # 384 if DINOv2 with ViT-Small
        else:
            # Load complex network
            network = MultiLayer(input_size = list(backbone_model.children())[-2].normalized_shape[0], # 384 if DINOv2 with ViT-Small
                                 output_size=num_classes) 

    elif args.backbone == 'ResNet50':
        # Load ResNet50 model from torch hub or local path if net not available
        try:
            network = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
            network.eval().to(device)
            print('\x1b[1;30;42m' + '>> ResNet50 model loaded from hub' + '\x1b[0m')
        except:
            network = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=False)
            network.load_state_dict(torch.load('./Segment-and-Track-Anything/ckpt/nvidia_resnet50.pth'))
            network.eval().to(device)
            print('\x1b[1;30;42m' + '>> ResNet50 model loaded locally' + '\x1b[0m')

        # Freeze layers and modify last one
        for param in network.parameters():
            param.requires_grad=False
        
        backbone_model = None
        num_ftrs = network.fc.in_features
        network.fc = nn.Linear(num_ftrs, num_classes).to(device)

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

    # Global metrics
    global_metrics = []

    for iter in range(args.iter):
        # Create PNN model for each iteration
        pnn_model = ProgressiveNeuralNetwork(base_network = network,
                                             backbone = backbone_model,
                                             last_layer_name = last_layer_name,
                                             lateral_connections = deepcopy(lateral_connections)).to(device)

        # Choose random sequences for training and testing
        train_dataloaders, val_dataloaders, test_dataloaders = get_split_TOROdataset(dataset_path=args.dataset_path,
                                                                                     tasks=tasks,
                                                                                     train_ratio=0.7,
                                                                                     data_transforms=data_transforms,
                                                                                     batch_size=1,
                                                                                     num_workers=0,
                                                                                     max_data_seq=20,
                                                                                     randomize=True,)
            # Initialize globel metrics
        metrics = {'accuracy': [], 
                   'train_time': [],
                   'used_GPU': [], 
                   'num_params': [],
                   'test_time': [],
                   'model_size': []}

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
                    name = 'exp_backbone_metrics_' + str(args.id),)
            
            end.record()
            torch.cuda.synchronize()
            train_time = start.elapsed_time(end)

            # Get GPU utilization
            utilization_GPU = torch.cuda.mem_get_info()
            used_GPU = utilization_GPU[1] - utilization_GPU[0]
            metrics['used_GPU'].append(used_GPU)

             # Get saved model size in B
            model_size = os.path.getsize('models/exp_backbone_metrics_' + str(args.id) + '/full_state_dict_{}.pt'.format(i+1))
            metrics['model_size'].append(model_size)

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

        global_metrics.append(metrics)

    # Save metrics
    np.save('results/exp_backbone_metrics/' + args.backbone + '_' + str(args.id) + '.npy', global_metrics)

def backbone_evaluation():
    # Load all metrics
    print('\x1b[1;37;47m' + '* BACKBONE EVALUATION *' + '\x1b[0m')
    print('*'*120)

    # Create folders to save results
    plots_path = 'results/plots/exp_backbone_metrics'
    os.makedirs(plots_path, exist_ok=True)

    list_metrics = os.listdir('results/exp_backbone_metrics')

    # Global metric variables
    backbones = []
    accuracy = []
    train_time = []
    test_time = []
    used_GPU = []
    num_params = []
    model_size = []

    for metric in list_metrics:
        metrics = np.load('results/exp_backbone_metrics/' + metric, allow_pickle=True)
        backbone = '_'.join(metric.split('_')[0:-1])
        
        for iter in metrics:
            # Get experiments metrics
            backbones.append(backbone)
            accuracy.append(iter['accuracy'])
            train_time.append(iter['train_time'])
            test_time.append(iter['test_time'])
            used_GPU.append(iter['used_GPU'])
            num_params.append(iter['num_params'])
            model_size.append(iter['model_size'])

    backbones = np.array(backbones)
    accuracy = np.array(accuracy)
    train_time = np.array(train_time)
    test_time = np.array(test_time)
    used_GPU = np.array(used_GPU)
    num_params = np.array(num_params)
    model_size = np.array(model_size)

    unique_backbones = np.unique(backbones)

    avg_acc_backbone = []
    std_acc_backbone = []
    avg_train_time_backbone = []
    std_train_time_backbone = []
    avg_test_time_backbone = []
    std_test_time_backbone = []
    avg_used_GPU_backbone = []
    std_used_GPU_backbone = []
    avg_num_params_backbone = []
    std_num_params_backbone = []
    avg_model_size_backbone = []
    std_model_size_backbone = []

    for backbone in unique_backbones:
        # Get indexes of experiments with same backbone
        idx = np.where(backbones == backbone)[0]

        # Get metrics
        accuracy_backbone = accuracy[idx]
        train_time_backbone = train_time[idx]
        test_time_backbone = test_time[idx]
        used_GPU_backbone = used_GPU[idx]
        num_params_backbone = num_params[idx]
        model_size_backbone = model_size[idx]

        # Get global averages and std 
        avg_train_time_backbone.append(np.mean(train_time_backbone,axis=0))
        std_train_time_backbone.append(np.std(train_time_backbone,axis=0))
        avg_used_GPU_backbone.append(np.mean(used_GPU_backbone,axis=0))
        std_used_GPU_backbone.append(np.std(used_GPU_backbone,axis=0))
        avg_num_params_backbone.append(np.mean(num_params_backbone,axis=0))
        std_num_params_backbone.append(np.std(num_params_backbone,axis=0))
        avg_model_size_backbone.append(np.mean(model_size_backbone,axis=0))
        std_model_size_backbone.append(np.std(model_size_backbone,axis=0))


        avg_acc_backbone.append(np.mean(accuracy_backbone,axis=0))
        std_acc_backbone.append(np.std(accuracy_backbone,axis=0))
        avg_test_time_backbone.append(np.mean(test_time_backbone,axis=0))
        std_test_time_backbone.append(np.std(test_time_backbone,axis=0))
        
    # Average accuracy
    avg_acc_task = []
    std_acc_task = []

    for i in range(len(avg_acc_backbone)):
        task_acc_avg = []
        task_acc_std = []
        for j in range(len(avg_acc_backbone[i])):
            task_acc_avg.append(np.mean(np.array(avg_acc_backbone[i])[j:,j]))
            task_acc_std.append(np.mean(np.array(std_acc_backbone[i])[j:,j]))
        avg_acc_task.append(task_acc_avg)
        std_acc_task.append(task_acc_std)

    # Average test time
    avg_test_time_task = []
    std_test_time_task = []

    for i in range(len(avg_test_time_backbone)):
        avg_test_time_task.append(np.mean(avg_test_time_backbone[i],axis=0))
        std_test_time_task.append(np.std(avg_test_time_backbone[i],axis=0))
    
    # Convert to numpy array
    avg_acc_task = np.array(avg_acc_task)*100
    std_acc_task = np.array(std_acc_task)*100
    avg_train_time_backbone = np.array(avg_train_time_backbone)/1000
    std_train_time_backbone = np.array(std_train_time_backbone)/1000
    avg_test_time_task = np.array(avg_test_time_task)/1000
    std_test_time_task = np.array(std_test_time_task)/1000
    avg_used_GPU_backbone = np.array(avg_used_GPU_backbone)/1024/1024/1024
    std_used_GPU_backbone = np.array(std_used_GPU_backbone)/1024/1024/1024
    avg_num_params_backbone = np.array(avg_num_params_backbone)
    std_num_params_backbone = np.array(std_num_params_backbone)
    avg_model_size_backbone = np.array(avg_model_size_backbone)/1024/1024
    std_model_size_backbone = np.array(std_model_size_backbone)/1024/1024

    # Plot accuracy
    fig, ax = plot_metric(tasks=tasks,
                          metric_avg=avg_acc_task,
                          metric_label='Average accuracy (%)',
                          metric_std=std_acc_task)
    ax.set_ylim(0, 100)
    fig.legend(unique_backbones,framealpha=1)
    fig.savefig(os.path.join(plots_path, 'backbone_accuracy.png'), bbox_inches='tight')

    # Plot training time
    fig, ax = plot_metric(tasks=tasks,
                          metric_avg=avg_train_time_backbone,
                          metric_label='Training time (s)',
                          metric_std=std_train_time_backbone)
    # fig.legend(unique_backbones)
    fig.savefig(os.path.join(plots_path, 'backbone_train_time.png'), bbox_inches='tight')

    # Plot test time
    fig, ax = plot_metric(tasks=tasks,
                          metric_avg=avg_test_time_task,
                          metric_label='Inference time (s)',
                          metric_std=std_test_time_task)
    # fig.legend(unique_backbones)
    fig.savefig(os.path.join(plots_path, 'backbone_test_time.png'), bbox_inches='tight')

    # Plot used GPU
    fig, ax = plot_metric(tasks=tasks,
                          metric_avg=avg_used_GPU_backbone,
                          metric_label='GPU utilisation (GB)',
                          metric_std=std_used_GPU_backbone)
    # fig.legend(unique_backbones)
    fig.savefig(os.path.join(plots_path, 'backbone_used_GPU.png'), bbox_inches='tight')

    # Plot number of parameters
    fig, ax = plot_metric(tasks=tasks,
                          metric_avg=avg_num_params_backbone,
                          metric_label='Number of parameters',
                          metric_std=std_num_params_backbone)
    fig.savefig(os.path.join(plots_path, 'backbone_num_params.png'), bbox_inches='tight')

    # Plot model size
    fig, ax = plot_metric(tasks=tasks,
                          metric_avg=avg_model_size_backbone,
                          metric_label='Model size (MB)',
                          metric_std=std_model_size_backbone)
    # fig.legend(unique_backbones)
    fig.savefig(os.path.join(plots_path, 'backbone_model_size.png'), bbox_inches='tight')

    print('Average accuracy: ')
    print(np.mean(avg_acc_task,axis=1))

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(prog='Continual Semantic Segmentation',
                                     description='Description of the program',
                                     epilog='End of the program description')
    parser.add_argument('--id', type=int, required=True, help='experiment id and seed')
    parser.add_argument('--backbone', 
                        type=str, 
                        required=True, 
                        default='DINOv2_singlelayer', 
                        choices={'DINOv2_singlelayer', 'DINOv2_multilayer', 'ResNet50'}, 
                        help='experiment backbone')
    parser.add_argument('--dataset_path', type=str, required=True, help='dataset directory')
    parser.add_argument('--iter', type=int, default=2, required=False, help='number of iterations')

    parser.add_argument('--eval', action='store_true', help='evaluate with existing metrics')
    parser.add_argument('--no-eval', dest='eval', action='store_false', help='get backbone metrics')
    parser.set_defaults(eval=False)
    args = parser.parse_args()

    # Set manual seed
    seed_all_rng(args.id)

    if args.eval:
        backbone_evaluation()
    else:
        backbone_metrics(args)