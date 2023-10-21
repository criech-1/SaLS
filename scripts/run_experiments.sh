# Run experiments
for i in 0 5 25 32 67
do
    python scripts/experiments/cl_metrics.py --id $i --dataset_path /u/52/echevec1/unix/toro/TORO_dataset/masks_reorganized --iter 1
done

for i in 25 32 67
do
    python scripts/experiments/cl_robustness.py --id $i --num_tasks 199
done

for i in 0 5 25 32 67
do
    python scripts/experiments/transform_metrics.py --id $i --dataset_path /u/52/echevec1/unix/toro/TORO_dataset/masks_reorganized --iter 1
done

for i in 0 5 25 32 67
do
    python scripts/experiments/ood_metrics.py --id $i --dataset_path /u/52/echevec1/unix/toro/TORO_dataset/masks_reorganized --iter 1
done

# Run experiment with two options "DINOv2_singlelayer" and "Resnet50"
for i in 0 5 25 32 67
do
    for j in "DINOv2_multilayer" "ResNet50"
        do
            python scripts/experiments/backbone_metrics.py --id $i --backbone $j --dataset_path /u/52/echevec1/unix/toro/TORO_dataset/masks_reorganized --iter 1
    done
done