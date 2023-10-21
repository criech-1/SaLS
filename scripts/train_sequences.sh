# Run experiments
for i in 0 5 25 32 67
do
    for j in 1 2 3 4 5 6 7
    do
        python scripts/experiments/sequences_metrics.py --id $i --seq $j --train_dir /u/52/echevec1/unix/toro/TORO_dataset/categories --cl
        pkill -9 python
    done
done