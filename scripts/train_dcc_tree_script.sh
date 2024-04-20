#!/bin/bash
#PBS -l select=1:ncpus=16:mem=50GB
#PBS -l walltime 100:00:00
#PBS -l software=python
#PBS -k oe
#PBS -q allq

source /etc/profile.d/modules.sh
module load python/3.8.3

cd $PBS_O_WORKDIR

source dcc-env/bin/activate

for i in $(seq 1 1 10)
do
#     python ./src/DCC-Tree.py --dataset cgm --datapath ./datasets/cgm/ --out_dir ./results/cgm/ --init_id $i --tag $i --h_init 0.01 --h_final 0.001 --num_chains 16 --num_samps 100 --num_warmup 5000 --T0 100 --T 500 --save 1 --alpha 0.95 --beta 1.0 --M 10  --prop_density 0
#    python ./src/DCC-Tree.py --dataset wu --datapath ./datasets/wu/ --out_dir ./results/wu/ --init_id $i --tag $i --h_init 0.5 --h_final 0.025 --num_chains 16 --num_samps 100 --num_warmup 2000 --T0 100 --T 500 --save 1 --alpha 0.95 --beta 1.0 --M 10  --prop_density 0
#     python ./src/DCC-Tree.py --dataset iris --datapath ./datasets/iris/ --out_dir ./results/iris/ --init_id $i --tag $i --h_init 0.01 --h_final 0.01 --num_chains 16 --num_samps 100 --num_warmup 1000 --T0 100 --T 500 --save 1 --alpha 0.95 --beta 1.0 --M 10  --prop_density 0
#    python ./src/DCC-Tree.py --dataset breast-cancer-wisconsin --datapath ./datasets/breast-cancer-wisconsin/ --out_dir ./results/breast-cancer-wisconsin/ --init_id $i --tag $i --h_init 0.1 --h_final 0.025 --num_chains 16 --num_samps 100 --num_warmup 1000 --T0 100 --T 500 --save 1 --alpha 0.95 --beta 1.0 --M 10  --prop_density 0
#    python ./src/DCC-Tree.py --dataset wine --datapath ./datasets/wine/ --out_dir ./results/wine/ --init_id $i --tag $i --h_init 0.025 --h_final 0.025 --num_chains 16 --num_samps 100 --num_warmup 1000 --T0 100 --T 500 --save 1 --alpha 0.95 --beta 1.0 --M 10  --prop_density 0
#    python ./src/DCC-Tree.py --dataset raisin --datapath ./datasets/raisin/ --out_dir ./results/raisin/ --init_id $i --tag $i --h_init 0.05 --h_final 0.001 --num_chains 16 --num_samps 100 --num_warmup 1000 --T0 100 --T 500 --save 1 --alpha 0.95 --beta 1.0 --M 10  --prop_density 0
done

exit 0
