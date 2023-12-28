CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --nproc_per_node 2 --master_port 1123  train.py configs/lf/UrbanLF_Real/lfienetplusrefine_hr48_432x432_80k_UrbanLF_Real.py --launcher 'pytorch' 
CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --nproc_per_node 2 --master_port 1123  train.py configs/lf/UrbanLF_Real/lfienetplusrefine_ref4_r50-d8_432x432_80k_UrbanLF_Real.py --launcher 'pytorch' 

