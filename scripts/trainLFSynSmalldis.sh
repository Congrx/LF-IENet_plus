CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --nproc_per_node 2 --master_port 1123  train.py configs/lf/UrbanLF_Syn_Small_dis/lfienetplusrefine_hr48_480x480_80k_UrbanLF_Syn_Small_dis.py --launcher 'pytorch' 
CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --nproc_per_node 2 --master_port 1123  train.py configs/lf/UrbanLF_Syn_Small_dis/lfienetplusrefine_ref3_r101-d8_480x480_80k_UrbanLF_Syn_Small_dis.py --launcher 'pytorch' 
CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --nproc_per_node 2 --master_port 1123  train.py configs/lf/UrbanLF_Syn_Small_dis/lfienetplusrefine_ref4_r50-d8_480x480_80k_UrbanLF_Syn_Small_dis.py --launcher 'pytorch' 

