empty_gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F ', ' '{if ($2 < 512) print $1}' | head -n 1)
if [ -z "$empty_gpu" ]; then
    echo "No empty GPU available"
    exit 1
fi
CUDA_VISIBLE_DEVICES=$empty_gpu python3 -m torch.distributed.run \
    --nnodes 1 \
    --master_addr $(hostname) \
    --master_port 12345 \
    --node_rank 0 \
    --nproc_per_node 1 \
    infer.py