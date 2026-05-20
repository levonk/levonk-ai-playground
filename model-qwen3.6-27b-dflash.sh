# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm
#
source ~/.local/bin/.env

# Authorize with hf
if [ "x$HUGGING_FACE_HUB_TOKEN" != "x" ]; then
	if command -v hf ; then
		hf auth login --token "$HUGGING_FACE_HUB_TOKEN"
	fi
fi
	
# Prevent Out of Memory
sync && echo 3 > /proc/sys/vm/drop_caches

ACCT=z-lab
MODEL=Qwen3.6-27B-DFlash

LOCAL_CACHE_DIR=/root/.cache
CONTAINER_CACHE_DIR=/root/.cache
docker run -it --gpus all -p 8000:8000 \
	--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
	-v $LOCAL_CACHE_DIR/huggingface:$CONTAINER_CACHE_DIR/huggingface \
	-v $LOCAL_CACHE_DIR/torch:$CONTAINER_CACHE_DIR/torch \
	-v $LOCAL_CACHE_DIR/torch_extensions:$CONTAINER_CACHE_DIR/torch_extensions \
	-v $LOCAL_CACHE_DIR/vllm:$CONTAINER_CACHE_DIR/vllm \
	-v $LOCAL_CACHE_DIR/flashinfer:$CONTAINER_CACHE_DIR/flashinfer \
	-e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-hf_YOUR_TOKEN} \
	--name $MODEL \
       	nvcr.io/nvidia/vllm:26.04-py3 \
	python3 -m vllm.entrypoints.openai.api_server \
	--trust-remote-code \
	--tensor-parallel-size 1 \
	--gpu-memory-utilization 0.90 \
	--attention-backend flash_attn \
	--max-num-batched-tokens 32768 \
	--speculative-config '{"method": "dflash", "model": "$ACCT/$MODEL", "num_speculative_tokens": 15}' \
	--model $ACCT/$MODEL
	
#--max-model-len 1024 \
# --quantization awq \
# google/gemma-4-31B-it

