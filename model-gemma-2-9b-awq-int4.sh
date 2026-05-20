# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm


# Authorize with hf
if [ "x$HUGGING_FACE_HUB_TOKEN" != "x" ]; then
	hf auth login --token "$HUGGING_FACE_HUB_TOKEN"
fi
	

# Prevent Out of Memory
sync && echo 3 > /proc/sys/vm/drop_caches

ACCT=hugging-quants
MODEL=gemma-2-9b-it-AWQ-INT4

LOCAL_CACHE_DIR=/root/.cache
CONTAINER_CACHE_DIR=/root/.cache
docker run -it --gpus all -p 8001:8000 \
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
	--max-model-len 1024 \
	--gpu-memory-utilization 0.90 \
	--quantization awq_marlin \
	--model $ACCT/$MODEL


	# --max-num-batched-tokens 4096 \
	# nvidia/Llama-3.1-8B-Instruct-FP8

