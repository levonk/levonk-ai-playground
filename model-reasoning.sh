# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm
# https://github.com/AlexsJones/llmfit
#
source .env

# Authorize with hf
if [ "x$HUGGING_FACE_HUB_TOKEN" != "x" ]; then
	if command -v hf ; then
		hf auth login --token "$HUGGING_FACE_HUB_TOKEN"
	fi
fi
	
# Prevent Out of Memory
sudo sync && echo 3 > /proc/sys/vm/drop_caches

ACCT=hesamation
MODEL=Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled

LOCAL_CACHE_DIR=/root/.cache
CONTAINER_CACHE_DIR=/root/.cache
docker stop $MODEL
docker run -it --rm --gpus all -p 8000:8000 \
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
	--max-num-batched-tokens 32768 \
	--quantization awq_merlin \
	--model $ACCT/$MODEL
	
docker ps

