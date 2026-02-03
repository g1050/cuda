# docker run -it \
#     --restart=always \
#     --runtime nvidia --gpus all \
#     --name xkgao_vllm \
#     -v /data:/data \
#     -p 10827-10837:10827-10837 \
#     --ipc=host \
#     --entrypoint bash \
#     vllm/vllm-openai:v0.10.1.1


docker run -it \
    --restart=always \
    --runtime nvidia --gpus all \
    --name xkgao \
    -v /data:/data \
    -v /data0:/data0 \
    --ipc=host \
    --entrypoint bash \
    pytorch_dev:latest

