FROM pytorch/torchserve:latest-gpu

RUN python -m pip install --no-cache-dir transformers==4.28.1
RUN apt-get update && apt-get install -y cuda-toolkit-11-7
RUN python -m pip install --no-cache-dir deepspeed==0.9.5

RUN mkdir /home/model-server/.cache
ENV TRANSFORMERS_CACHE=/home/model-server/.cache