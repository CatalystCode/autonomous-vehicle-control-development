# Distributed Tensorflow Training

The code from this sample is based loosely on training scripts in the [tensorflow models repository](https://github.com/tensorflow/models/tree/master/research/inception). 
In particular, we used the image preprocessing steps from the inception sample, but heavily modified the training and model scripts for a new classifier.

This sample shows how to set up a neural network for distributed training on GPUs using Tensorflow (TF).
In particular, this sample uses between-graph replication where multiple workers send their gradient 
updates to a training chief, who will actually apply them to shared graph. We used the SyncReplicasOptimizer
to synchronize steps between the different worker nodes.

The data and python scripts for this sample are deployed through Docker containers to the nodes of a mixed GPU/CPU cluster
for training. To avoid manually configuring the cluster host addresses for TF distributed, we would recommend running 
training and eval through the provided [helm chart](../Helm/README.md).

## Running on Cluster
See the README under the [helm chart folder](../Helm/README.md) to run the training and eval 
using helm commands.

If you do have a need to run the commands manually, see below for the commands that need to be 
run from each node of the cluster.

To run training:
```bash
# Run on GPU Node 1
# Worker 0 = Chief, by default
python src/python/train.py \
    --job_name="worker" \
    --task_id=0 \
    --ps_hosts=pshost-0_endpoint:9090 \
    --worker_hosts=workerhost-0_endpoint:9090,workerhost-1_endpoint:9090 \
    --train_dir=/train/ \
    --max_steps=10000
    
# Run on GPU Node 2
# Worker 1
python src/python/train.py \
    --job_name="worker" \
    --task_id=1 \
    --ps_hosts=pshost-0_endpoint:9090 \
    --worker_hosts=workerhost-0_endpoint:9090,workerhost-1_endpoint:9090 \
    --train_dir=/train/ \
    --max_steps=10000
    
# Run on CPU Node 1    
# Parameter Server 0
python src/python/train.py \
    --job_name="worker" \
    --task_id=0 \
    --ps_hosts=pshost-0_endpoint:9090 \
    --worker_hosts=workerhost-0_endpoint:9090,workerhost-1_endpoint:9090 \
    --train_dir=/train/ \
    --max_steps=10000
```

To run eval:
```bash
python src/python/eval.py \
    --checkpoint_dir="/train/" \
    --eval_dir="/eval/"
```
## Deploying from Source

The data and scripts for this sample are deployed to the cluster nodes through docker 
images. To make changes to the data or scripts, a new set of images will need to be built and then published 
to a container registry.

Changes can be deployed by setting REPO to your own docker image repository
and then running the following scripts:
```bash
REPO="repo_name"
./build-docker.sh ${REPO}
./publish-docker.sh ${REPO}
```

## Info about training images

The labeled dataset for this sample came from auto-generated images from a Unity project. For more information,
see the code under /tools/ImageGenerator in this repo.
