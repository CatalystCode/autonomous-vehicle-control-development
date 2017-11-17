# Helm Charts for Distributed Tensorflow

Helm charts are a great tool for managing kubernetes deployments. The expressiveness of the go templates gives us more control to automate configuration of a distributed tensorflow cluster without needing to find and list the host name for each node. There are 3 helm charts included with this sample:

1. distrib-tensorflow - runs distributed training across multiple nodes
2. eval-tensorflow - runs eval on one node
3. tensorboard - runs tensorboard on one node

These charts build on existing work done by Samuel Cozannet for his [Kubernetes + GPU blog series](https://medium.com/intuitionmachine/kubernetes-gpus-tensorflow-8696232862ca) on running distributed Tensorflow on AWS, as well as work done by William Buchwalter (wibuch@microsoft.com), who let us know about the ongoing work to support distributed tensorflow as a kubernetes object (see project [here](https://github.com/tensorflow/k8s)).

The training data and code for distrib-tensorflow is deployed through docker images. We used two sets of docker images for the worker and parameter server (ps) nodes. This is because only workers need to have underlying GPU support (tensorflow-gpu), and only workers need access to the training images. The ps nodes only need the tensorflow model code and access to the shared checkpoint directory. That means the ps docker image is a lighter weight image.

For this example, the default shared storage is Azure File Storage. Each kubernetes pod mounts the same read and write file share, so that they can share common checkpoint files. Azure File Storage is great for setting up quick storage for smaller jobs. If training on large datasets, we'd suggest using a storage solution that caters specifically to big data workloads. 

Other options include provisioning a linux server with high performance SSDs on the same virtual network, and then creating a network file share (NFS) on that server. Then, you can easily mount the nfs drive to all of the kubernetes pods. There are also other alternatives such as mounting a Lustre filesystem or using distributed file systems.

We also mount ephemeral SSD storage (under /mnt/) as part of the kubernetes job. Since the docker image is responsible for pulling down and preprocessing the images, we only need temporary high performance storage during the duration of the job for quick access to images. If the job fails, the pod will be recreated with a fresh set of images. Be careful if using the ephemeral SSD storage for data that needs to be persisted, since it is wiped after each restart of the host machine.

## Prerequisites
- Provision kubernetes cluster with at least 2 GPU enabled nodes and 2 CPU nodes. Follow the guide [here](TO-DO:link to deployment article)
    - The helm chart GPU configuration assumes NVIDIA driver version 384, installed through the ppa method
    - The helm chart GPU configuration also assumes that CUDA has been configured for each GPU machine.
- Install [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/). This is taken care of for you in the Docker file from the provisioning samples.
- Install [helm](https://docs.helm.sh/install/). This is taken care of for you in the Docker file from the provisioning samples.

## Run Sample

1. Initialize Helm

    Make sure that helm is properly configured on the kubernetes cluster. Run the following command:

    ```bash
    helm init
    ```

    if the installation was successful, you should see an output message that goes along the lines of "Happy Tilling!".

2. Set Configuration Values

    The credentials for the azure file share need to be set up prior to installing the helm charts. 

    First, open the values.override.yaml file. Since, all of the charts need access to the same variables, such as azure file share keys, we can provide them in one configuration file and use them to override the values for each chart.

    The azure file share name and key need to be base64 encoded. Run the following in a bash terminal to get the base64 encoded string:

    ```bash
    echo -n "<azure account name>" | base64
    echo -n "<azure account key>" | base64
    ``` 

    Then, put the base64 encoded strings in the values.override.yaml file. We can pass this one configuration file into each of the helm charts, since there are shared variables. Having override files also makes it easy to set up configuration by environment (dev, prod)--just add an override file per environment.

    You can also provide the directory paths for traindir, evaldir, and logdir in this one location, instead of needing to make changes inside each helm chart's values.yaml.

    We are currently mounting the azure file share on each pod at ```/azure/<file share name>```, where the file share name you provide for the storage set up is the file share name used to mount the shared drive.

    ```yaml
    storage:
        accountbase64: <base64 account name>
        keybase64: <base64 account key>
        sharename: <file share name>

    traindir: /azure/<file share name>/training-distrib
    evaldir: /azure/<file share name>/eval
    logdir: /azure/<file share name>/training-distrib
    ```

    Replace the <REPO NAME> in [distrib-tensorflow/values.yaml](distrib-tensorflow/values.yaml) and [eval-tensorflow/values.yaml](eval-tensorflow/values.yaml) for your own base images (see [this guide](../TensorflowModel/README.md) to publish your own Docker images).

    If you are using different scripts, you can make modifications to the entry command in the [distrib-tensorflow/templates/tensorflow-deployment.yaml](distrib-tensorflow/templates/tensorflow-deployment.yaml) and in [eval-tensorflow/templates/tensorflow-deployment.yaml]. If you are running the sample from the Docker images linked above, then no changes are necessary.

3. Run Training

    ```bash
    helm install -f values.override.yaml distrib-tensorflow/
    ```

    Watch the training happen by running:

    ```bash
    kubectl get pods
    ```

    And then to see the logs:

    ```bash
    kubectl logs <pod name> -f
    ```

    After the image preprocessing, the output logs for training on the worker nodes should look like this:

    <img src="/assets/TrainingOutput.png" height="400px"/>

4. Run Eval

    ```bash
    helm install -f values.override.yaml eval-tensorflow/
    ```

    Watch the eval happen by running:

    ```bash
    kubectl get pods
    ```

    And then to see the logs:

    ```bash
    kubectl logs <pod name> -f
    ```

    After image preprocessing, the output logs for running eval should look like this:

    <img src="/assets/EvalOutput.png" height="300px"/>

5. Run Tensorboard

    ```bash
    helm install -f values.override.yaml tensorboard/
    ```

    Check the progress of the deployment by running:
    ```bash
    kubectl get pods
    ```

    Once it is provisioned, we can access the tensorboard instance at ```http://<pod name>:6006```. If you don't want to expose ports of the kubernetes agent to the internet, you can use port-forwarding to get access to the Tensorboard UI.

    ```bash
    kubectl port-forward $(kubectl get pods | grep "tensorboard"| awk '{print $1}') 6006
    ```

    Then from your local machine, go to ```http://localhost:6006``` to see the Tensorboard UI.

    <img src="/assets/Tensorboard.png" height="400px"/>

6. Delete Deployment

    To cancel or delete your helm deployment, it only takes 2 commands. First, run `helm list`, and then add the release name below to delete:

    ```bash
    helm delete <ReleaseName>
    ```