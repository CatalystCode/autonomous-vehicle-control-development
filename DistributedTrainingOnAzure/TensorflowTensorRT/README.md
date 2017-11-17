# Tensorflow to TensorRT
    
## Introduction to TensorRT

Fast predictions are crucial in many real-world applications. Self-driving cars, for example, need to constantly evaluate and react to a quickly changing environment. A fast prediction can mean the difference between moving away to safety or crashing into another vehicle. NVIDIA TensorRT is a runtime engine that gives the ability to optimize prediction tasks on both embedded devices (such as in a self-driving car) and in the cloud.

Specifically, TensorRT speeds up the prediction time of neural networks. It does this by optimizing the layers of 
the neural net for execution across NVIDIA GPUs. The end result is performance that is multitudes faster than on CPU alternatives.

## Introduction to Tensorflow to TensorRT Conversion

Tensorflow is a popular framework for implementing neural networks. However, despite their similar names, TensorRT and Tensorflow have shared little common ground other than their base data structure (a Tensor). In past releases, TensorRT has only supported models implemented in the Caffe framework. Models implemented in other frameworks, such as Tensorflow, required the developer to build out the neural net layers using the TensorRT C++ API. This was prone to error unless the developer understood how each type of layer could be converted to its respective TensorRT representation. It also meant that developers had to maintain two versions of the same model. Finally, it posed a high barrier of entry for developers not familiar with the C++ language.

The latest version of TensorRT gives out-of-the-box support for converting Tensorflow models to TensorRT, as well as access to a python library for running TensorRT. The typical workflow for converting tensorflow models to TensorRT includes the following steps:

1. Freeze Tensorflow model
2. Convert frozen Tensorflow model to UFF
3. Create TensorRT engine
4. Serialize TensorRT engine to PLAN file
5. Load TensorRT engine and run inference

This walkthrough includes reusable python scripts to implement the steps of the Tensorflow to TensorRT workflow.

## Prerequisites

- Set up linux machine with NVIDIA Driver (one option is to provision an N-Series VM on Azure, then follow \
the instructions [here](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup))

- Install [CUDA Toolkit](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) on host machine. 

- Install [Tensorflow](https://www.tensorflow.org/install/install_linux) on host machine.

- Create an NVIDIA [developer account](https://developer.nvidia.com/developer-program).

- Install [TensorRT 3 release candidate](https://developer.nvidia.com/nvidia-tensorrt3rc-download) \
for appropriate GPU (currently supports Tesla GPUS and Jetson platforms)

    - Follow the installation instructions in the TensorRT 3 RC Installation Guide (download available at the link above).
    - Inside the .tar package for TensorRT 3 RC, there should be a User Guide under /docs/TensorRT-3-User-Guide.pdf.
    
## Run Sample Scripts

For this sample, we created a neural network that is a binary classifier for Car/NoCar images. The model was trained in Tensorflow over a distributed GPU-enabled cluster. The model takes an input called 'images\_placeholder' with dimensions 3 channel x 244 x 244 and results in an output called 'softmax\_tensor'. The model was frozen as a .pb file and provided under
model/frozen_model.pb. 

To use your own model, make a frozen model file from your checkpoint folder using the freeze.py script:

```bash
python freeze.py --checkpoint_folder=/path/to/checkpoint --output_node_names=names_of_output_tensors
```

The first step is to convert the model to a TensorRT engine and serialize it to disk. Use the script below if running with
the sample model. To use your own model, update the input\_placeholder name and its dimensions as well as the output\_placeholders. 

```bash
python deploy-to-tensorrt.py \
    --frozen_file='model/frozen_model.pb' \
    --input_placeholder='images_placeholder' \
    --dimensions='3,244,244' \
    --output_placeholders='softmax_tensor' \
    --file_path='model/model.PLAN' \
    --max_batch_size=1
```

Running the above script should result in a model/model.PLAN file. 
Then, run inference using the serialized PLAN file.

```bash
python infer.py \
    --plan_file='model/model.PLAN'
    --image_path='images/Car.png'
    --desired_size='3,224,224'
```

The printed result is an array of probabilities for each class. We then run argmax on the result to get the 
classification result.

## Breakdown of Workflow Steps

All that is needed to run the samples is provided in the steps above. If you'd like a walkthrough of the steps that we took in the code, see below for a more detailed explanation.

### Step 1: Freeze Tensorflow model

First, we converted a Tensorflow checkpoint directory to a frozen model (typically a .pb file). A frozen
Tensorflow model removes unnecessary nodes from your graph that were only needed for training. It also converts
the weights to constants. For the sample scripts, we provided an already frozen file under the model folder.

We implemented a simple freeze.py script to freeze a Tensorflow model from the checkpoint files. 
Running this script will output a frozen_model.pb file in the given checkpoint directory.

```bash
python freeze.py --checkpoint_folder=/path/to/checkpoint --output_node_names=names_of_output_tensors
```

*Tensorflow also provides its own freeze_graph.py script, which can also be used. The freeze_graph script needs to
first be built with bazel. For simplicity, we wanted a self-contained freeze script.*

If using your own tensorflow model, there a few limitations to keep in mind. TensorRT currently works best with models
trained in NCHW format, where channel comes before height and width in the image feed shape. Also, only certain layer 
types are supported for conversion to TensorRT (the list below was provided in the TensorRT 3 User Guide--see the prerequisites section for where to find the user guide):

**Supported TensorFlow Operations**
The current exporter supports the following native layers of TensorFlow:
- Placeholder is converted into an UFF Input layer.
- Const is converted into a UFF Const layer.
- Add, Sub, Mul, Div, Minimum and Maximum are converted into a UFF Binary layer.
- BiasAdd is converted into a UFF Binary layer.
- Negative, Abs, Sqrt, Rsqrt, Pow, Exp and Log are converted into a UFF Unary layer.
- FusedBatchNorm is converted into a UFF Batchnorm layer.
- Tanh, Relu and Sigmoid are converted into a UFF Activation layer.
- Softmax is converted into a UFF Softmax layer.
- Mean is converted into a UFF Reduce layer.
- ConcatV2 is converted into a UFF Concat layer.
- Reshape is converted into a UFF Reshape layer.
- Transpose is converted into a UFF Transpose layer.
- Conv2D and DepthwiseConv2dNative are converted into a UFF Conv layer.
- ConvTranspose2D are converted into a UFF ConvTranspose layer.
- MaxPool and AvgPool are converted into a UFF Pooling layer.
- Pad is supported if followed by one of these TensorFlow layers: Conv2D, DepthwiseConv2dNative, MaxPool, and AvgPool.

### Step 2: Convert frozen Tensorflow model to UFF

Next, we converted the .pb file to a Universal File Format (UFF) representation of the model. TensorRT 3 RC comes with a 
uff python library that converts a frozen Tensorflow model to UFF. We chose to implement this step using the stream
approach in the script deploy-to-tensorrt.py. Since this code is embedded in deploy-to-tensorrt.py, there is no need
to run this code as a separate step.

```python
import uff

uff_model = uff.from_tensorflow_frozen_model(frozen_file='model/frozen_model.py',
                                                 input_nodes=['images_placeholder'],
                                                 output_nodes=['softmax_tensor'])
                                                 
# Pass uff_model to TensorRT parser...
```

### Step 3: Create TensorRT engine

The TensorRT runtime engine is created by passing a parser and registering the input node of the model. Since this code
is embedded in deploy_to_tensorrt.py, there is no need to run this code as a separate step.

```python
import tensorrt.parsers.uffparser as uffparser
import tensorrt as trt

# Create parser for uff file and register input placeholder
# Dimensions for the input placeholder must be in CHW (Channel, Height, Width)
parser = uffparser.create_uff_parser()
parser.register_input('images_placeholder', [3,224,224], uffparser.UffInputOrder_kNCHW)

# Create a tensorRT engine which is ready for immediate use.
# For this example, we will serialize it for fast instantiation later.
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser,
                                     args.max_batch_size, 1 << args.max_workspace_size, trt.infer.DataType.FLOAT)
assert (engine)
```

### Step 4: Serialize TensorRT engine to PLAN file

The best way to reuse a TensorRT engine is to serialize it to a PLAN file. To serialize to disk, we can directly 
call the following function. Since this code is embedded in deploy-to-tensorrt.py, there is no need to run this code 
as a separate step.

```python
import tensorrt as trt

# Create engine above

serialized = engine.serialize()
trt.utils.write_engine_to_file('model/model.PLAN', engine.serilize())
engine.destroy()
```

Thsi will create a model.PLAN file in the model folder.

### Step 5: Load TensorRT engine and run inference

Finally, we can reload the serialized engine and call infer to get the inference results. We used the LITE engine,
which greatly simplifies the interface for running inference--all you need to do is call engine.infer(input). Behind 
the scenes, the LITE engine handles allocating memory, loading the input into the memory buffers, running inference, 
and loading the output from the memory buffers.

```python
import tensorrt as trt

# Create the engine
engine = trt.lite.Engine(PLAN='model/model.PLAN')

# Run inference
result = engine.infer(image)[0]

```
