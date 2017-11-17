# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
"""
Converts a frozen Tensorflow model file (.pb) to a serialized TensorRT PLAN file.

Currently, it supports one input placeholder and multiple output placeholders.

An example of usage:
python deploy-to-tensorrt.py
--frozen_file=model/frozen_model.pb \
--input_placeholder=images_placeholder_name \
--dimensions=3,244,244 \
--output_placeholders=softmax_tensor
--file_path=model/model.PLAN
"""
import argparse
import errno
import os

import tensorrt as trt
import tensorrt.parsers.uffparser as uffparser
import uff

parser = argparse.ArgumentParser(
    description="Converts a frozen Tensorflow model to a serialized PLAN for TensorRT.")

parser.add_argument('--frozen_file',
                    type=str,
                    required=True,
                    help='Path to the frozen .pb file from Tensorflow.')

parser.add_argument('-i', '--input_placeholder',
                    type=str,
                    default='x',
                    help='Name of the input placeholder for the tensorflow model')

parser.add_argument('-d', '--dimensions',
                    type=str,
                    default='3,224,224',
                    help='Dimensions for the input placeholder in Channel,Height,Width (CHW).' +
                        'Example: 3,224,224 (for a 3 channel RGB 224 x 224 image)'
                    )

parser.add_argument('-o', '--output_placeholders',
                    type=str,
                    default='y,z',
                    help='Names of the output placeholders for the tensorflow model. Comma delimited.')

parser.add_argument('-f', '--file_path',
                    type=str,
                    default='/tmp/trt_output/model.PLAN',
                    help='Path where serialized plan will be outputted.')

parser.add_argument('--max_batch_size',
                    type=int,
                    default=1,
                    help='Max batch size for the number of input images. ' +
                    'Engine will be optimized for this batch size--' +
                    'smaller batch sizes can be used at runtime, but performance will be optimal at this size.')

parser.add_argument('--max_workspace_size',
                    type=int,
                    default=20,
                    help='Max workspace size for the engine.')

def main(args):

    input = [args.input_placeholder]
    output = args.output_placeholders.split(',')

    dims = map(int, args.dimensions.split(','))
    assert (len(dims) == 3), 'Input dimensions must be given in CHW format.'

    # Convert tensorflow pb file to uff stream for tensorRT
    uff_model = uff.from_tensorflow_frozen_model(frozen_file=args.frozen_file,
                                                 input_nodes=input,
                                                 output_nodes=output)

    # Create parser for uff file and register input placeholder
    parser = uffparser.create_uff_parser()
    parser.register_input(args.input_placeholder, dims, uffparser.UffInputOrder_kNCHW)

    # Create a tensorRT engine which is ready for immediate use.
    # For this example, we will serialize it for fast instantiation later.
    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
    engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser,
                                         args.max_batch_size, 1 << args.max_workspace_size, trt.infer.DataType.FLOAT)
    assert (engine)

    # Serialize the engine to given file path
    serialize_engine(engine, args.file_path)
    engine.destroy()

def serialize_engine(engine, output_file):
    """Serializes a given ICudaEngine to a PLAN file"""
    try:
        os.makedirs(os.path.dirname(output_file))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

    serialized = engine.serialize()
    trt.utils.write_engine_to_file(output_file, serialized)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)