# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
"""
Loads a PLAN file into a LITE TensorRT runtime engine.
Then, performs inference on the given image.

Currently, processes 1 image at at a time.
To run batches, convert batch of images to NCHW and pass to engine.infer method.
"""
import argparse
import numpy as np
import tensorrt as trt

from image_processor import ImagePreProcessor

parser = argparse.ArgumentParser(
    description="Creates a runtime engine for TensorRT and performs inference.")

parser.add_argument('--plan_file',
                    type=str,
                    required=True,
                    help='Path to the PLAN file for the serialized engine.')

parser.add_argument('--image_path',
                    type=str,
                    help='Path to the input image.')

parser.add_argument('--desired_size',
                    type=str,
                    default='3,224,224',
                    help='Image size needed for prediction.')

def set_up_engine(engine_file_path):
    engine = trt.lite.Engine(PLAN=engine_file_path)
    assert (engine)
    return engine

def infer(engine, image):
    result = engine.infer(image)
    return result[0]

def main(args):

    dims = map(int, args.desired_size.split(','))
    assert (len(dims) == 3), "Dimensions must be given as 'channel,height,width'"

    image = ImagePreProcessor().get_single_image(args.image_path, dims[1], dims[2])

    # Load the serialized engine from given PLAN file
    engine = set_up_engine(args.plan_file)

    # Run inference
    result = infer(engine,image)

    print ("Result: %s" % result)

    print ("Decision: %s" % ("Car" if np.argmax(result) == 0 else "NoCar"))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
