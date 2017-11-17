# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
"""
Preprocesses images before inference.

Includes reading an image from a file path, normalizing to [0,1], 
resizing, and converting to CHW (Channel, Height, Width) format.
"""
import cv2
import numpy as np

class ImagePreProcessor(object):

    def get_single_image(self, file_path, height, width):
        """Preprocesses a single image at the given file path"""

        rgb_image = self.get_rgb_image(file_path)

        normalized_image = self.normalize_image(rgb_image)

        resized_image = self.resize_image(normalized_image, height, width)

        CHW_image = self.convert_HWC_to_CHW(resized_image)

        return CHW_image


    def get_rgb_image(self, file_path):
        """Reads a color image from the given file path and returns it in RGB"""

        # Read from file
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)

        # Convert image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def normalize_image(self, image, bits = 8):
        """Normalizes an RGB image to [0,1]"""

        # Cast image array to float32 and normalize
        image = image.astype(np.float32)
        scale = 1.0 / (2**bits - 1)
        image = scale * image
        return image

    def resize_image(self, image, height, width, interpolation = cv2.INTER_LINEAR):
        """Resizes an image to the specified Height x Width"""
        image = cv2.resize(image, (height, width), 0, 0, interpolation)
        return image

    def convert_NHWC_to_NCHW(self, images):
        """Converts a batch of images from NHWC to NCHW format"""
        return np.transpose(images, [0,3,1,2])

    def convert_HWC_to_CHW(self, image):
        """Converts a single image from HWC to CHW format"""
        return np.transpose(image, [2,0,1])
