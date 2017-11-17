#!/bin/bash
# 
# Copyright 2016 Google Inc. All Rights Reserved.
# Modified 2017 xiou@microsoft.com. Changed to preprocess different cars dataset.
#
# Based on image preprocessing code from 
# https://github.com/tensorflow/models/blob/master/research/inception/inception/data/download_and_preprocess_flowers.sh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

if [ -z "$3" ]; then
    echo "Usage: prepare_data.sh [car zip] [no car zip] [out dir]"
    exit
fi

CAR_ZIP="$1"
NO_CAR_ZIP="$2"
DATA_DIR="$3"
RAW_DIR="${DATA_DIR}/raw"

echo "Creating data directory"
mkdir -p $RAW_DIR

TRAIN_DIR="${RAW_DIR}/train"
VALIDATION_DIR="${RAW_DIR}/validation"

echo "Cleaning training and validation directories"
rm -rf ${TRAIN_DIR} ${VALIDATION_DIR}

echo "Extracting cars data from zip file at ${CAR_ZIP}"
unzip "${CAR_ZIP}" -d "${TRAIN_DIR}"

echo "Extracting cars data from zip file at ${NO_CAR_ZIP}"
unzip "${NO_CAR_ZIP}" -d "${TRAIN_DIR}"

LABELS_FILE="${RAW_DIR}/labels.txt"
ls -1 "${TRAIN_DIR}" | sed 's/\///' | sort > "${LABELS_FILE}"

while read LABEL; do
    echo "Processing label ${LABEL}"
    LABEL_TRAIN_DIR="${TRAIN_DIR}/${LABEL}/"
    LABEL_VALIDATION_DIR="${VALIDATION_DIR}/${LABEL}/"

    echo "Making label validation dir ${LABEL_VALIDATION_DIR}"
    mkdir -p $LABEL_VALIDATION_DIR

    VALIDATION_IMAGES=$(ls -1 "${LABEL_TRAIN_DIR}" | shuf | head -100)

    for IMAGE in ${VALIDATION_IMAGES}; do
        mv -vf "${LABEL_TRAIN_DIR}/$IMAGE" "${LABEL_VALIDATION_DIR}"
    done
done < "${LABELS_FILE}"

python python/build_image_data.py --train_directory="${TRAIN_DIR}" \
    --validation_directory="${VALIDATION_DIR}" \
    --output_directory="${DATA_DIR}" \
    --labels_file="${LABELS_FILE}"

