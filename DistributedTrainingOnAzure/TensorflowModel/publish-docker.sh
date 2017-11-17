#!/bin/bash

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

REPO="$1"

docker push ${REPO}/distrib-tensorflow-cars:ps
docker push ${REPO}/distrib-tensorflow-cars:worker-gpu