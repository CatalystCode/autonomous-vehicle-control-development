#!/bin/bash

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

SSH_KEY_PATH=$1
USERNAME=$2
HOST=$3

cp "${SSH_KEY_PATH}" /tmp/sshkey
chmod 700 /tmp/sshkey

mkdir -p ~/.kube/

scp -C -i /tmp/sshkey $USERNAME@$HOST:.kube/config ~/.kube/config