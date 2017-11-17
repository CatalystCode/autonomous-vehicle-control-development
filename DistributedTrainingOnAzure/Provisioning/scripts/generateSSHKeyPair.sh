#!/bin/bash

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

keyname=$1
passphrase=$2
ssh-keygen -t rsa -b 2048 -f "$keyname" -q -N "${passphrase}"
