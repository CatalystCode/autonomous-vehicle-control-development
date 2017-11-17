#!/bin/bash

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

SUBSCRIPTION=$1
RESOURCE_GROUP=$2

az login

az account set --subscription "${SUBSCRIPTION}"

echo "Deleting resource group..."
az group delete --name "${RESOURCE_GROUP}"