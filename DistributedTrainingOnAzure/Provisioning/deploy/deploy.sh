#!/bin/bash

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
#
# Usage:
# ./deploy.sh <subscription_id> <resource-group> <location>
#
# If provisioning N-Series VMs, make sure to choose a location that supports the specific N-Series class.
#

SUBSCRIPTION_ID=$1
RESOURCE_GROUP=$2
LOCATION=$3

set -e

cd "${BASH_SOURCE%/*}" || exit

az login
az account set --subscription "${SUBSCRIPTION_ID}"

SECS=$(date +%s)

# Deploy azure resources from kubernetes.json file 
# Auto-generates service principal 
# Auto-generates ssh key in _output/autocluster/azureuser_rsa
acs-engine deploy \
    --subscription-id "${SUBSCRIPTION_ID}" \
    --resource-group "${RESOURCE_GROUP}" \
    --dns-prefix autocluster \
    --location "${LOCATION}" \
    --auto-suffix \
    --api-model kubernetes.json \
    --output-directory ./_output/autocluster${SECS}/

# Create storage account and file share
../scripts/provision-storage-file-share.sh "autoclusterpod${SECS}" "${RESOURCE_GROUP}" "${LOCATION}"

# deploy custom script for GPU (if needed)
../scripts/provision-custom-script.sh "${SUBSCRIPTION_ID}" "${RESOURCE_GROUP}"

# get kube.config from master (persist on local machine)
DNS=$(jq '.parameters.masterEndpointDNSNamePrefix.value' "./_output/autocluster${SECS}/azuredeploy.parameters.json")
DNS=$(echo $DNS | tr -d '\"')
$LOCATION=$(echo $LOCATION | tr -d '\"')

echo "Getting kube.config from $DNS.$LOCATION.cloudapp.azure.com"
../scripts/configure-local-kubectl.sh ./_output/autocluster${SECS}/azureuser_rsa azureuser $DNS.$LOCATION.cloudapp.azure.com

echo "Initializing helm in cluster..."
helm init