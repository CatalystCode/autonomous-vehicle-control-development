#!/bin/bash

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

STORAGE_NAME=$1
RESOURCE_GROUP=$2
LOCATION=$3

STORAGE=$(az storage account show --resource-group "${RESOURCE_GROUP}" --name "${STORAGE_NAME}")
SHARE_NAME="share"

if [ -z "${STORAGE}" ]
then
    echo "Storage account not found. Creating..."

    # Create storage account, if not exists
    az storage account create \
        --location $LOCATION \
        --name $STORAGE_NAME \
        --resource-group $RESOURCE_GROUP \
        --sku Standard_LRS
else
    echo "Storage account already exists. Skipping creation..."
fi

# Get connection string of storage account
CONNECTION_STRING=$(az storage account show-connection-string -n "$STORAGE_NAME" -g "${RESOURCE_GROUP}" --query 'connectionString' -o tsv)

if [ -z "${CONNECTION_STRING}" ]
then  
    echo "Couldn't retrieve the connection string."
    exit 1
fi

echo "Creating the file share."
az storage share create --name "${SHARE_NAME}" --quota 2048 --connection-string $CONNECTION_STRING > /dev/null
