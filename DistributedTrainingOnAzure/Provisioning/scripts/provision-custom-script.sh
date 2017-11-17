#!/bin/bash

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

SUBSCRIPTION=$1
RESOURCE_GROUP=$2

set -e

cd "${BASH_SOURCE%/*}" || exit

# get GPU vm names
VMS=$(az vm list --resource-group $RESOURCE_GROUP --query "[?contains(hardwareProfile.vmSize,'NC')].name")
VMNames=$(echo $VMS | jq '.[]')

for vmName in $VMNames
do 
    vmName=$(echo $vmName | tr -d '\"')
    echo "Found GPU vm ${vmName}"

    Extensions=$(az vm extension list \
        --resource-group $RESOURCE_GROUP \
        --vm-name "${vmName}" --query "[?contains(virtualMachineExtensionType,'CustomScript')].{name: name}")
    ExtensionsNames=$(echo $Extensions | jq '.[].name')

    for ext in $ExtensionsNames
    do
        ext=$(echo $ext | tr -d '\"')
        echo "Deleting extension ${ext} on vm ${vmName}"

        az vm extension delete \
            --name "${ext}"\
            --resource-group $RESOURCE_GROUP \
            --vm-name "${vmName}"

    done
    
done

# use azure deploy to run the script
echo "Deploying new custom script"
VMSCompact=$(echo $VMS | jq -c '.')

az group deployment create \
    --name "CustomScripts" \
    --resource-group "${RESOURCE_GROUP}" \
    --template-file "./vm-custom-script.json" \
    --parameters "{\"vmNames\":{\"value\":${VMSCompact}}}"
