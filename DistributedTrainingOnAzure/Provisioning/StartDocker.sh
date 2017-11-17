#!/bin/bash

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

set -euo pipefail
set -x

docker build --pull -t deployvms .

docker run -it \
    --privileged \
    -v /var/run/docker.sock:/var/run/docker.sock \
    --mount type=bind,source=`pwd`,target=/src/provisioning/ \
    -v ~/.azure:/root/.azure \
    -w /src/provisioning/ deployvms /bin/bash

chown -R "$(logname):$(id -gn $(logname))" . ~/.azure
