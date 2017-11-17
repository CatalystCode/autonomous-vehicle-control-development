# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

$pwd = (Get-Location).Path

docker build --pull -t deployvms .
docker run -it `
    --mount type=bind,source="$pwd",target=/src/provisioning/ `
    -w /src/provisioning/ deployvms /bin/bash
