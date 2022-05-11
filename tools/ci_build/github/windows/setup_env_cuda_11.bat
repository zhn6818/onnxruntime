REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

REM This file is used by "DML Nuget Pipeline", "Zip-Nuget-Java-Nodejs Packaging Pipeline", "Windows GPU CI Pipeline".
set PATH=C:\azcopy;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\extras\CUPTI\lib64;%PATH%
set GRADLE_OPTS=-Dorg.gradle.daemon=false
