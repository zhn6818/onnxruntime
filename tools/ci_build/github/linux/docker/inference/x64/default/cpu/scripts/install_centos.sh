#!/bin/bash
set -e -x

os_major_version=$(cat /etc/redhat-release | tr -dc '0-9.'|cut -d \. -f1)
TOOL_VERSION="-11-11.0-3.el7.x86_64"
echo "installing for CentOS version : $os_major_version"
yum install -y centos-release-scl-rh
yum install -y which gdb redhat-lsb-core expat-devel tar unzip zlib-devel make libunwind  aria2  bzip2 bzip2-devel java-11-openjdk-devel graphviz devtoolset-11-binutils devtoolset-11-gcc$TOOL_VERSION devtoolset-11-gcc-c++$TOOL_VERSION devtoolset-11-gcc-gfortran$TOOL_VERSION python3 python3-pip

pip3 install --upgrade pip
localedef -i en_US -f UTF-8 en_US.UTF-8
