#!/bin/bash
set -e -x

os_major_version=$(cat /etc/redhat-release | tr -dc '0-9.'|cut -d \. -f1)

echo "installing for CentOS version : $os_major_version"
yum install -y centos-release-scl-rh
#TODO: Change devtoolset-10* to devtoolset-11* once cuda >=11.6, also remove centos-release-scl from the list and remove line #10 $scl enable devtoolset-10 bash
yum install -y which gdb redhat-lsb-core expat-devel tar unzip zlib-devel make libunwind  aria2  bzip2 bzip2-devel java-11-openjdk-devel graphviz centos-release-scl devtoolset-10-binutils devtoolset-10-gcc devtoolset-10-gcc-c++ devtoolset-10-gcc-gfortran python3 python3-pip
scl enable devtoolset-10 bash
pip3 install --upgrade pip
localedef -i en_US -f UTF-8 en_US.UTF-8
