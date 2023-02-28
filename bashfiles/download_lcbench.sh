#!/bin/bash

mkdir ./data
mkdir ./data/downloads
mkdir ./data/downloads/LCBench
mkdir ./data/raw
mkdir ./data/raw/LCBench
mkdir ./data/preprocessing
mkdir ./data/preprocessing/LCBench

cd ./data/downloads/LCBench

wget https://figshare.com/ndownloader/files/21188673 -O meta_features.json
wget https://figshare.com/ndownloader/files/22859435 -O bench_full.zip
wget https://figshare.com/ndownloader/files/21188607 -O data_2k.zip
wget https://figshare.com/ndownloader/files/21188598 -O data_2k_lw.zip
wget https://figshare.com/ndownloader/files/21001311 -O mnist.zip
wget https://figshare.com/ndownloader/articles/11604705/versions/1 -O six_datasets_lw.zip


unzip '*.zip'
rm *.zip

# using wget with some input txt file specifiying the urls
