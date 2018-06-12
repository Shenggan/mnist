#!/bin/sh

set -e

wget -c http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
gzip -d mnist.pkl.gz 
mv mnist.pkl ../mnist/