#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/deepface
DATA=data/flip
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/deepface_train_lmdb \
  $DATA/deepface_mean.binaryproto

echo "Done."
