#!/usr/bin/env bash

NAME="nerf"
WORKDIR=/home/$USER/nerf

# Check if the container exists.
if docker ps -a --format "{{.Names}}" | grep -q "^$NAME$"; then
  echo >&2 "Container [$NAME] already exists"
  echo >&2 "Run 'docker stop $NAME && docker rm $NAME' before creating a new one"
  exit
fi

# The directory contains current script.
DIR=$(dirname $(realpath "$BASH_SOURCE"))

docker run -it -d --name $NAME \
  --user $(id -u -n) \
  --privileged \
  --net host \
  --hostname $NAME \
  --add-host in_docker:127.0.0.1 \
  --add-host $(hostname):127.0.0.1 \
  -e DISPLAY \
  -v /mnt:/mnt \
  -v $DIR/../../:$WORKDIR \
  -w $WORKDIR \
  $NAME \
  /bin/bash

echo "Done"
