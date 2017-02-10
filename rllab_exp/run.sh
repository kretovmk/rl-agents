#!/bin/bash
S="docker run -v "$PWD"/:/exp dd210/docker-rllab python /exp/"$1
$S
