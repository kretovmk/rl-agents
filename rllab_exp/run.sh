#!/bin/bash
S="docker run -v "$PWD"/:/exp dementrock/rllab-shared python /exp/"$1
$S