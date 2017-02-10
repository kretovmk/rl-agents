#!/bin/bash
S="docker run -v "$PWD"/:/exp dementrock/rllab3-shared python /exp/"$1
$S