#!/usr/bin/env bash

SPDZROOT="/MP-SPDZ"

session_id=${1:-1}
invocation_key=${2:-2}

# if symlink, readlink returns not empty string (the symlink target)
# if string is not empty, test exits w/ 0 (normal)
#
# if non symlink, readlink returns empty string
# if string is empty, test exits w/ 1 (error)
simlink? () {
  test "$(readlink "${1}")";
}

data_dir=$SPDZROOT/tmp/$session_id/$invocation_key
mkdir -p $data_dir

if simlink? "${data_dir}/Player-Data"; then
  echo ${data_dir}/Player-Data is a symlink
else
  echo creating the symlink ${data_dir}/Player-Data
  ln -s $SPDZROOT/Player-Data/ $data_dir
fi


