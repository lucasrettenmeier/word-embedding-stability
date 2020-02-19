#!/bin/usr/env sh
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e

ROOT=${1}
FILE=${2}

$ROOT/filter_utf8 < "$FILE.txt" \
    | $ROOT/dedup > "$FILE.dedup"
