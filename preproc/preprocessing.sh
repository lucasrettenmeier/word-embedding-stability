#!/usr/bin/env sh
set -e

export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

LANG=${1}

DOWNLOAD_DATE=20190901
ROOT=/home/rettenls/code/preproc
DATA=/home/rettenls/data/texts/wiki/
FILE=/home/rettenls/data/texts/wiki/"$LANG"/preproc/preproc
THREAD_NUM=32

echo "Working directory: $ROOT"
echo "Language: $LANG"

## Convert XML Dump to Text
sh $ROOT/wiki_to_text.sh "$DATA" "$LANG" "$DOWNLOAD_DATE" "$FILE"

## Deduplication
g++ -std=c++11 -O3 -o $ROOT/dedup $ROOT/dedup.cc
g++ -std=c++11 -O3 -o $ROOT/filter_utf8 $ROOT/filter_utf8.cc
sh $ROOT/filter_dedup.sh "$ROOT" "$FILE"

## Data Filtering + Tokenization
#git clone https://github.com/moses-smt/mosesdecoder.git

if [ $LANG = "hi" ]
then
	python3  $ROOT/icu_tokenizer.py "$FILE".dedup "$FILE".tok "$LANG"
elif [ $LANG = "zh" ]
then
	sh $ROOT/stanford_segmenter/segment.sh -k ctb "$FILE".dedup UTF-8 0 > "$FILE".ctb.k.tok
else
	perl $ROOT/mosesdecoder/scripts/tokenizer/tokenizer.perl -no-escape -l $LANG -threads 72 < "$FILE".dedup > "$FILE".tok 
fi