#!/bin/sh

HEXFILE="${1}_hex"
STRIPPEDFILE="${1}_stripped"
OUTFILE="${1}_dec"

echo "ibase=16">$HEXFILE
sed -r 's/0x(.*)->.0x([0-9A-F][0-9A-F])([0-9A-F][0-9A-F])([0-9A-F][0-9A-F])([0-9A-F][0-9A-F])(.*)/if(\2>7F) -\2+7F else  \2\
if(\3>7F) -\3+7F else  \3\
if(\4>7F) -\4+7F else  \4\
if(\5>7F) -\5+7F else  \5/' $1 >>$HEXFILE
tr '\r' '\n' < $HEXFILE > $STRIPPEDFILE
sed '/^$/d' $STRIPPEDFILE >$HEXFILE
cat $HEXFILE | bc > $OUTFILE
rm $HEXFILE
rm $STRIPPEDFILE

