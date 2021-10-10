#!/bin/bash
folder=$1

for d in $folder*/ ; do
        ( cd "$d" && zip -r "${d%/}".zip . )
done
