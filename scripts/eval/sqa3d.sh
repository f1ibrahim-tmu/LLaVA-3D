#!/bin/bash


declare -A arr
KEY=""
VALUE=""
for ARGUMENT in "$@"; do
        
        if [[ ${ARGUMENT:0:2} == "--" ]]; then

                if [[ $KEY != "" ]]; then
                        arr["$KEY"]="$VALUE"
                        KEY=""
                        VALUE=""
                fi

                KEY=${ARGUMENT:2}
        else
                if [[ $VALUE == "" ]]; then
                        VALUE=$ARGUMENT
                else
                        VALUE="$VALUE $ARGUMENT"
                fi
        fi

done

if [[ $KEY != "" ]]; then
        arr["$KEY"]="$VALUE"
fi




echo "arguments:"
for key in "${!arr[@]}"
do
  echo "Key: $key, Value: ${arr[$key]}"
done


export CUDA_VISIBLE_DEVICES="$GPU"
