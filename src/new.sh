#!/bin/bash


echo $1 $2

first_folder="${1/.\//}"
second_folder="${2/.\//}"
new_folder="$first_folder"_$second_folder


echo $new_folder