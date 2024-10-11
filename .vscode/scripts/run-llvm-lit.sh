#!/bin/bash

# Extract the workspace folder and filename from the arguments
workspace_folder="$1"
filename="$2"

# Define the prefix to strip
prefix="${workspace_folder}/chemcomp/test/"

# Strip the prefix from the filename
modified_filename=$(echo "$filename" | sed "s|^$prefix||")

# Run the llvm-lit command with the modified filename
FILECHECK_OPTS="--color" "${workspace_folder}/builds/llvm-project/build/bin/llvm-lit" -v "$modified_filename"