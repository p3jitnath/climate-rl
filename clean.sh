#!/bin/sh

# Check if a command-line argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <tag_name>"
    exit 1
fi

# Set the tag name from the first argument
TAG_NAME=$1

# Remove the specified tag from each directory
rm -rf "runs/$TAG_NAME"*
rm -rf "videos/$TAG_NAME"*
rm -rf "slurm"/*.err
rm -rf "slurm"/*.out
rm -rf "tune/tmp"/*.tmp
