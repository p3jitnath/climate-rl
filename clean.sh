#!/bin/sh

# Check if a command-line argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <folder_name>"
    exit 1
fi

# Set the folder name from the first argument
FOLDER_NAME=$1

# Remove the specified folder from each directory
rm -rf "runs/$FOLDER_NAME"
rm -rf "videos/$FOLDER_NAME"
rm -rf "wandb/$FOLDER_NAME"
