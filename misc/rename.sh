#!/bin/bash

# Script Name: rename.sh
# Description: Recursively renames all directories and files by replacing a specified substring with another substring.
# Usage: ./rename.sh <search_substring> <replace_substring> [directory]
# If no directory is provided, the current directory is used.

# Function to display usage information
usage() {
    echo "Usage: $0 <search_substring> <replace_substring> [directory]"
    echo "  <search_substring>: The substring to search for in file and directory names."
    echo "  <replace_substring>: The substring to replace the search substring with."
    echo "  [directory]: Optional. The target directory to perform renaming. Defaults to the current directory."
    exit 1
}

# Check for the required number of arguments (at least 2, at most 3)
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    usage
fi

SEARCH=$1
REPLACE=$2

# Determine the target directory
if [ "$#" -eq 3 ]; then
    TARGET_DIR="$3"
    # Check if the specified directory exists
    if [ ! -d "$TARGET_DIR" ]; then
        echo "Error: Directory '$TARGET_DIR' does not exist."
        exit 1
    fi
else
    TARGET_DIR="."
fi

echo "Starting renaming process in directory: '$TARGET_DIR'"
echo "Replacing substring '$SEARCH' with '$REPLACE'."

# Rename directories first to avoid path issues
find "$TARGET_DIR" -depth -type d -name "*$SEARCH*" | while IFS= read -r DIR; do
    # Determine the new directory name by replacing the search substring with the replace substring
    NEW_DIR="$(dirname "$DIR")/$(basename "$DIR" | sed "s/$SEARCH/$REPLACE/g")"

    # Rename the directory
    if mv "$DIR" "$NEW_DIR"; then
        echo "Renamed directory: '$DIR' -> '$NEW_DIR'"
    else
        echo "Failed to rename directory: '$DIR'"
    fi
done

# Rename files
find "$TARGET_DIR" -depth -type f -name "*$SEARCH*" | while IFS= read -r FILE; do
    # Determine the new file name by replacing the search substring with the replace substring
    NEW_FILE="$(dirname "$FILE")/$(basename "$FILE" | sed "s/$SEARCH/$REPLACE/g")"

    # Rename the file
    if mv "$FILE" "$NEW_FILE"; then
        echo "Renamed file: '$FILE' -> '$NEW_FILE'"
    else
        echo "Failed to rename file: '$FILE'"
    fi
done

echo "Renaming process completed."
