#!/bin/bash

# converts all files in the audio directory that are not .wav to .wav

cd audio
for i in *; do
    filename="${i%.*}"
    extension=".${i##*.}"

    if [ "$extension" != ".wav" ]; then
        echo "Converting $i to $filename.wav..."
        ffmpeg -i "$i" "$filename.wav"
        rm "$i"
    fi

done
