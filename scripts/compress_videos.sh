#!/bin/bash

INPUT_DIR="/home/techteam/0xmudit_cattle_reID/input"
OUTPUT_DIR="/home/techteam/0xmudit_cattle_reID/input_compressed"
TARGET_SIZE_MB=50

mkdir -p "$OUTPUT_DIR"

for video in "$INPUT_DIR"/*.mp4; do
    filename=$(basename "$video")
    filesize=$(stat -c%s "$video")
    filesize_mb=$((filesize / 1024 / 1024))

    echo "========================================="
    echo "Processing: $filename"
    echo "Original size: ${filesize_mb} MB"

    if [ "$filesize_mb" -le "$TARGET_SIZE_MB" ]; then
        echo "Already under ${TARGET_SIZE_MB}MB, copying as-is."
        cp "$video" "$OUTPUT_DIR/$filename"
        continue
    fi

    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$video" | cut -d. -f1)

    # Target 48MB to have some margin
    target_bytes=$((48 * 1024 * 1024))
    bitrate=$((target_bytes * 8 / duration))
    # Subtract audio bitrate (128k)
    video_bitrate=$((bitrate - 128000))

    if [ "$video_bitrate" -lt 100000 ]; then
        video_bitrate=100000
    fi

    echo "Duration: ${duration}s, Target video bitrate: $((video_bitrate / 1000))k"

    # Pass 1
    echo "  Pass 1/2..."
    ffmpeg -y -i "$video" \
        -c:v libx264 -b:v "${video_bitrate}" -pass 1 \
        -an -f null /dev/null 2>/dev/null

    # Pass 2
    echo "  Pass 2/2..."
    ffmpeg -y -i "$video" \
        -c:v libx264 -b:v "${video_bitrate}" -pass 2 \
        -c:a aac -b:a 128k \
        "$OUTPUT_DIR/$filename" 2>/dev/null

    new_size=$(stat -c%s "$OUTPUT_DIR/$filename")
    new_size_mb=$((new_size / 1024 / 1024))
    echo "Compressed: ${new_size_mb} MB"
    echo ""

    # Clean up ffmpeg2pass logs
    rm -f ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree
done

echo "========================================="
echo "Done! Compressed videos saved to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
