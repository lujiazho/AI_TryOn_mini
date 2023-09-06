// expand mask ----------------------------------------------------------------
function expandImageData(imageData) {
    // Create a new empty ImageData object with the same dimensions
    const expandedImageData = new ImageData(imageData.width, imageData.height);
    let pixels = [];

    // Iterate through all the pixels in the original image data
    for (let y = 0; y < imageData.height; y++) {
        for (let x = 0; x < imageData.width; x++) {
            const index = (y * imageData.width + x) * 4;
            const alpha = imageData.data[index + 3];

            // If the current pixel is not transparent, set the corresponding pixels in the expanded image
            if (alpha !== 0) {
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        const newX = x + dx;
                        const newY = y + dy;

                        // Check if the new coordinates are within the image bounds
                        if (newX >= 0 && newX < imageData.width && newY >= 0 && newY < imageData.height) {
                            const newIndex = (newY * imageData.width + newX) * 4;
                            expandedImageData.data[newIndex] = imageData.data[index];
                            expandedImageData.data[newIndex + 1] = imageData.data[index + 1];
                            expandedImageData.data[newIndex + 2] = imageData.data[index + 2];
                            expandedImageData.data[newIndex + 3] = 255;

                            // add index to pixels
                            pixels.push(newIndex);
                        }
                    }
                }
            }
        }
    }

    drawnPaths.push({
      points: pixels,
      type: "expand",
    });

    return expandedImageData;
}

const expandButton = document.getElementById('expand-mask')
expandButton.addEventListener('click', function() {
    stopMagicDrawing();
    stopDrawing();
    stopErasing();
    stopRecting();

    // Get the current mask image data
    const imageData = context.getImageData(0, 0, canvas.width, canvas.height);

    // Apply the expansion to the mask image data
    const expandedImageData = expandImageData(imageData);

    // Draw the expanded mask image back on the canvas
    context.putImageData(expandedImageData, 0, 0);
});