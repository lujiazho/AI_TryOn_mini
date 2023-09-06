// Add event listener for the draw-rect button
drawRectButton.addEventListener('click', () => {
    // stop any other drawing
    stopDrawing();
    stopMagicDrawing();
    stopErasing();
    // stop drawing
    if (isDrawingRect) {
        canvas.style.cursor = 'auto';
        isDrawingRect = false;
    }
    // start drawing
    else {
        canvas.style.cursor = `crosshair`; // change cursor to a crosshair
        isDrawingRect = true; 
    }
});