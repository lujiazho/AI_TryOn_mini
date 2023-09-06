// eraser tool ----------------------------------------------------------------
const eraserButton = document.getElementById('eraser');

eraserButton.addEventListener('click', (event) => {
  stopDrawing();
  stopMagicDrawing();
  stopRecting();

  // stop erase drawing
  if (isErasing) {
    canvas.style.cursor = 'auto';
    isErasing = false;
  }
  // start erase
  else {
    canvas.style.cursor = `crosshair`; // change cursor to a crosshair
    isErasing = true; 
  }
});