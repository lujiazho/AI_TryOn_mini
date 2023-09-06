// undo button ----------------------------------------------------------------
const undoPathButton = document.getElementById('undo-path');

undoPathButton.addEventListener('click', () => {
  if (promptPoints.length > 0) {
      const lastPoint = promptPoints.pop();
      const lastPath = interactivePaths.pop();

      redrawPaths(canvas, drawnPaths);

      const curLen = interactivePaths.length;
      if (curLen > 0) {
          putDataOnCanvas(canvas, interactivePaths[curLen - 1]);
      }

      for (const thisPrompt of promptPoints) {
          drawPromptPointOnClick(thisPrompt, canvas);
      }

  }
  else if (drawnPaths.length > 0) {
    // Remove the last path from the array
    const lastPath = drawnPaths.pop();
    console.log(lastPath)

    if (lastPath.type === 'magic') {
      fetch("/undo", {
          method: "POST",
      })
      .then((response) => response.json())
      .then((data) => {
          console.log("Success:", data);
      })
      .catch((error) => {
          console.error("Error:", error);
      }).finally(() => {
          // // Hide the spinner
          // spinnerContainer.classList.add("d-none");
      });
    }

    // Clear the canvas
    context.clearRect(0, 0, canvas.width, canvas.height);

    // Redraw the remaining paths
    for (const path of drawnPaths) {
      if (path.type === "magic" || path.type === "expand") {
        const canvasData = context.getImageData(0, 0, canvas.width, canvas.height);
        const data = canvasData.data;

        for (let i = 0; i < path.points.length; i += 1) {
          data[path.points[i]] = replacementColor.r; // red
          data[path.points[i]+1] = replacementColor.g; // green
          data[path.points[i]+2] = replacementColor.b; // blue
          data[path.points[i]+3] = 255; // alpha
        }
        context.putImageData(canvasData, 0, 0);

      } else {
        context.lineWidth = path.lineWidth;
        for (const point of path.points) {
          context.beginPath();
          context.moveTo(point.fromX, point.fromY);
          context.lineTo(point.toX, point.toY);
          if (path.type === "eraser") {
            context.globalCompositeOperation = 'destination-out';
          } else {
            context.globalCompositeOperation = 'source-over';
          }
          context.strokeStyle = path.type === "brush" ? 'rgba(0, 0, 0, 1)' : 'rgba(0, 0, 0, 1)';
          context.stroke();
          context.beginPath();
          context.arc(point.toX, point.toY, path.lineWidth / 2, 0, 2 * Math.PI);
          context.fillStyle = path.type === "brush" ? 'rgba(0, 0, 0, 1)' : 'rgba(0, 0, 0, 1)';
          context.fill();
        }
      }
    }
  }
});