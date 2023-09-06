// canvas image xy to original image xy
function getXYLocationInOriImage(x, y) {
    x -= scaledX;
    y -= scaledY;

    x /= scaleFactor;
    y /= scaleFactor;

    x = Math.round(x);
    y = Math.round(y);

    return { x, y };
  }

function getEventLocationInOriImage(event) {
    let x = event.offsetX;
    let y = event.offsetY;

    const scaleFactor = Math.min(canvas.width / originalUnresizedImageData.width, canvas.height / originalUnresizedImageData.height);
    const scaledWidth = originalUnresizedImageData.width * scaleFactor;
    const scaledHeight = originalUnresizedImageData.height * scaleFactor;

    const delta_x = (canvas.width - scaledWidth) / 2;
    const delta_y = (canvas.height - scaledHeight) / 2;

    x -= delta_x;
    y -= delta_y;

    x /= scaleFactor;
    y /= scaleFactor;

    x = Math.round(x);
    y = Math.round(y);

    return { x, y };
}

function drawPromptPointOnClick(thisPrompt, canvas) {
    let x = thisPrompt.origPoint[0];
    let y = thisPrompt.origPoint[1];

    const fillColor = `rgba(255, 255, 255, 0.75)`;
    const strokeColor = thisPrompt.type ? `green` : `red`;

    const context = canvas.getContext("2d");

    context.beginPath();
    context.arc(x, y, 3, 0, Math.PI * 2);
    context.fillStyle = fillColor;
    context.fill();
    context.strokeStyle = strokeColor;
    context.stroke();
}

function putDataOnCanvas(thisCanvas, pixels) {
    const thisContext = thisCanvas.getContext("2d");
    const canvasData = thisContext.getImageData(
        0,
        0,
        thisCanvas.width,
        thisCanvas.height
    );
    const data = canvasData.data;

    for (let i = 0; i < pixels.length; i += 1) {
        data[pixels[i]] = replacementColor.r; // red
        data[pixels[i] + 1] = replacementColor.g; // green
        data[pixels[i] + 2] = replacementColor.b; // blue
        data[pixels[i] + 3] = 255; // alpha
    }
    thisContext.putImageData(canvasData, 0, 0);
}

function redrawPaths(thisCanvas, thisDrawnPaths) {
    // Clear the thisCanvas
    const thisContext = thisCanvas.getContext("2d");
    thisContext.clearRect(0, 0, thisCanvas.width, thisCanvas.height);

    // Redraw the remaining paths
    for (const path of thisDrawnPaths) {
        if (path.type === "magic" || path.type === "rect") {
            putDataOnCanvas(thisCanvas, path.points);
        }
        else {
            thisContext.lineWidth = path.lineWidth;
            for (const point of path.points) {
            thisContext.beginPath();
            thisContext.moveTo(point.fromX, point.fromY);
            thisContext.lineTo(point.toX, point.toY);
            if (path.type === "eraser") {
                thisContext.globalCompositeOperation = 'destination-out';
            } else {
                thisContext.globalCompositeOperation = 'source-over';
            }
            thisContext.strokeStyle = path.type === "brush" ? 'rgba(0, 0, 0, 1)' : 'rgba(0, 0, 0, 1)';
            thisContext.stroke();
            thisContext.beginPath();
            thisContext.arc(point.toX, point.toY, path.lineWidth / 2, 0, 2 * Math.PI);
            thisContext.fillStyle = path.type === "brush" ? 'rgba(0, 0, 0, 1)' : 'rgba(0, 0, 0, 1)';
            thisContext.fill();
            }
        }
    }
}
  
// stop funcs ---------------------------------------------------------------
function stopDrawing() {
    canvas.style.cursor = "auto";
    // stop drawing
    isDrawing = false;
    sliderWrapper.style.display = 'none';
}
function stopMagicDrawing() {
    canvas.style.cursor = "auto";
    // stop magic drawing
    try {
        canvas.removeEventListener("mousedown", magicToolHandler);
        console.log('remove magic tool handler')
    } catch (error) {
        console.log(error) // do nothing
    }
    isMagicToolActive = false;

    if (interactivePaths.length != 0) {
        const mask_idx = interactivePaths.length - 1;
        const lastPath = interactivePaths.pop();
        promptPoints.splice(0);
        interactivePaths.splice(0);

        // draw on scaled-size canvas
        drawnPaths.push({
            points: lastPath,
            type: "magic",
        });
        redrawPaths(canvas, drawnPaths);

        const formData = new FormData();
        formData.append("mask_idx", mask_idx);
        fetch("/finish_click", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                console.log("Success:", data);
            })
            .catch((error) => {
                console.error("Error:", error);
            })
            .finally(() => {
                // // Hide the spinner
                // spinnerContainer.classList.add("d-none");
            });
    }
}
function stopErasing() {
    canvas.style.cursor = "auto";
    // stop erasing
    isErasing = false;
}

function stopRecting() {
    canvas.style.cursor = "auto";
    // stop draw recting for face correction
    isDrawingRect = false;
  }