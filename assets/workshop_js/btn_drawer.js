// drawer ----------------------------------------------------------------
const sliderWrapper = document.getElementById('slider-wrapper');
sliderWrapper.style.display = 'none';

drawMaskButton.addEventListener('click', (event) => {
  stopMagicDrawing();
  stopErasing();
  stopRecting();

  // toggle slider visibility
  sliderWrapper.style.display = (sliderWrapper.style.display == 'none') ? 'block' : 'none';
  // stop drawing
  if (isDrawing) {
    canvas.style.cursor = 'auto';
    isDrawing = false;
  }
  // start drawing
  else {
    canvas.style.cursor = `crosshair`; // change cursor to a crosshair
    isDrawing = true; 
  }
});

canvas.addEventListener('mousedown', (event) => {
  if (isDrawing || isErasing) {
    lastX = event.offsetX;
    lastY = event.offsetY;
    isDown = true;

    // Start a new path
    drawnPaths.push({
      type: isErasing ? 'eraser' : 'brush',
      points: [],
      lineWidth: brushSize,
    });
  } else if (isDrawingRect) {
    rectStartX = event.offsetX;
    rectStartY = event.offsetY;
    isDown = true;
  }
});

canvas.addEventListener('mouseup', (event) => {
  if (isDrawingRect && isDown) {
    const spinnerContainer = document.getElementById("spinner-container");
    // Show the spinner
    spinnerContainer.classList.remove("d-none");

    // Clear the temporary canvas
    tempContext.clearRect(0, 0, tempCanvas.width, tempCanvas.height);

    // context.beginPath();
    // context.rect(rectStartX, rectStartY, rectEndX - rectStartX, rectEndY - rectStartY);
    // context.strokeStyle = 'rgba(147, 112, 219, 1)';
    // context.lineWidth = 2;
    // context.stroke();

    // get start and end points of the rect
    const {x: startX, y: startY} = getXYLocationInOriImage(rectStartX, rectStartY);
    const {x: endX, y: endY} = getXYLocationInOriImage(event.offsetX, event.offsetY);
    console.log(rectStartX, rectStartY, event.offsetX, event.offsetY);
    console.log(startX, startY, endX, endY);
    const formData = new FormData();
    formData.append("start_x", startX);
    formData.append("start_y", startY);
    formData.append("end_x", endX);
    formData.append("end_y", endY);
    // Send a POST request to the server API
    fetch("/rect", {
        method: "POST",
        body: formData,
    })
    .then((response) => response.json())
    .then((data) => {
        // console.log("Success:", data.masks);
        let pixels = [];

        // Get the base64-encoded image strings from the JSON response
        const maskBase64 = data.masks;
        const maskImage = new Image();
        maskImage.src = `data:image/png;base64,${maskBase64}`;
        maskImage.onload = function() {
          renderContext.drawImage(maskImage, rectStartX, rectStartY, event.offsetX - rectStartX, event.offsetY - rectStartY);

          // // Create a original size element and draw the image onto it
          // let tmpcanvas = document.createElement('canvas');
          // tmpcanvas.width = canvas.width;
          // tmpcanvas.height = canvas.height;
          // let tmpcontext = tmpcanvas.getContext('2d');

          // tmpcontext.drawImage(maskImage, scaledX, scaledY, scaledWidth, scaledHeight);

          // // Get the image data from the canvas
          // let imageData = tmpcontext.getImageData(0, 0, tmpcanvas.width, tmpcanvas.height);
          // let pixelData = imageData.data;

          // // console.log(pixelData.length)
          // // let uniquePixelData = [...new Set(pixelData)];
          // // console.log(uniquePixelData);

          // // Get the pixel indices of the mask
          // for (let i = 0; i < pixelData.length; i += 4) {
          //   if (pixelData[i] == 255 && pixelData[i + 1] == 255 && pixelData[i + 2] == 255) {
          //     pixels.push(i);
          //   }
          // }
          // console.log(pixels.length)
          // // step 4: put magic mask on canvas
          // const canvasData = context.getImageData(0, 0, canvas.width, canvas.height);
          // const data = canvasData.data;
          // console.log(data.length)
          // for (let i = 0; i < pixels.length; i += 1) {
          //   data[pixels[i]] = replacementColor.r; // red
          //   data[pixels[i] + 1] = replacementColor.g; // green
          //   data[pixels[i] + 2] = replacementColor.b; // blue
          //   data[pixels[i] + 3] = 255; // alpha
          // }
          // context.putImageData(canvasData, 0, 0);

          // // step 5: Add the rect mask to drawnPaths array
          // drawnPaths.push({
          //   points: pixels,
          //   type: "rect",
          // });
        };
    })
    .catch((error) => {
        console.error("Error:", error);
    }).finally(() => {
        // Hide the spinner
        spinnerContainer.classList.add("d-none");
    });
  }
  isDown = false;
});

canvas.addEventListener('mousemove', (event) => {
  if ((isDrawing || isErasing) && isDown) {
    const x = event.offsetX;
    const y = event.offsetY;
    context.beginPath();
    context.moveTo(lastX, lastY);
    context.lineTo(x, y);
    if (isErasing) {
      context.globalCompositeOperation = 'destination-out';
    } else {
      context.globalCompositeOperation = 'source-over';
    }
    context.strokeStyle = isDrawing ? 'rgba(0, 0, 0, 1)' : 'rgba(0, 0, 0, 1)';
    context.lineWidth = brushSize;
    context.stroke();
    context.beginPath();
    context.arc(x, y, brushSize / 2, 0, 2 * Math.PI);
    context.fillStyle = isDrawing ? 'rgba(0, 0, 0, 1)' : 'rgba(0, 0, 0, 1)';
    context.fill();

    // Add the point to the current path
    const currentPath = drawnPaths[drawnPaths.length - 1];
    currentPath.points.push({ fromX: lastX, fromY: lastY, toX: x, toY: y });
    
    lastX = x;
    lastY = y;
  } else if (isDrawingRect && isDown) {
    console.log('drawing rect', rectStartX, rectStartY, event.offsetX, event.offsetY);
    const x = event.offsetX;
    const y = event.offsetY;

    // Clear the temporary canvas
    tempContext.clearRect(0, 0, tempCanvas.width, tempCanvas.height);

    // Draw the rectangle on the temporary canvas
    tempContext.beginPath();
    tempContext.rect(rectStartX, rectStartY, x - rectStartX, y - rectStartY);
    tempContext.strokeStyle = `rgba(${replacementColor.r}, ${replacementColor.g}, ${replacementColor.b}, 1)`;
    tempContext.lineWidth = 2;
    tempContext.stroke();
  }
});