// magic tool ----------------------------------------------------------------
function magicToolHandler(event) {
    if (!isMagicToolActive) return;
    console.log("magic tool handler")

    // step 1: get starting point
    const { x, y } = getEventLocationInOriImage(event);
    promptPoints.push({
        type: (event.which == 1) ? true : false,
        origPoint: [event.offsetX, event.offsetY],
        scaledPoint: [x, y],
    });

    // step 2: get image data for flood fill
    const imageWidth = imageCanvas.width;
    const imageHeight = imageCanvas.height;
    const imageData = imageContext.getImageData(0, 0, imageWidth, imageHeight);
    const pixelData = imageData.data;

    const formData = new FormData();
    const typeList = [];
    const clickList = [];

    for (const thisPrompt of promptPoints) {
        typeList.push(thisPrompt.type ? 1 : 0);
        clickList.push(thisPrompt.scaledPoint[0]);
        clickList.push(thisPrompt.scaledPoint[1]);
    }
    formData.append("type", typeList);
    formData.append("click_list", clickList);

    // Send a POST request to the server API
    fetch("/segmentation/click", {
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
          // Create a canvas element and draw the image onto it
          const tmpcanvas = document.createElement('canvas');
          tmpcanvas.width = canvas.width;
          tmpcanvas.height = canvas.height;
          const tmpcontext = tmpcanvas.getContext('2d');

          const scaleFactor = Math.min(tmpcanvas.width / maskImage.width, tmpcanvas.height / maskImage.height);
          const scaledWidth = maskImage.width * scaleFactor;
          const scaledHeight = maskImage.height * scaleFactor;

          const x = (canvas.width - scaledWidth) / 2;
          const y = (canvas.height - scaledHeight) / 2;

          tmpcontext.drawImage(maskImage, x, y, scaledWidth, scaledHeight);

          // Get the image data from the canvas
          const imageData = tmpcontext.getImageData(0, 0, tmpcanvas.width, tmpcanvas.height);
          const pixelData = imageData.data;

          console.log(pixelData.length)
          // Get the pixel indices of the mask
          for (let i = 0; i < pixelData.length; i += 4) {
            if (pixelData[i] == 255 && pixelData[i + 1] == 255 && pixelData[i + 2] == 255) {
              pixels.push(i);
            }
          }
          console.log(pixels.length)

          redrawPaths(canvas, drawnPaths);

          // step 4: put magic mask on canvas
          putDataOnCanvas(canvas, pixels);

          for (const thisPrompt of promptPoints) {
              drawPromptPointOnClick(thisPrompt, canvas);
          }

          // step 5: Add the magic mask to drawnPaths array
          interactivePaths.push(pixels);
        };
    })
    .catch((error) => {
        console.error("Error:", error);
    }).finally(() => {
        // // Hide the spinner
        // spinnerContainer.classList.add("d-none");
    });
  }

  magicToolButton.addEventListener("click", (event) => {
    stopDrawing();
    stopErasing();
    stopRecting();

    if (!isMagicToolActive) {
      canvas.style.cursor = "crosshair";
      canvas.addEventListener("mousedown", magicToolHandler);
      isMagicToolActive = true;
    } else {
      stopMagicDrawing();
    }
  });