function hideFileInput() {
  fileInputWrapper.style.display = 'none';
}

function showFileInput() {
  fileInputWrapper.style.display = 'block';
}

async function canvasToBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
          resolve(blob);
      } else {
          reject(new Error("Canvas to Blob conversion failed"));
      }
    });
  });
}
  
async function blobToByteArray(blob) {
  return new Uint8Array(await new Response(blob).arrayBuffer());
}

// for processing image
function handleImage(file, isVideo = false) {
  // Extract and store the uploaded image name
  imageName = file.name.split('.')[0];
  console.log(imageName)

  let image = new Image();
  image.onload = async () => {
    const spinnerContainer = document.getElementById("spinner-container");
    // Show the spinner
    spinnerContainer.classList.remove("d-none");

    // first limit the height or width of image to be less than 768
    const max_size = isVideo ? 1024 : 768;
    let width = image.width;
    let height = image.height;
    if (width > max_size || height > max_size) {
      if (width > height) {
        height *= max_size / width;
        width = max_size;
      } else {
        width *= max_size / height;
        height = max_size;
      }
    }
    // make sure multiple of 8
    width = Math.floor(width / 8) * 8;
    height = Math.floor(height / 8) * 8;

    // for keeping original sized image
    const tmp_canvas = document.createElement('canvas');
    tmp_canvas.width = width;
    tmp_canvas.height = height;

    const ctx = tmp_canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, width, height);

    originalUnresizedImageData = ctx.getImageData(0, 0, width, height);

    // console.log(originalUnresizedImageData)
    // console.log(imageCanvas.width, imageCanvas.height)
    // console.log(image.width, image.height)
    const widthRatio = imageCanvas.width / width;
    const heightRatio = imageCanvas.height / height;
    scaleFactor = Math.min(widthRatio, heightRatio);

    scaledWidth = width * scaleFactor;
    scaledHeight = height * scaleFactor;
    scaledX = (imageCanvas.width - scaledWidth) / 2;
    scaledY = (imageCanvas.height - scaledHeight) / 2;

    // reset width and height can refresh the canvas so that prev image will not be kept
    imageCanvas.width = 500;
    imageCanvas.height = 500;
    imageContext.drawImage(tmp_canvas, scaledX, scaledY, scaledWidth, scaledHeight);
    originalImageDataBackup = imageContext.getImageData(0, 0, imageCanvas.width, imageCanvas.height);

    hideFileInput();

    const imageBlob = await canvasToBlob(tmp_canvas);
    const imageByteArray = await blobToByteArray(imageBlob);

    const formData = new FormData();
    formData.append("image", new Blob([imageByteArray]), "image.png");
    fetch("/uploadimage", {
        method: "POST",
        body: formData,
    })
    .then((response) => response.json())
    .then((data) => {
        console.log("Success:", data);
    })
    .catch((error) => {
        console.error("Error:", error);
    }).finally(() => {
      // Hide the spinner
      spinnerContainer.classList.add("d-none");
      console.log('finally')
    });
  };
  image.src = URL.createObjectURL(file);
}

fileInput.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  const fileType = file.type;

  const validVideoTypes = ['video/mp4', 'video/quicktime'];
    if (validVideoTypes.includes(fileType)) {
        console.log("not supported yet");
    } else {
        console.log(file);
        handleImage(file);
    }
});

removeImageButton.addEventListener('click', () => {
  imageContext.clearRect(0, 0, imageCanvas.width, imageCanvas.height); // clear the canvas
  context.clearRect(0, 0, canvas.width, canvas.height); // clear the canvas
  renderContext.clearRect(0, 0, renderCanvas.width, renderCanvas.height); // clear the canvas
  tempContext.clearRect(0, 0, tempCanvas.width, tempCanvas.height); // clear the canvas
  stopDrawing();
  stopErasing();
  stopMagicDrawing();
  stopRecting();
  
  while (drawnPaths.length > 0) {
    drawnPaths.pop();
  }
  showFileInput();
  // Reset the file input value, force the browser to treat the re-uploaded file as a new file
  fileInput.value = '';

});

slider.addEventListener("input", () => {
  brushSize = slider.value;
});