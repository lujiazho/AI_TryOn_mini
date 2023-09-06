document.addEventListener("DOMContentLoaded", function () {
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

    const submitButton = document.getElementById("submit");
    // const saveButton = document.getElementById("save");
    // const sendXrayToInputButton = document.getElementById("sendXrayToInput");

    let renderImage = new Image();
    // Function to calculate the scale factor
    function calculateScaleFactor(image, canvas) {
        const scaleX = canvas.width / image.width;
        const scaleY = canvas.height / image.height;
        return Math.min(scaleX, scaleY);
    }

    submitButton.addEventListener("click", async function () {    
        stopMagicDrawing();
        stopDrawing();
        stopErasing();
        stopRecting();

        const spinnerContainer = document.getElementById("spinner-container");
        // Show the spinner
        spinnerContainer.classList.remove("d-none");

        // Collect the values of the components
        // const preprocessor = document.getElementById("preprocessor").value;
        // const guidance = document.getElementById("guidance").value;

        const tmp_canvas = document.createElement('canvas');
        tmp_canvas.width = originalUnresizedImageData.width;
        tmp_canvas.height = originalUnresizedImageData.height;

        const ctx = tmp_canvas.getContext('2d');
        ctx.putImageData(originalUnresizedImageData, 0, 0);

        // Save the mask image
        const mask_canvas = document.createElement('canvas');
        mask_canvas.width = originalUnresizedImageData.width;
        mask_canvas.height = originalUnresizedImageData.height;

        const mask_ctx = mask_canvas.getContext('2d');
        // this can resize the image automatically
        mask_ctx.drawImage(canvas, scaledX, scaledY, scaledWidth, scaledHeight, 
                                    0, 0, originalUnresizedImageData.width, originalUnresizedImageData.height);

        const imageBlob = await canvasToBlob(tmp_canvas);
        const maskBlob = await canvasToBlob(mask_canvas);

        // Convert the Blob objects to byte arrays
        const imageByteArray = await blobToByteArray(imageBlob);
        const maskByteArray = await blobToByteArray(maskBlob);

        // Create a FormData object to hold the data
        const formData = new FormData();
        // formData.append("preprocessor", preprocessor);
        // formData.append("weight", 1);
        formData.append("image", new Blob([imageByteArray]), "image.png");
        formData.append("mask", new Blob([maskByteArray]), "mask.png");
        // formData.append("resizeMode", "INNER_FIT");
        // formData.append("processorRes", 512);
        // formData.append("thresholdA", 0.4);
        // formData.append("thresholdB", 64);
        // formData.append("guidanceStart", 0);
        // formData.append("guidanceEnd", guidance);
        // formData.append("guidance", guidance);              

        // Send a POST request to the server API
        fetch("/controlnet/img2img", {
            method: "POST",
            body: formData,
        })
        .then((response) => response.json())
        .then((data) => {
            console.log("Success:", data);

            // Get the base64-encoded image strings from the JSON response
            const renderImageBase64 = data.render;

            // Create Image objects from the base64-encoded image strings
            renderImage.src = `data:image/png;base64,${renderImageBase64}`;

            // Draw the images on the canvases when they are loaded
            renderImage.onload = () => {
                renderContext.drawImage(renderImage, scaledX, scaledY, scaledWidth, scaledHeight);
            };
        })
        .catch((error) => {
            console.error("Error:", error);
        }).finally(() => {
            // Hide the spinner
            spinnerContainer.classList.add("d-none");
        });

        console.log("Submit button clicked");
    });

    // saveButton.addEventListener("click", async function () {
    //     // Your code to handle the save button click event
    //     console.log("Save button clicked");
    //     // upload the xray image
    //     // const xray_canvas = document.getElementById("xrayImage");

    //     // Create a new canvas element with the desired dimensions
    //     const new_canvas = document.createElement('canvas');
    //     new_canvas.width = xrayImage.width;
    //     new_canvas.height = xrayImage.height;

    //     // const scaleFactor = calculateScaleFactor(xrayImage, xray_canvas);
    //     // const scaledWidth = xrayImage.width * scaleFactor;
    //     // const scaledHeight = xrayImage.height * scaleFactor;

    //     // Draw the image on the new canvas
    //     const new_ctx = new_canvas.getContext('2d');
    //     // new_ctx.drawImage(xray_canvas, (xray_canvas.width - scaledWidth) / 2, (xray_canvas.height - scaledHeight) / 2, scaledWidth, scaledHeight, 0, 0, xrayImage.width, xrayImage.height);
    //     new_ctx.drawImage(xrayImage, 0, 0, xrayImage.width, xrayImage.height);

    //     const xrayImageBlob = await canvasToBlob(new_canvas);
    //     const xrayImageByteArray = await blobToByteArray(xrayImageBlob);

    //     const xrayImageFormData = new FormData();
    //     xrayImageFormData.append("xray", new Blob([xrayImageByteArray]), "xray.png");

    //     fetch("/controlnet/savepair", {
    //         method: "POST",
    //         body: xrayImageFormData,
    //     })
    //     .then((response) => response.json())
    //     .then((data) => {
    //         console.log("save xray Success:", data);
    //         // Show the Toast after a successful save
    //         var toastSaveSuccess = new bootstrap.Toast(document.getElementById('toastSaveSuccess'));
    //         toastSaveSuccess.show();
    //     })
    //     .catch((error) => {
    //         console.error("Error:", error);
    //     }).finally(() => {
    //         // // Hide the spinner
    //         // spinnerContainer.classList.add("d-none");
    //     });
    // });

    // sendXrayToInputButton.addEventListener("click", async function () {
    //     // Your code to handle the send xray to input button click event
    //     console.log("Send xray to input button clicked");
    //     // reset the original image
    //     // for keeping original sized image
    //     const tmp_canvas = document.createElement('canvas');
    //     tmp_canvas.width = xrayImage.width;
    //     tmp_canvas.height = xrayImage.height;

    //     const ctx = tmp_canvas.getContext('2d');
    //     ctx.drawImage(xrayImage, 0, 0, xrayImage.width, xrayImage.height);
    //     originalUnresizedImageData = ctx.getImageData(0, 0, xrayImage.width, xrayImage.height);
    //     // reset the image in the original canvas
    //     const scaleFactor = calculateScaleFactor(xrayImage, canvas);
    //     const scaledWidth = xrayImage.width * scaleFactor;
    //     const scaledHeight = xrayImage.height * scaleFactor;
    //     imageContext.drawImage(xrayImage, (canvas.width - scaledWidth) / 2, (canvas.height - scaledHeight) / 2, scaledWidth, scaledHeight);
    //     // clear all drawing
    //     context.clearRect(0, 0, canvas.width, canvas.height);
    //     // clear drawnPaths
    //     while (drawnPaths.length > 0) {
    //         drawnPaths.pop();
    //     }
    // });
});