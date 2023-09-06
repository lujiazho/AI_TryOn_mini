// save images ----------------------------------------------------------------
const saveImagesButton = document.getElementById('save-images');

saveImagesButton.addEventListener('click', () => {
    // stop everything
    stopDrawing();
    stopMagicDrawing();
    stopErasing();
    stopRecting();
    // Save the original sized image
    const tmp_canvas = document.createElement('canvas');
    tmp_canvas.width = originalUnresizedImageData.width;
    tmp_canvas.height = originalUnresizedImageData.height;

    const ctx = tmp_canvas.getContext('2d');
    ctx.putImageData(originalUnresizedImageData, 0, 0);
    const DataUrl = tmp_canvas.toDataURL('image/png');
    const ImageLink = document.createElement('a');
    ImageLink.href = DataUrl;
    ImageLink.download = imageName + '.png';
    ImageLink.click();

    // Save the mask image
    const mask_canvas = document.createElement('canvas');
    mask_canvas.width = originalUnresizedImageData.width;
    mask_canvas.height = originalUnresizedImageData.height;

    const mask_ctx = mask_canvas.getContext('2d');
    // this can resize the image automatically
    mask_ctx.drawImage(canvas, scaledX, scaledY, scaledWidth, scaledHeight, 
                                0, 0, originalUnresizedImageData.width, originalUnresizedImageData.height);
    // change to binary mask
    const maskdata = mask_ctx.getImageData(0, 0, mask_canvas.width, mask_canvas.height);

    const maskDataUrl = mask_canvas.toDataURL('image/png');
    const maskLink = document.createElement('a');
    maskLink.href = maskDataUrl;
    maskLink.download = imageName + '_mask.png';
    maskLink.click();

    // save the rendered image
    const rendered_canvas = document.createElement('canvas');
    rendered_canvas.width = originalUnresizedImageData.width;
    rendered_canvas.height = originalUnresizedImageData.height;

    const rendered_ctx = rendered_canvas.getContext('2d');
    rendered_ctx.drawImage(renderCanvas, scaledX, scaledY, scaledWidth, scaledHeight, 
        0, 0, originalUnresizedImageData.width, originalUnresizedImageData.height);
    const renderedDataUrl = rendered_canvas.toDataURL('image/png');
    const renderedLink = document.createElement('a');
    renderedLink.href = renderedDataUrl;
    renderedLink.download = imageName + '_rendered.png';
    renderedLink.click();
});