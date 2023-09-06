document.addEventListener('contextmenu', function(event) {
event.preventDefault(); // Prevent the default context menu from appearing (right-click menu in browsers)
});

// Magic tool functionality
const replacementColor = { r: 215, g: 187, b: 245 };

const fileInput = document.getElementById('file-input');
let imageName = "original_image";
// for drawer
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
// for image
const imageCanvas = document.getElementById('image-canvas');
const imageContext = imageCanvas.getContext('2d');
imageCanvas.style.zIndex = -1;
// for render
const renderCanvas = document.getElementById("xrayImage");
const renderContext = renderCanvas.getContext("2d");
// for rect real-time visualization
const tempCanvas = document.getElementById('temp-canvas');
const tempContext = tempCanvas.getContext('2d');
tempCanvas.style.zIndex = 1;

canvas.width = 500;
canvas.height = 500;
imageCanvas.width = 500;
imageCanvas.height = 500;
renderCanvas.width = 500;
renderCanvas.height = 500;
tempCanvas.width = 500;
tempCanvas.height = 500;
// for saving mask purpose
let scaledWidth = -1;
let scaledHeight = -1;
let scaledX = -1;
let scaledY = -1;
// global variables for image
let scaleFactor = -1;

let originalUnresizedImageData;
let originalImageDataBackup;

// Prompt points in magic tool
const promptPoints = [];
const interactivePaths = [];

// Add a new array to store drawn paths
const drawnPaths = [];
// for magic tool
let isMagicToolActive = false;
// for draw rect
let isDrawingRect = false;
let rectStartX = 0;
let rectStartY = 0;
// for eraser
let isErasing = false;
// for draw mask, keep track of whether user is currently drawing
let isDrawing = false;
let isDown = false;
let lastX, lastY; // keep track of last position of the pointer

const slider = document.getElementById("brush-size");
let brushSize = slider.value;

const removeImageButton = document.getElementById('remove-image');
const drawMaskButton = document.getElementById('draw-mask');
const magicToolButton = document.getElementById('magic-tool');
const fileInputWrapper = document.getElementById('file-input-wrapper');
const drawRectButton = document.getElementById('draw-rect'); // face correction