import {
    HandLandmarker,
    FaceLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.js";

// const outputCanvas = document.getElementById("canvas");
// const canvasCtx = outputCanvas.getContext("2d");

let handLandmarker;
let faceLandmarker; 

// Load the HandLandmarker model (First)
async function createLandmarkers() {
    try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        handLandmarker = await HandLandmarker.createFromOptions(
            filesetResolver, {
                baseOptions: {
                    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                    delegate: "GPU" // Try "GPU" for better performance, fall back to "CPU" if issues
                },
                runningMode: "IMAGE", // Crucially changed to IMAGE mode
                numHands: 2 // Detect up to 2 hands
            }
        );
        
        faceLandmarker = await FaceLandmarker.createFromOptions( // Or FaceDetector
	        filesetResolver, {
	            baseOptions: {
	                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
	                delegate: "GPU"
	            },
	            runningMode: "IMAGE",
	            numFaces: 1
	        }
	    );
        
        console.log("Landmarker models loaded successfully!");
    } catch (error) {
        console.error("Failed to load Landmarker models:", error);
        loadingMessage.textContent = "Error loading model. Please check console.";
    }
}

createLandmarkers();

// Evaluating amount of image blur
async function validateImageQuality(inputImage, canvas) {
  cv = await cv;

  let verdictVariance = -1

  const ctx = canvas.getContext('2d');
  
  canvas.width = inputImage.width / 2;
  canvas.height = inputImage.height / 2;
  ctx.drawImage(inputImage, 0, 0);

  // Blur detection
  const checkBlur = () => {
    const src = cv.imread(canvas);
    let gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    let laplacian = new cv.Mat();
    cv.Laplacian(gray, laplacian, cv.CV_64F);
    let mean = new cv.Mat(), stddev = new cv.Mat();
    cv.meanStdDev(laplacian, mean, stddev);
    let variance = Math.pow(stddev.doubleAt(0, 0), 2);
    
    verdictVariance = variance;
    
    src.delete(); gray.delete(); laplacian.delete(); mean.delete(); stddev.delete();
  };

  if (typeof cv === 'undefined') {
    // Wait a bit if OpenCV hasn't loaded yet
    setTimeout(checkBlur, 1000);
  } else {
    checkBlur();
  }
  
  return verdictVariance;
}

// Detecting Hand and Face Gesture
async function predictImage(inputImage, canvas) {
    if (!handLandmarker) {
        // Model not loaded yet. Please wait.
        return -1;
    }
    if (!inputImage.src) {
		// "No image selected."
        return 0;
    }

    // Ensure the image has its dimensions loaded
    // This check is the most critical for the error you're seeing
    if (inputImage.naturalWidth === 0 || inputImage.naturalHeight === 0) {
        console.warn("Image dimensions not yet available. Retrying prediction soon...");
        // You might want to add a small delay and retry, or ensure onload handles it.
        // For static images, the `inputImage.onload` event listener above should prevent this.
        
        // Image not fully loaded. Please try again.
        return -2;
    }

	// when successfully passed the ifs, proceed on processing hand detection

    // Perform detection - MediaPipe should read dimensions directly from the <img> element
    const handDetections = handLandmarker.detect(inputImage);
    const faceDetections = faceLandmarker.detect(inputImage);

	const verdict = {
		hasHands : false,
		hasFace : false,
		variance : validateImageQuality(inputImage, canvas)
	};

    // Detecting hand gestures
    if (handDetections.landmarks && handDetections.landmarks.length > 0) { 
        verdict.hasHands = true;
    } 
    
    // Detecting face appearance
    if (faceDetections.faceLandmarks && faceDetections.faceLandmarks.length > 0) { // Or results.detections for FaceDetector
        verdict.hasFace = true;
    } 
    
    // finalizing verdict
    if(verdict.hasFace == true || verdict.hasHands == true || verdict.variance > 0){
    	return verdict;
    }else{
    	return false;
    }
}

window.predictImage = predictImage;

// Custom drawing function for connections (simplified from MediaPipe's drawing_utils)
function drawConnectors(ctx, landmarks) {
    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8], // Index
        [0, 9], [9, 10], [10, 11], [11, 12], // Middle
        [0, 13], [13, 14], [14, 15], [15, 16], // Ring
        [0, 17], [17, 18], [18, 19], [19, 20] // Pinky
    ];

    ctx.strokeStyle = "lightblue";
    ctx.lineWidth = 2;

    for (const connection of connections) {
        const start = landmarks[connection[0]];
        const end = landmarks[connection[1]];
        if (start && end) {
            ctx.beginPath();
            ctx.moveTo(start.x * outputCanvas.width, start.y * outputCanvas.height);
            ctx.lineTo(end.x * outputCanvas.width, end.y * outputCanvas.height);
            ctx.stroke();
        }
    }
}


// --- Part 2: Gesture Recognition Logic ---

// Landmark indices for reference:
// Thumb: 0 (wrist), 1, 2, 3, 4 (tip)
// Index: 0 (wrist), 5, 6, 7, 8 (tip)
// Middle: 0 (wrist), 9, 10, 11, 12 (tip)
// Ring: 0 (wrist), 13, 14, 15, 16 (tip)
// Pinky: 0 (wrist), 17, 18, 19, 20 (tip)

function recognizeGesture(landmarks, handedness) {
    // Helper to check if a finger is curled (bent)
    // We check if the tip is significantly below (y-coord greater) the knuckle,
    // or if the distance from metacarpal to tip is small.
    const isFingerCurled = (fingerTipIndex, knuckleIndex, secondKnuckleIndex) => {
        // Simple Y-coordinate check (works well for upright hand)
        // If the tip's Y is significantly greater than the knuckle's Y, it's curled downwards.
        if (landmarks[fingerTipIndex].y > landmarks[knuckleIndex].y + 0.05) { // 0.05 is a threshold, adjust as needed
             // More robust check: distance from base knuckle (MCP) to tip.
             // If this distance is significantly smaller than expected for an extended finger, it's curled.
             const distTipKnuckle = Math.sqrt(
                Math.pow(landmarks[fingerTipIndex].x - landmarks[knuckleIndex].x, 2) +
                Math.pow(landmarks[fingerTipIndex].y - landmarks[knuckleIndex].y, 2)
            );
            const distKnuckleSecond = Math.sqrt(
                Math.pow(landmarks[knuckleIndex].x - landmarks[secondKnuckleIndex].x, 2) +
                Math.pow(landmarks[knuckleIndex].y - landmarks[secondKnuckleIndex].y, 2)
            );

            // If tip is close to the second knuckle, it's probably curled
            return distTipKnuckle < distKnuckleSecond * 0.7; // Adjust multiplier
        }
        return false;
    };


    // Helper to check if a finger is extended (straight up or out)
    // We check if the tip is significantly above (y-coord smaller) the knuckle.
    const isFingerExtended = (fingerTipIndex, knuckleIndex, secondKnuckleIndex) => {
        // A simple check: tip's Y is significantly less than knuckle's Y (for upright hand)
        if (landmarks[fingerTipIndex].y < landmarks[knuckleIndex].y - 0.05) { // Threshold
            // Also ensure it's not significantly behind (x-coord) if it's very far down
            return true;
        }

        // More robust: ensure the tip is generally "above" or "outward" from its base knuckle
        // and that the segment from base to tip is relatively long and straight.
        // This is more complex and might involve angles or checking alignment of points.
        // For simplicity, we'll primarily rely on the Y-check for now, but a full solution
        // would involve vector math (dot product for angle, cross product for orientation).
        return false;
    };


    // --- Check for Peace Sign ---
    // Index and Middle fingers are extended.
    // Thumb, Ring, Pinky fingers are curled.
    const peaceSign =
        isFingerExtended(8, 5, 6) && // Index finger (tip 8, knuckles 5, 6)
        isFingerExtended(12, 9, 10) && // Middle finger (tip 12, knuckles 9, 10)
        !isFingerExtended(4, 1, 2) && // Thumb is bent/curled (tip 4, knuckles 1, 2)
        !isFingerExtended(16, 13, 14) && // Ring finger is bent/curled
        !isFingerExtended(20, 17, 18); // Pinky finger is bent/curled

    if (peaceSign) {
        // Additional check for peace sign: ensure index and middle are somewhat parallel/separated
        const indexTip = landmarks[8];
        const middleTip = landmarks[12];
        const distBetweenTips = Math.sqrt(
            Math.pow(indexTip.x - middleTip.x, 2) +
            Math.pow(indexTip.y - middleTip.y, 2)
        );

        // A very rough heuristic: distance between tips should be within a reasonable range
        // This depends on the hand's scale in the image, so using normalized coordinates is key.
        // A typical normalized distance between index and middle tips for a peace sign might be 0.05-0.15.
        // This needs careful tuning.
        if (distBetweenTips > 0.04 && distBetweenTips < 0.2) {
             return "Peace Sign ✌️";
        }
    }


    // --- Check for OK Sign ---
    // Thumb tip (4) and Index tip (8) are close, forming a circle.
    // Middle, Ring, Pinky fingers are extended.
    const okSign =
        isFingerExtended(12, 9, 10) && // Middle finger extended
        isFingerExtended(16, 13, 14) && // Ring finger extended
        isFingerExtended(20, 17, 18); // Pinky finger extended

    if (okSign) {
        const thumbTip = landmarks[4];
        const indexTip = landmarks[8];
        const indexMCP = landmarks[5]; // Metacarpal of index finger

        // Calculate distance between thumb tip and index tip
        const distThumbIndexTip = Math.sqrt(
            Math.pow(thumbTip.x - indexTip.x, 2) +
            Math.pow(thumbTip.y - indexTip.y, 2)
        );

        // A robust way to check "closeness" for OK sign: compare distance between thumb and index tips
        // to a reference length like the length of the index finger itself.
        // This makes it scale-independent.
        const indexFingerLength = Math.sqrt(
            Math.pow(indexTip.x - indexMCP.x, 2) +
            Math.pow(indexTip.y - indexMCP.y, 2)
        );

        // If the distance between thumb and index tips is small relative to index finger length,
        // it's likely they are touching or very close.
        if (distThumbIndexTip < indexFingerLength * 0.3) { // Adjust threshold (e.g., 0.3 or 0.4)
            return "OK Sign 👌";
        }
    }

    // Default if no specific gesture is recognized
    return "Unknown";
}