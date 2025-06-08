async function validateImage(base64Image) {
  cv = await cv;
  const img = document.getElementById('inputImage');
  img.src = base64Image;
  await new Promise(resolve => img.onload = resolve);

  let verdict = {
  	posing : false,
  	variance : -1
  };

  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = img.width / 2;
  canvas.height = img.height / 2;
  ctx.drawImage(img, 0, 0);

  // Pose detection
  const pose = new Pose({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.4/${file}`
  });

  pose.setOptions({
    modelComplexity: 1,
    enableSegmentation: false,
    minDetectionConfidence: 0.3,
    minTrackingConfidence: 0.3
  });

  pose.onResults(results => {
    if (results.poseLandmarks) {
      verdict.posing = true;
      verdict.posingLandmarks = results.poseLandmarks;
    }
  });
  
  const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/${file}`
  });
  
  hands.setOptions({
    maxNumHands: 2,
    minDetectionConfidence: 0.3,
    minTrackingConfidence: 0.3,
  });
  
  hands.onResults(results => {
    if (results.multiHandLandmarks.length > 0) {
      const peaceSignDetected = detectPeaceSign(results.multiHandLandmarks[0]);
      if (peaceSignDetected) {
        verdict.peaceSign = true;
      }
    }
  });

  await pose.send({ image: canvas });

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
    
    verdict.variance = variance;
    
    src.delete(); gray.delete(); laplacian.delete(); mean.delete(); stddev.delete();
  };

  if (typeof cv === 'undefined') {
    // Wait a bit if OpenCV hasn't loaded yet
    setTimeout(checkBlur, 1000);
  } else {
    checkBlur();
  }
  
  return verdict;
}