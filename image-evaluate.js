async function validateImage(base64Image) {
  cv = await cv;
  const img = document.getElementById('inputImage');
  img.src = base64Image;
  await new Promise(resolve => img.onload = resolve);

  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);

  // Pose detection
  const pose = new Pose({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.4/${file}`
  });

  pose.setOptions({
    modelComplexity: 0,
    enableSegmentation: false,
    minDetectionConfidence: 0.3,
    minTrackingConfidence: 0.3
  });

  pose.onResults(results => {
    if (results.poseLandmarks) {
      console.log("Pose OK", results.poseLandmarks);
    } else {
      console.log("❌ Pose not detected");
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
    console.log("Blur variance:", variance);
    alert(variance < 100 ? "❌ Blurry" : "✅ Sharp");
    src.delete(); gray.delete(); laplacian.delete(); mean.delete(); stddev.delete();
  };

  if (typeof cv === 'undefined') {
    // Wait a bit if OpenCV hasn't loaded yet
    setTimeout(checkBlur, 1000);
  } else {
    checkBlur();
  }
}