const verdict = {
	hasHands: false,
	hasFace: false
}

async function validateImage(image, verdictCallback){
	verdict.hasFace = false;
	verdict.hasHands = false;

	startFaceDetection(image, verdictCallback)
}

function startFaceDetection(image, verdictCallback) {
	const faceMesh = new FaceMesh({
	  locateFile: (file) =>
	    `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
	});
	
	faceMesh.setOptions({
	  maxNumFaces: 1,
	  refineLandmarks: false,
	  minDetectionConfidence: 0.5,
	  minTrackingConfidence: 0.5
	});
	
	faceMesh.onResults((results) => {
	  verdict.hasFace = results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0;
	  faceMesh.close(); // Cleanup FaceMesh before loading Hands
	  
	  startHandDetection(image, verdictCallback);
	});
	
	faceMesh.initialize().then(() => {
	  faceMesh.send({ image });
	});
	
	return verdict;
}

function startHandDetection(image, verdictCallback) {
	const handsScript = document.createElement('script');
	
	handsScript.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js';
	
	handsScript.onload = () => {
	  const hands = new Hands({
	    locateFile: (file) =>
	      `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
	  });
	
	  hands.setOptions({
	    maxNumHands: 2,
	    modelComplexity: 1,
	    minDetectionConfidence: 0.5,
	    minTrackingConfidence: 0.5
	  });
	
	  hands.onResults((results) => {
			verdict.hasHands = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;
	
			if(typeof verdictCallback !== "undefined"){
				verdictCallback(verdict)
			}
	
	    	hands.close(); // cleanup
	  });
	
	  hands.initialize().then(() => {
	    hands.send({ image });
	  });
	};
	
	document.body.appendChild(handsScript); // load dynamically to avoid conflict
}
