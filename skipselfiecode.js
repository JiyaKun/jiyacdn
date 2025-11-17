function openCamera(){
	const container = document.getElementById("cameraInputContainer");

    // Remove old input if exists
    const oldInput = document.getElementById("cameraInput");
    if (oldInput) oldInput.remove();

    // Create a fresh input element
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.capture = "user"; // front camera
    input.id = "cameraInput";

    // Attach change listener
    input.addEventListener("change", handleCameraChange);

    // Append to container
    container.appendChild(input);

    // Trigger camera
    input.click();
}

async function handleCameraChange(event){
	const file = event.target.files[0];
    if (!file) return; // user cancelled
        
    if(typeof window.skipCameraPreExecute !== "undefined"){
    	window.skipCameraPreExecute()
    }
    
    const uploading = async () => {
        const formData = new FormData();

        const now = new Date();
        const timestamp = now.toISOString().replace(/:/g, "-"); // Replace colons to make it file-safe
        const newFileName = `selfie_${timestamp}.jpg`
        const renamedFile = new File([file], newFileName, {
  		    type: file.type
	    });
	        
        const uploadUrl = document.querySelector("#skip-upload-url").value;

        formData.append("file", renamedFile); 
        let response = await fetch(uploadUrl, {
    	    method: "POST",
    	    body: formData,
  	    });
  	    
        let result = await response.json();
        
        if(typeof window.skipCameraResult !== "undefined"){
        	window.skipCameraResult(result)
        }
    }
    
    const inputImage = document.querySelector("#inputImage");
    const imgURL = URL.createObjectURL(file);
    
    inputImage.src = imgURL;
    inputImage.onload = async () => {
        if(typeof window.skipValidateSelfie !== "undefined" && window.skipValidateSelfie){
        	validateImage(inputImage, (verdict) => {
	            const messages = typeof window.skipCollectVerdicts !== "undefined" ? window.skipCollectVerdicts(verdict) : [];
	
	            if(messages.length !== 0){
	                if(typeof window.skipCameraPostExecute !== "undefined"){
	                	window.skipCameraPostExecute(messages)
	                }
	            }else{
		            uploading()
	            } 
	        })
        }else{
        	uploading()
        }
    }
}