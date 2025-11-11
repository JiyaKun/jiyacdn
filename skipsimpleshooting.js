document.getElementById('cameraInput').addEventListener('change', function(event) {
	event.preventDefault();
	console.log("taken");
	bubble_fn_showProcessingDialog()
	const file = event.target.files[0];
	
	const uploading = async () => {
	    const formData = new FormData();
		
	    const now = new Date();
	    const timestamp = now.toISOString().replace(/:/g, "-"); // Replace colons to make it file-safe
	    const newFileName = `selfie_owner_${timestamp}.jpg`
	    
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
	
	   bubble_fn_photoTaken(result)
	}
	
	const inputImage = document.getElementById("inputImage");
	const imgURL = URL.createObjectURL(file);
	inputImage.src = imgURL;
          
    inputImage.onload = () => {            
		validateImage(inputImage, async (verdict) => {
			const messages = [];

			if(!verdict.hasFace){
			    messages.push(document.querySelector("#skip-no-face-found").value);
			}

			if(messages.length === 0){
			    uploading()
			}else{
			    alert(messages.join(" "))
				bubble_fn_hideProcessingDialog()
			}
		})
    };
});
