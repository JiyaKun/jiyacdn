// Main function to open the camera
function openCamera() {
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

// The robust change handler
async function handleCameraChange(event) {
    const file = event.target.files[0];
    if (!file) return; // user cancelled

    console.log("taken");
    bubble_fn_showProcessingDialog();

    const inputImage = document.getElementById("inputImage");
    const imgURL = URL.createObjectURL(file);
    inputImage.src = imgURL;

	const uploading = async () => {
		// Prepare file for upload
		const now = new Date();
		const timestamp = now.toISOString().replace(/:/g, "-");
		const newFileName = `selfie_owner_${timestamp}.jpg`;
		
		const renamedFile = new File([file], newFileName, { type: file.type });
		const formData = new FormData();
		formData.append("file", renamedFile);
		
		const uploadUrl = document.getElementById("skip-upload-url").value;
		const response = await fetch(uploadUrl, { method: "POST", body: formData });
		const result = await response.json();
		
		bubble_fn_photoTaken(result);
	}

    inputImage.onload = async () => {
        validateImage(inputImage, async (verdict) => {
            const messages = [];
            if (!verdict.hasFace) {
                messages.push(document.getElementById("skip-no-face-found").value);
            }

            if (messages.length === 0) {
                uploading()
            } else {
                alert(messages.join(" "));
            }

            bubble_fn_hideProcessingDialog();
        });
    };
}
