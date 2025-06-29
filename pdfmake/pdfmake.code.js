document.addEventListener("DOMContentLoaded", () => {
	pdfMake.vfs["NotoSans.ttf"] = notosansBase64;
	pdfMake.vfs["NotoSans-Bold.ttf"] = notosansBoldBase64;

	pdfMake.fonts = {
		Roboto: { // Keep the original Roboto definition from vfs_fonts.js
	        normal: 'Roboto-Regular.ttf',
	        bold: 'Roboto-Medium.ttf',
	        italics: 'Roboto-Italic.ttf',
	        bolditalics: 'Roboto-MediumItalic.ttf'
	    },
		"NotoSans" : {
			"normal" : "NotoSans.ttf",
			"bold" : "NotoSans-Bold.ttf",
			"italics" : "NotoSans.ttf",
			"bolditalics" : "NotoSans.ttf"
		}
	}
})

function rotateAndDisplayImage(imgUrl, callback) {
 	const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = function () {
        const shouldRotate = img.width > img.height;

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        if (shouldRotate) {
            canvas.width = img.height;
            canvas.height = img.width;

            ctx.translate(canvas.width / 2, canvas.height / 2);
            ctx.rotate(90 * Math.PI / 180);
            ctx.drawImage(img, -img.width / 2, -img.height / 2);
        } else {
            canvas.width = img.width;
            canvas.height = img.height;

            ctx.drawImage(img, 0, 0);
        }

        callback(canvas.toDataURL('image/jpeg'));
    };

    img.onerror = function () {
        console.error('Image load failed. Check URL and CORS policy.');
    };

    img.src = imgUrl;
}

async function rotateAndDisplayImage(imgUrl) {
	return new Promise((resolve, reject) => {
		const img = new Image();
	    img.crossOrigin = 'anonymous';
	
	    img.onload = function () {
	        const shouldRotate = img.width > img.height;
	
	        const canvas = document.createElement('canvas');
	        const ctx = canvas.getContext('2d');
	
	        if (shouldRotate) {
	            canvas.width = img.height;
	            canvas.height = img.width;
	
	            ctx.translate(canvas.width / 2, canvas.height / 2);
	            ctx.rotate(90 * Math.PI / 180);
	            ctx.drawImage(img, -img.width / 2, -img.height / 2);
	        } else {
	            canvas.width = img.width;
	            canvas.height = img.height;
	
	            ctx.drawImage(img, 0, 0);
	        }
	
			resolve(canvas.toDataURL("image/jpeg"));
	    };
	
	    img.onerror = function () {
	        reject(new Error('Image failed to load or CORS issue occurred.'));
	    };
	
	    img.src = imgUrl;
    });
}