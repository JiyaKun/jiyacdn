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

async function getImageBase64FromUrl(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`画像の読み込みに失敗しました: ${response.statusText}`);
        }
        const blob = await response.blob();

        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                resolve(reader.result); // This is the Base64 Data URI
            };
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    } catch (error) {
    	alert(`画像の変換中にエラーが発生しました。\n\n詳細: ${error.message || error}`);
        console.error('画像変換エラー:', error);
        throw error; // Re-throw to propagate the error
    }
}