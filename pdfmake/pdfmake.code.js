document.addEventListener("DOMContentLoaded", () => {
	pdfMake.vfs["NotoSans.ttf"] = notosansBase64

	pdfMake.fonts = {
		Roboto: { // Keep the original Roboto definition from vfs_fonts.js
	        normal: 'Roboto-Regular.ttf',
	        bold: 'Roboto-Medium.ttf',
	        italics: 'Roboto-Italic.ttf',
	        bolditalics: 'Roboto-MediumItalic.ttf'
	    },
		"NotoSans" : {
			"normal" : "NotoSans.ttf",
			"bold" : "NotoSans.ttf",
			"italics" : "NotoSans.ttf",
			"bolditalics" : "NotoSans.ttf"
		}
	}
})

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
        console.error('画像変換エラー:', error);
        throw error; // Re-throw to propagate the error
    }
}