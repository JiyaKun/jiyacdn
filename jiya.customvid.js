class CustomVideo extends HTMLElement {
    connectedCallback() {
      const src = this.getAttribute("src");
      const video = document.createElement("video");
      const thumb = this.getAttribute("thumbnail");
      const loadingMessage = this.getAttribute("loading-message") || "Processing video...";
      video.setAttribute("controls", true);
      video.style.width = "100%";
      video.poster = thumb;
      this.appendChild(video);
    
      const player = dashjs.MediaPlayer().create();
      player.initialize(video, src, false);
        
      video.addEventListener("loadedmetadata", () => {
        let { videoWidth, videoHeight } = video;

          
        if (videoWidth === 0 || videoHeight === 0) {
            setTimeout(() => {
              videoWidth = video.videoWidth;
    			videoHeight = video.videoHeight;
              this.adjustVideoHeight(video, videoWidth, videoHeight)
            }, 700); // 500ms delay
        } else {
        	this.adjustVideoHeight(video, videoWidth, videoHeight)
        }
          
         
      });  
        
      // Catch playback or network errors (404, etc.)
      player.on(dashjs.MediaPlayer.events.ERROR, (e) => {
        console.warn("Playback error:", e);
        if (e.error && e.error.code === 404) {
          this.showThumbnail(thumb, loadingMessage)
        } else {
          this.showThumbnail(thumb, loadingMessage)
        }
      });
    }
        
    adjustVideoHeight(video, videoWidth, videoHeight){
    	// Detect if the video is portrait
        if (videoHeight > videoWidth) {
          video.style.height = "60vh"; // takes full viewport height (like Facebook)
          video.style.objectFit = "cover"; // crop slightly to fill
          video.style.borderRadius = "0"; // edge-to-edge look
        } else {
          video.style.height = "auto"; // Landscape video
          video.style.objectFit = "contain";  
        }
    }  
        
    showThumbnail(thumb, loadingMessage) { 
       this.innerHTML = `
          <div style="
            width: 100%;
            height: 300px;
            background: url('${thumb}') center/cover no-repeat;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.2em;
            border-radius: 8px;">
            <!-- Overlay -->
            <div style="
              position: absolute;
              top: 0;
              left: 0;
              width: 100%;
              height: 100%;
              background: rgba(0, 0, 0, 0.45); /* semi-transparent black */
            "></div>
            	<span style="
                  position: relative;
                  z-index: 1;
                  color: white;
				font-family: Poppins;
                  font-size: 18px;
                  font-weight: 500;
                  padding: 8px 16px;
                  background: rgba(0, 0, 0, 0.5);
                  border-radius: 6px;">${loadingMessage}</span>
          </div>
        `;
    }
}
    
customElements.define("custom-video", CustomVideo);

class FlexVid extends HTMLElement {
  connectedCallback() {
    let src = this.getAttribute("src"); // original YouTube URL
    const aspectRatio = this.getAttribute("aspect") || "16:9"; // optional

    // Extract VIDEO_ID from YouTube URLs
    let videoId = null;
    if (src.includes("youtube.com/watch")) {
      const urlParams = new URL(src).searchParams;
      videoId = urlParams.get("v");
    } else if (src.includes("youtube.com/shorts")) {
      videoId = src.split("/shorts/")[1].split("?")[0];
    } else if (src.includes("youtu.be")) {
      videoId = src.split("youtu.be/")[1].split("?")[0];
    }

    if (videoId) {
      src = `https://www.youtube.com/embed/${videoId}`;
    } else {
      console.warn("Could not parse YouTube video ID from URL:", src);
    }

    // Create responsive wrapper
    const wrapper = document.createElement("div");
    wrapper.style.position = "relative";
    wrapper.style.width = "100%";
    wrapper.style.paddingBottom = aspectRatio === "16:9" ? "56.25%" : "75%"; // 16:9 or 4:3
    wrapper.style.height = 0;

    const iframe = document.createElement("iframe");
    iframe.src = src;
    iframe.style.position = "absolute";
    iframe.style.top = 0;
    iframe.style.left = 0;
    iframe.style.width = "100%";
    iframe.style.height = "100%";
    iframe.frameBorder = 0;
    iframe.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture";
    iframe.allowFullscreen = true;

    wrapper.appendChild(iframe);
    this.appendChild(wrapper);
  }
}

customElements.define("flex-vid", FlexVid);