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
        
      if (video.canPlayType("application/vnd.apple.mpegurl")) {  
      	video.src = src;
        video.load();
      } else if (Hls.isSupported()){
        const hls = new Hls();
       	hls.loadSource(src);
      	hls.attachMedia(video);

      	hls.on(Hls.Events.ERROR, (event, data) => {
        	console.warn("HLS error:", data);
        	this.showThumbnail(thumb, loadingMessage);
      	});
      } else{
        console.error("HLS not supported on this browser");
      	this.showThumbnail(thumb, loadingMessage);
      }
        
      video.addEventListener("loadedmetadata", () => {
        let { videoWidth, videoHeight } = video;

        if (videoWidth === 0 || videoHeight === 0) {
            setTimeout(() => {
              videoWidth = video.videoWidth;
    		  videoHeight = video.videoHeight;
              this.adjustVideoHeight(video, videoWidth, videoHeight)
            }, 750);
        } else {
        	this.adjustVideoHeight(video, videoWidth, videoHeight)
        }         
      });  
        
      video.addEventListener("error", (e) => {
	      const error = e.target.error;
	      console.warn("Video Error:", error);
	
	      // MEDIA_ERR_SRC_NOT_SUPPORTED → 4 (video still being processed or inaccessible)
	      if (error && error.code === MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED) {
	        this.showThumbnail(thumb, loadingMessage);
	      } else {
	        this.showThumbnail(thumb, "Video unavailable or still processing...");
	      }
	  });  
    }
        
    adjustVideoHeight(video, videoWidth, videoHeight){
    	const isDesktop = window.innerWidth > 768; // adjust breakpoint as needed

		if (videoHeight > videoWidth) {
		  if (isDesktop) {
		    // For desktop, don’t crop, use auto height
		    video.style.height = "auto";
		    video.style.objectFit = "contain";
		    video.style.maxHeight = "80vh"; // optional, prevent too large
		  } else {
		    // Mobile
		    video.style.height = "60vh";
		    video.style.objectFit = "cover";
		    video.style.borderRadius = "0";
		  }
		} else {
		  video.style.height = "auto";
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
    let src = this.getAttribute("src"); // original URL
    const aspectRatio = this.getAttribute("aspect") || "16:9"; // optional

    // Only allow YouTube URLs
    const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\//;
    if (!youtubeRegex.test(src)) {
      console.warn("Blocked non-YouTube URL:", src);
      this.innerHTML = `<p style="color:red;">Invalid video URL</p>`;
      return;
    }

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

    if (!videoId) {
      console.warn("Could not parse YouTube video ID from URL:", src);
      this.innerHTML = `<p style="color:red;">Invalid YouTube URL</p>`;
      return;
    }

    src = `https://www.youtube.com/embed/${videoId}`;

    // Create responsive wrapper
    const wrapper = document.createElement("div");
    wrapper.style.position = "relative";
    wrapper.style.width = "100%";
    wrapper.style.paddingBottom = aspectRatio === "16:9" ? "56.25%" : "75%"; 
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
