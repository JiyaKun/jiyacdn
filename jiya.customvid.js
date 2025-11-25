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
        
      video.addEventListener("canplay", () => {
          // The video dimensions should be reliably available here.
          const { videoWidth, videoHeight } = video;

          if (videoWidth > 0 && videoHeight > 0) {
             // console.log(`Weeee ${videoWidth} ${videoHeight}`)
             this.adjustVideoHeight(video, videoWidth, videoHeight);
          } else {
             // Fallback for extremely rare cases, or if other metadata is still pending.
             // This is usually unnecessary with 'canplay'.
             console.warn("videoWidth or videoHeight is still 0 after 'canplay'.");
             // You might still consider a short timeout as a final safeguard
             // or re-running the check with 'this.adjustVideoHeight' if the function 
             // handles the zero case internally.
         }
      }, { once: true });
        
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
		let src = this.getAttribute("src"); // Youtube, Tiktok and Instagram URL
		let postId = this.getAttribute("post-id");
		const aspectRatio = this.getAttribute("aspect") || "16:9"; // optional
		const frameId = this.getAttribute("frame-id");
			
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
			
			// Youtube
		if (videoId) {
		  src = `https://www.youtube.com/embed/${videoId}`;
		  this.renderYouTube(videoId, aspectRatio);
		  return;
		}
			
			// --- If TikTok ---
		if (src.includes("tiktok.com")) {
		  this.renderTikTok(src, postId, frameId);
		  return;
		}
		
		// --- Instagram ---
		if (src.includes("instagram.com")) {
		  this.renderInstagram(src, aspectRatio, frameId);
		  return;
		}
	
		this.innerHTML = `<p style="color:#880808;">Invalid Video URL (${src})</p>`;
		console.warn("Unsupported video source:", src);
	}
  
	renderYouTube(videoId, aspectRatio) {
		const src = `https://www.youtube.com/embed/${videoId}`;
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
		iframe.allow =
		  "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture";
		iframe.allowFullscreen = true;
			
		wrapper.appendChild(iframe);
		this.appendChild(wrapper);
	}
  
	async renderTikTok(src, postId, frameId) {
		// If it's a vt.tiktok.com redirect link, try to resolve it
		if (src.includes("vt.tiktok.com")) {
		  try {
		    const resolvedUrl = await this.resolveTiktokRedirect(`https://resolve-tiktok.chiakishinichi12.workers.dev/?url=${src}`);
		    if (resolvedUrl && resolvedUrl.includes("tiktok.com/@")) {
		      src = resolvedUrl;
		      this.cacheTiktok(resolvedUrl, postId)
		    } else {
		      console.warn("Unable to resolve TikTok redirect:", src);
		    }
		  } catch (e) {
		    console.warn("Redirect resolution failed:", e);
		  }
		}
		
		const videoId = this.extractTiktokId(src);
		
		const iframe = document.createElement("iframe");
		iframe.src = `https://www.tiktok.com/embed/v2/${videoId}`;
		iframe.style.position = "absolute";
		iframe.style.top = 0;
		iframe.style.left = 0;
		iframe.style.width = "100%";
		iframe.style.height = "100%";
		iframe.frameBorder = 0;
		iframe.allow = "clipboard-write; encrypted-media; picture-in-picture; web-share";
		iframe.allowFullscreen = true;
		
		this.appendChild(iframe);
		
		let searchContainer = setInterval(() => {
            const iframeContainer = document.getElementById(frameId);
            
            if (iframeContainer) {
                clearInterval(searchContainer)
                iframeContainer.style.minHeight = "780px";
                console.log(`Set min-height: ${iframeContainer.style.minHeight}`);
            }
        }, 700)
	}
	
	async resolveTiktokRedirect(url) {
		try {
		  const res = await fetch(url, { method: "GET" });
		
		  if (!res.ok) {
		    throw new Error(`HTTP ${res.status}`);
		  }
		
		  const data = await res.json(); // <--- parse the JSON payload
		  console.log("Resolved TikTok response:", data);
		
		  // The Worker returns { resolved_url: "..." }
		  return data.resolved_url || url;
		} catch (e) {
		  console.warn("Fetch redirect failed:", e);
		  return url; // fallback to original
		}
	}
  
	extractTiktokId(url) {
		// Try to extract /video/{id}
		const match = url.match(/\/video\/(\d+)/);
		return match ? match[1] : "";
	}
	
	cacheTiktok(resolvedUrl, postId){
		bubble_fn_cacheTiktok({
		    output1 : resolvedUrl,
		    output2: postId
		})
	}
  
	renderInstagram(src, aspectRatio, frameId) {
		const wrapper = document.createElement("div");
		wrapper.style.width = "100%";
		wrapper.style.height = 0;
		
	    const encodedUriString = encodeURIComponent(src);		
			
		const iframe = document.createElement("iframe");
		iframe.src = `https://jiyakun.github.io/jiyacdn/igembed.html?url=${encodedUriString}&frameId=${frameId}`;
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
		
        let searchContainer = setInterval(() => {
            const iframeContainer = document.getElementById(frameId);
            
            if (iframeContainer) {
                clearInterval(searchContainer)
                if (this.isMobile()) {
                    // Phone/tablet: smaller viewport
                    iframeContainer.style.minHeight = "640px"; 
                } else {
                    // Desktop: larger viewport
                    iframeContainer.style.minHeight = "960px"; 
                }
        
                console.log(`Set min-height: ${iframeContainer.style.minHeight}`);
            }
        }, 700)
	}


    isMobile() {
        return /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
    }
}

customElements.define("flex-vid", FlexVid);
