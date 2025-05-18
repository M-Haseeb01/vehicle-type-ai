// Only these four categories:
const VEHICLE_TERMS = {
    car:       ['car','sedan','coupe','hatchback','suv','wagon','convertible','limousine', 'cab', 'jeep', 'minivan', 'police car', 'race car', 'sports car', 'taxi'],
    truck:     ['truck','pickup','lorry','semi','tanker','dump truck','tow truck', 'fire truck', 'garbage truck', 'trailer truck', 'van'],
    bus:       ['bus','minibus','coach','double-decker','school bus','trolleybus', 'passenger van', 'omnibus'],
    motorcycle:['motorcycle','motorbike','moped','scooter','dirt bike','chopper']
  };
  
  const canvas        = document.getElementById('imageCanvas');
  const ctx           = canvas.getContext('2d');
  const wrapper       = document.getElementById('canvasWrapper');
  const loading       = document.getElementById('loading');
  const dropMsg       = wrapper.querySelector('.drop-message');
  const fileInput     = document.getElementById('fileInput');
  const uploadBtn     = document.getElementById('uploadBtn');
  const clearBtn      = document.getElementById('clearBtn');
  const resultDiv     = document.getElementById('result');
  
  let model = null;
  let imageLoadedOnCanvas = false; // To track if an image is displayed
  
  // Show or hide spinner and drop message
  function showLoadingUI(isLoading) {
    loading.style.display = isLoading ? 'block' : 'none';
    if (isLoading) {
      dropMsg.style.display = 'none';
    } else {
      // Show drop message only if no image is on canvas
      dropMsg.style.display = imageLoadedOnCanvas ? 'none' : 'block';
    }
  }
  
  // Utility to fit and draw image
  function drawImage(img) {
    ctx.fillStyle = 'white';
    ctx.fillRect(0,0,canvas.width,canvas.height);
    const hR = canvas.width  / img.width;
    const vR = canvas.height / img.height;
    const r  = Math.min(hR, vR);
    const w  = img.width  * r;
    const h  = img.height * r;
    const xO = (canvas.width  - w)/2;
    const yO = (canvas.height - h)/2;
    ctx.drawImage(img, 0,0,img.width,img.height, xO,yO,w,h);
    imageLoadedOnCanvas = true;
    showLoadingUI(false); // Hide spinner, update dropMsg based on imageLoadedOnCanvas
    classify();
  }
  
  // Match a predicted label to our categories
  function matchCategory(label) {
    const primaryLabel = label.toLowerCase().split(',')[0].trim(); // Consider only primary label before comma
    for (const [cat, terms] of Object.entries(VEHICLE_TERMS)) {
      if (terms.some(t => primaryLabel.includes(t))) return cat;
    }
    return null;
  }
  
  // Classify canvas contents
  async function classify() {
    if (!model) {
      console.warn("Model not loaded yet. Classification aborted.");
      resultDiv.textContent = 'Model still loading or failed. Try again shortly.';
      return;
    }
    showLoadingUI(true);
    resultDiv.textContent = 'Analyzing…';
    try {
      const preds = await model.classify(canvas, 5); // Get top 5 predictions
      const match = preds.map(p => ({
        ...p,
        category: matchCategory(p.className)
      })).find(p => p.category);
  
      if (match) {
        const pct = (match.probability * 100).toFixed(1);
        resultDiv.textContent = `Detected: ${match.category.charAt(0).toUpperCase() + match.category.slice(1)} (${pct}%)`;
      } else {
        resultDiv.textContent = 'Not a Car/Truck/Bus/Motorcycle';
      }
    } catch (e) {
      console.error("Classification error:", e);
      resultDiv.textContent = 'Classification Error: ' + e.message;
    } finally {
      showLoadingUI(false);
    }
  }
  
  // Load model on start
  async function initializeModel() {
    if (typeof mobilenet === 'undefined') {
      console.error("MobileNet library (mobilenet) is not defined. Ensure the CDN script is loaded before this script.");
      resultDiv.textContent = 'Error: MobileNet library not found. Check console.';
      return;
    }
    try {
      model = await mobilenet.load({version: 2, alpha: 1.0}); // Specify version and alpha
      resultDiv.textContent = 'Model loaded. Upload an image.';
      console.log("MobileNet model loaded successfully.");
    } catch (err) {
      console.error("Model load failed:", err);
      resultDiv.textContent = 'Model load failed: ' + err.message;
    }
  }
  
  initializeModel(); // Call model initialization
  
  // File handling via button
  uploadBtn.addEventListener('click', () => fileInput.click());
  
  fileInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file || !file.type.match('image.*')) {
      resultDiv.textContent = 'Please select an image file.';
      return;
    }
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      drawImage(img);
      URL.revokeObjectURL(url);
    };
    img.onerror = () => {
      resultDiv.textContent = 'Error loading image.';
      URL.revokeObjectURL(url);
    };
    img.src = url;
    fileInput.value = '';
  });
  
  // Drag & drop
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evtName => {
    wrapper.addEventListener(evtName, e => {
      e.preventDefault();
      e.stopPropagation();
      if (evtName === 'dragenter' || evtName === 'dragover') {
        wrapper.classList.add('active');
      } else {
        wrapper.classList.remove('active');
      }
      if (evtName === 'drop') {
        const file = e.dataTransfer.files[0];
        if (file && file.type.match('image.*')) {
          const img = new Image();
          const url = URL.createObjectURL(file);
          img.onload = () => {
            drawImage(img);
            URL.revokeObjectURL(url);
          };
          img.onerror = () => {
            resultDiv.textContent = 'Error loading dropped image.';
            URL.revokeObjectURL(url);
          };
          img.src = url;
        } else {
          resultDiv.textContent = 'Drop a valid image file.';
        }
      }
    });
  });
  
  // Click on canvas wrapper to upload
  wrapper.addEventListener('click', (e) => {
      // Prevent click if user is clicking on buttons inside (though not applicable here)
      if (e.target === wrapper || e.target === canvas || e.target === dropMsg) {
          fileInput.click();
      }
  });
  
  
  // Clear canvas
  clearBtn.addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0,0,canvas.width,canvas.height);
    imageLoadedOnCanvas = false;
    resultDiv.textContent = model ? 'Model loaded. Upload an image.' : 'Initializing model...';
    showLoadingUI(false); // Ensure drop message is visible and spinner is off
  });
  
  // Initial canvas setup
  ctx.fillStyle = 'white';
  ctx.fillRect(0,0,canvas.width,canvas.height);
  showLoadingUI(false); // Set initial state for drop message

