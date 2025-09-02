function setupVideoValidation(inputId, maxDuration, maxSize) {
    // Use a small delay to ensure the Gradio element is in the DOM
    const fileInput = document.querySelector(`#${inputId} input[type="file"]`);
    if (!fileInput) {
        setTimeout(() => setupVideoValidation(inputId, maxDuration), 100);
        return;
    }

    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (!file) return;

        if (!file.type.startsWith('video/')) {
            return; // Only validate video files
        }

        // 1. Check file size
        if (file.size > maxSize) {
            alert(`File is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). The maximum allowed size is ${maxSize / 1024 / 1024} MB.`);
            event.target.value = ''; // Clear the invalid file
            return;
        }


        const video = document.createElement('video');
        video.preload = 'metadata';
        video.src = URL.createObjectURL(file);

        video.onloadedmetadata = function() {
            URL.revokeObjectURL(video.src);
            if (video.duration > maxDuration) {
                alert(`Video is too long (${Math.round(video.duration / 60)} minutes). The maximum allowed duration is ${maxDuration / 60} minutes.`);
                event.target.value = '';
            }
        };

        video.onerror = function() {
            console.error("Error loading video metadata.");
            alert("Could not read video metadata. The file might be corrupt or in an unsupported format.");
            event.target.value = '';
        };
    });
}

// The call to this function with the correct parameters will be injected from Python.
// e.g., setupVideoValidation('video-file-input', 3600);
