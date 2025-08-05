import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

export const uploadVideo = async (file, onUploadProgress) => {
    const formData = new FormData();
    formData.append('video', file);

    try {
        const response = await axios.post(`${API_URL}/upload-video`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
            onUploadProgress: (progressEvent) => {
                const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                onUploadProgress(percentCompleted);
            },
        });
        return response.data;
    } catch (error) {
        console.error("Error uploading video:", error.response?.data || error.message);
        throw new Error(error.response?.data?.detail || "Server error during upload.");
    }
};

export const mergeClips = async (clipUrls) => {
    const formData = new FormData();
    clipUrls.forEach(url => {
        formData.append('clips', url);
    });

    try {
        const response = await axios.post(`${API_URL}/merge-clips`, formData, {
             headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
        });
        return response.data;
    } catch (error) {
        console.error("Error merging clips:", error.response?.data || error.message);
        throw new Error(error.response?.data?.detail || "Server error during merge.");
    }
}; 