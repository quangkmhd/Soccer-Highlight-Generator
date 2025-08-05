import React, { useState } from 'react';

const UploadForm = ({ onUpload, isUploading, uploadProgress, disabled }) => {
    const [selectedFile, setSelectedFile] = useState(null);

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setSelectedFile(file);
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (selectedFile) {
            onUpload(selectedFile);
        }
    };

    return (
        <form onSubmit={handleSubmit} className="upload-form">
            <div className="file-input-wrapper">
                <label htmlFor="file-upload" className="file-input-label">
                    Choose a video file
                </label>
                <input
                    id="file-upload"
                    type="file"
                    accept="video/mp4,video/x-m4v,video/*"
                    onChange={handleFileChange}
                    disabled={disabled || isUploading}
                />
                {selectedFile && <span className="file-name">Selected: {selectedFile.name}</span>}
            </div>
            
            <button type="submit" disabled={!selectedFile || disabled || isUploading}>
                {isUploading ? `Uploading...` : 'Start Highlight Pipeline'}
            </button>

            {isUploading && (
                <div className="progress-bar-container">
                    <div className="progress-bar" style={{ width: `${uploadProgress}%` }}>
                        {uploadProgress}%
                    </div>
                </div>
            )}
        </form>
    );
};

export default UploadForm; 