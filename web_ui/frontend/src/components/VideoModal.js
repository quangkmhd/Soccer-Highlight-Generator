import React, { useEffect } from 'react';
import './VideoModal.css';

const VideoModal = ({ videoUrl, onClose }) => {
    // Effect to handle the 'Escape' key press for closing the modal
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'Escape') {
                onClose();
            }
        };

        window.addEventListener('keydown', handleKeyDown);

        // Cleanup the event listener when the component unmounts
        return () => {
            window.removeEventListener('keydown', handleKeyDown);
        };
    }, [onClose]);

    if (!videoUrl) {
        return null;
    }

    return (
        <div className="video-modal-overlay" onClick={onClose}>
            <div className="video-modal-content" onClick={(e) => e.stopPropagation()}>
                <video
                    src={videoUrl}
                    controls
                    autoPlay
                    loop
                    className="video-modal-player"
                />
                <button className="video-modal-close-btn" onClick={onClose}>
                    &times;
                </button>
            </div>
        </div>
    );
};

export default VideoModal; 