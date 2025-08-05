import React from 'react';

const MergedVideo = ({ videoSrc, downloadUrl }) => {
    if (!videoSrc) return null;

    return (
        <div className="merged-video-container">
            <p>Your highlight reel is ready! You can watch it below or download it.</p>
            <video src={videoSrc} controls autoPlay muted loop />
            <a href={downloadUrl} download="highlight_reel.mp4" className="download-button">
                 <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width="24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                </svg>
                Download Highlights
            </a>
        </div>
    );
};

export default MergedVideo; 