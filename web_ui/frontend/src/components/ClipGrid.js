import React from 'react';

const ClipGrid = ({ clips, selectedClips, onClipSelect, onMerge, isMerging, onPreviewClick }) => {
    const API_HOST = `http://${window.location.hostname}:8000`;

    if (!clips || clips.length === 0) {
        return <p>No clips were generated. Please check the logs for more details.</p>;
    }

    const handleSelectClick = (e, clipUrl) => {
        e.stopPropagation(); // Prevent the preview from opening when clicking the button
        onClipSelect(clipUrl);
    };

    return (
        <div className="clip-grid-container">
            <p>
                Click on a clip's preview to watch it. Click the checkmark to select/deselect it for the final reel.
            </p>
            <div className="clip-grid">
                {clips.map((clipUrl) => (
                    <div
                        key={clipUrl}
                        className={`clip-card ${selectedClips.has(clipUrl) ? 'selected' : ''}`}
                        onClick={() => onPreviewClick(`${API_HOST}${clipUrl}`)} // Click the card to preview
                    >
                        <video
                            src={`${API_HOST}${clipUrl}`}
                            muted
                            onMouseOver={e => e.currentTarget.play().catch(err=>console.log("Autoplay blocked"))}
                            onMouseOut={e => e.currentTarget.pause()}
                            loop
                            title="Click to watch full screen"
                        />
                        <button 
                            className="select-button" 
                            onClick={(e) => handleSelectClick(e, clipUrl)}
                            title={selectedClips.has(clipUrl) ? "Deselect clip" : "Select clip"}
                        >
                            <span className="checkmark">{selectedClips.has(clipUrl) ? '✔' : '+'}</span>
                        </button>
                    </div>
                ))}
            </div>
            <div className="clip-grid-controls">
                <span>{selectedClips.size} clip(s) selected.</span>
                <button onClick={onMerge} disabled={isMerging || selectedClips.size === 0}>
                    {isMerging ? 'Merging...' : `Merge ${selectedClips.size} Selected Clips`}
                </button>
            </div>
        </div>
    );
};

export default ClipGrid; 