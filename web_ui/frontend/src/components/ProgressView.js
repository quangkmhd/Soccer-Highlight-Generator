import React, { useEffect, useRef } from 'react';

const ProgressView = ({ logs, progress, stageName, status }) => {
    const logContainerRef = useRef(null);

    useEffect(() => {
        // Auto-scroll to the bottom of the log container
        if (logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [logs]);

    const getStatusText = () => {
        if (status === 'completed' || status === 'merged' || progress === 100) {
            return `✅ Pipeline Finished!`;
        }
        if (stageName) {
            return `🏃 Running: ${stageName}...`;
        }
        return `⏳ Initializing Pipeline...`;
    }

    return (
        <div className="progress-section">
            <div className="progress-info">
                <span>{getStatusText()}</span>
                <span>{progress}%</span>
            </div>
            <div className="progress-bar-container">
                <div className="progress-bar" style={{ width: `${progress}%` }}></div>
            </div>
            <div ref={logContainerRef} className="progress-view">
                {logs.map((log, index) => (
                    <div key={index} className="log-entry">
                        {log}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default ProgressView; 