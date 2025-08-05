
import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import UploadForm from './components/UploadForm';
import ProgressView from './components/ProgressView';
import ClipGrid from './components/ClipGrid';
import MergedVideo from './components/MergedVideo';
import VideoModal from './components/VideoModal'; // Import the new component
import { uploadVideo, mergeClips } from './services/api';
// import logo from './assets/logo.png';  (No longer needed)

const App = () => {
    const [status, setStatus] = useState('idle'); // idle, uploading, processing, completed, merged, error
    const [videoName, setVideoName] = useState(null);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [logs, setLogs] = useState([]);
    const [clips, setClips] = useState([]);
    const [selectedClips, setSelectedClips] = useState(new Set());
    const [mergedVideoUrl, setMergedVideoUrl] = useState(null);
    const [mergedDownloadUrl, setMergedDownloadUrl] = useState(null); // New state for download link
    const [isMerging, setIsMerging] = useState(false);
    const [error, setError] = useState(null);
    // New state for structured progress
    const [pipelineProgress, setPipelineProgress] = useState(0);
    const [currentStageName, setCurrentStageName] = useState('');
    // New state for the modal
    const [modalVideoUrl, setModalVideoUrl] = useState(null);

    const ws = useRef(null);

    useEffect(() => {
        if (videoName && status === 'processing') {
            const wsUrl = `ws://${window.location.hostname}:8000/ws/run-inference/${videoName}`;
            ws.current = new WebSocket(wsUrl);
            setLogs(prev => [...prev, 'Attempting to connect to WebSocket...']);

            ws.current.onopen = () => {
                console.log("WebSocket connection established.");
                setLogs(prev => [...prev, '✅ WebSocket connection established. Starting pipeline...']);
            };

            ws.current.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log("WS Message:", data);

                if (data.log) {
                    setLogs(prev => [...prev, data.log]);
                }

                // Update progress and stage name if they exist in the message
                if (data.progress !== undefined) {
                    setPipelineProgress(data.progress);
                }
                if (data.stage_name) {
                    setCurrentStageName(data.stage_name);
                }

                if (data.status === 'completed') {
                    setClips(data.clips || []);
                    setStatus('completed');
                } else if (data.status === 'error') {
                    setError(data.message);
                    setStatus('error');
                }
            };

            ws.current.onerror = (err) => {
                console.error("WebSocket error:", err);
                const errorMsg = "WebSocket connection failed. Please ensure the backend server is running and accessible.";
                setError(errorMsg);
                setLogs(prev => [...prev, `❌ ${errorMsg}`]);
                setStatus('error');
            };

            ws.current.onclose = () => {
                console.log("WebSocket connection closed.");
                 if (status !== 'completed' && status !== 'error') {
                    setLogs(prev => [...prev, 'WebSocket connection closed.']);
                }
            };

            return () => ws.current?.close();
        }
    }, [videoName, status]);

    const handleUpload = async (file) => {
        setStatus('uploading');
        setError(null);
        setLogs([]);
        setClips([]);
        setMergedVideoUrl(null);
        setMergedDownloadUrl(null); // Reset download URL
        setSelectedClips(new Set());
        setModalVideoUrl(null); // Reset modal on new upload
        try {
            const response = await uploadVideo(file, setUploadProgress);
            setVideoName(response.video_name);
            setStatus('processing');
        } catch (err) {
            setError(err.message || "Failed to upload video.");
            setStatus('error');
        } finally {
            setUploadProgress(0);
        }
    };

    const handleMerge = async () => {
        if (selectedClips.size === 0) {
            setError("Please select at least one clip to merge.");
            return;
        }
        setIsMerging(true);
        setError(null);
        try {
            const response = await mergeClips(Array.from(selectedClips));
            // Set both URLs from the response
            setMergedVideoUrl(`http://${window.location.hostname}:8000${response.video_url}`);
            setMergedDownloadUrl(`http://${window.location.hostname}:8000${response.download_url}`);
            setStatus('merged');
        } catch (err) {
            setError(err.message || "Failed to merge clips.");
        } finally {
            setIsMerging(false);
        }
    };

    const handleReset = () => {
        setStatus('idle');
        setVideoName(null);
        setUploadProgress(0);
        setLogs([]);
        setClips([]);
        setSelectedClips(new Set());
        setMergedVideoUrl(null);
        setMergedDownloadUrl(null); // Reset download URL
        setIsMerging(false);
        setError(null);
        setPipelineProgress(0);
        setCurrentStageName('');
        setModalVideoUrl(null); // Reset modal
        ws.current?.close();
    };

  return (
    <div className="App">
      <header className="App-header">
                 <div className="header-title">
                    {/* <img src={logo} alt="Soccer Ball Logo" /> */}
                    <h1>Soccer Highlight Pipeline</h1>
                </div>
                <p>Upload a video to automatically generate highlight clips</p>
      </header>

            <main className="App-main">
                <div className="pipeline-steps">
                    {error && (
                        <div className="error-banner">
                            {error}
                            <button onClick={() => setError(null)}>X</button>
                        </div>
                    )}

                    {/* Step 1: Upload */}
                    <div className="step-card">
                        <h2>1. Upload Video</h2>
                        <UploadForm
                            onUpload={handleUpload}
                            isUploading={status === 'uploading'}
                            uploadProgress={uploadProgress}
                            disabled={status !== 'idle' && status !== 'error'}
                        />
                    </div>

                    {/* Step 2: Processing Log */}
                    {(status === 'processing' || status === 'completed' || status === 'merged') && (
                        <div className="step-card">
                            <h2>2. Pipeline Progress</h2>
                            <ProgressView 
                                logs={logs} 
                                progress={pipelineProgress}
                                stageName={currentStageName}
                                status={status}
                            />
                        </div>
                    )}
                    
                    {/* Step 3: Select Clips */}
                    {status === 'completed' && clips.length > 0 && (
                        <div className="step-card">
                             <h2>3. Select Clips & Merge</h2>
                            <ClipGrid
                                clips={clips}
                                selectedClips={selectedClips}
                                onClipSelect={(clipUrl) => {
                                    const newSelection = new Set(selectedClips);
                                    if (newSelection.has(clipUrl)) newSelection.delete(clipUrl);
                                    else newSelection.add(clipUrl);
                                    setSelectedClips(newSelection);
                                }}
                                onMerge={handleMerge}
                                isMerging={isMerging}
                                onPreviewClick={(clipUrl) => setModalVideoUrl(clipUrl)}
                            />
                        </div>
                    )}

                    {/* Step 4: Final Video */}
                    {status === 'merged' && mergedVideoUrl && (
                        <div className="step-card">
                            <h2>4. Your Highlight Reel</h2>
                            <MergedVideo 
                                videoSrc={mergedVideoUrl} 
                                downloadUrl={mergedDownloadUrl}
                            />
                        </div>
                    )}
                </div>
                
                {(status !== 'idle' || mergedVideoUrl) && (
                     <div className="reset-button-container">
                        <button onClick={handleReset} className="reset-button">
                            Start Over
                        </button>
                    </div>
                )}
            </main>

            <footer className="App-footer">
                <p>Nguyen Huu Quang</p>
            </footer>

            {/* Render the modal outside the main layout */}
            <VideoModal 
                videoUrl={modalVideoUrl} 
                onClose={() => setModalVideoUrl(null)} 
            />
        </div>
  );
};

export default App;
