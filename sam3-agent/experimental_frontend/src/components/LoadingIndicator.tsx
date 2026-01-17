import { useEffect, useState } from 'react';

interface LoadingIndicatorProps {
    stage?: 'starting' | 'encoding_text' | 'encoding_images' | 'processing_tiles' | 'verification' | 'finalizing';
    progress?: number; // 0-100
    message?: string;
}

export default function LoadingIndicator({ stage = 'starting', progress, message }: LoadingIndicatorProps) {
    const [dots, setDots] = useState('');

    // Animated dots for loading effect
    useEffect(() => {
        const interval = setInterval(() => {
            setDots(prev => (prev.length >= 3 ? '' : prev + '.'));
        }, 500);
        return () => clearInterval(interval);
    }, []);

    const getStageInfo = () => {
        switch (stage) {
            case 'encoding_text':
                return {
                    icon: '📝',
                    label: 'Encoding Text',
                    description: 'Processing prompt with language model'
                };
            case 'encoding_images':
                return {
                    icon: '🖼️',
                    label: 'Encoding Images',
                    description: 'Batch processing image tiles (16 at a time)'
                };
            case 'processing_tiles':
                return {
                    icon: '🔄',
                    label: 'Processing Tiles',
                    description: 'Running SAM3 inference on tiles'
                };
            case 'verification':
                return {
                    icon: '✓',
                    label: 'Verifying Detections',
                    description: 'VLM verification in progress'
                };
            case 'finalizing':
                return {
                    icon: '🎯',
                    label: 'Finalizing',
                    description: 'Combining results and applying NMS'
                };
            default:
                return {
                    icon: '⏳',
                    label: 'Starting',
                    description: 'Initiating batch processing'
                };
        }
    };

    const stageInfo = getStageInfo();

    return (
        <div className="loading-indicator">
            <div className="loading-header">
                <span className="loading-icon">{stageInfo.icon}</span>
                <h4>{stageInfo.label}{dots}</h4>
            </div>

            <p className="loading-description">{message || stageInfo.description}</p>

            {progress !== undefined && (
                <div className="progress-bar-container">
                    <div className="progress-bar" style={{ width: `${progress}%` }}>
                        <span className="progress-text">{Math.round(progress)}%</span>
                    </div>
                </div>
            )}

            <div className="loading-spinner">
                <div className="spinner-ring"></div>
            </div>

            <style>{`
        .loading-indicator {
          text-align: center;
          padding: 2rem 1rem;
          background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
          border-radius: 12px;
          border: 1px solid #334155;
        }

        .loading-header {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.75rem;
          margin-bottom: 1rem;
        }

        .loading-icon {
          font-size: 2rem;
          animation: pulse 2s ease-in-out infinite;
        }

        .loading-header h4 {
          margin: 0;
          font-size: 1.25rem;
          color: #60a5fa;
          font-weight: 600;
        }

        .loading-description {
          color: #94a3b8;
          font-size: 0.875rem;
          margin-bottom: 1.5rem;
        }

        .progress-bar-container {
          width: 100%;
          height: 24px;
          background: #1e293b;
          border-radius: 12px;
          overflow: hidden;
          margin-bottom: 1.5rem;
          border: 1px solid #334155;
        }

        .progress-bar {
          height: 100%;
          background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%);
          transition: width 0.3s ease-in-out;
          display: flex;
          align-items: center;
          justify-content: center;
          position: relative;
        }

        .progress-text {
          color: white;
          font-weight: 600;
          font-size: 0.75rem;
          text-shadow: 0 1px 2px rgba(0,0,0,0.5);
        }

        .loading-spinner {
          display: flex;
          justify-content: center;
          margin-top: 1rem;
        }

        .spinner-ring {
          width: 40px;
          height: 40px;
          border: 3px solid #334155;
          border-top-color: #60a5fa;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.7; transform: scale(1.1); }
        }
      `}</style>
        </div>
    );
}
