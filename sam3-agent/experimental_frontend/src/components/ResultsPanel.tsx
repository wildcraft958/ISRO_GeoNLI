import { useState } from 'react';
import { SegmentResponse } from '../utils/api';
import LoadingIndicator from './LoadingIndicator';

interface ResultsPanelProps {
  response: SegmentResponse | null;
  loading: boolean;
  loadingStage?: 'starting' | 'encoding_text' | 'encoding_images' | 'processing_tiles' | 'verification' | 'finalizing';
  loadingProgress?: number;
}

export default function ResultsPanel({ response, loading, loadingStage, loadingProgress }: ResultsPanelProps) {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['summary']));

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  if (loading) {
    return (
      <div className="results-panel">
        <h3>Results</h3>
        <LoadingIndicator stage={loadingStage} progress={loadingProgress} />
      </div>
    );
  }

  if (!response) {
    return (
      <div className="results-panel">
        <h3>Results</h3>
        <p className="no-results">No results yet. Upload an image and run segmentation.</p>
      </div>
    );
  }

  if (response.status === 'error') {
    return (
      <div className="results-panel">
        <h3>Results</h3>
        <div className="error-state">
          <h4>Error</h4>
          <p>{response.message || 'Unknown error occurred'}</p>
          {response.traceback && (
            <details>
              <summary>Traceback</summary>
              <pre>{response.traceback}</pre>
            </details>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="results-panel">
      <h3>Results</h3>

      <div className="result-section">
        <button
          className="section-header"
          onClick={() => toggleSection('summary')}
        >
          <span>Summary</span>
          <span>{expandedSections.has('summary') ? '▼' : '▶'}</span>
        </button>
        {expandedSections.has('summary') && (
          <div className="section-content">
            <p>{response.summary || 'No summary available'}</p>
            {response.llm_config && (
              <div className="llm-info">
                <p><strong>Model:</strong> {response.llm_config.model}</p>
                <p><strong>Provider:</strong> {response.llm_config.name || 'Unknown'}</p>
              </div>
            )}
          </div>
        )}
      </div>

      {response.regions && response.regions.length > 0 && (
        <div className="result-section">
          <button
            className="section-header"
            onClick={() => toggleSection('regions')}
          >
            <span>Regions ({response.regions.length})</span>
            <span>{expandedSections.has('regions') ? '▼' : '▶'}</span>
          </button>
          {expandedSections.has('regions') && (
            <div className="section-content">
              {response.regions.map((region, idx) => (
                <div key={idx} className="region-item">
                  <h5>Region {idx + 1}</h5>
                  {region.score !== undefined && (
                    <p><strong>Score:</strong> {region.score.toFixed(3)}</p>
                  )}
                  {region.bbox && (
                    <p><strong>BBox:</strong> [{region.bbox.map(v => v.toFixed(3)).join(', ')}]</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {response.debug_image_b64 && (
        <div className="result-section">
          <button
            className="section-header"
            onClick={() => toggleSection('debug-image')}
          >
            <span>Debug Image</span>
            <span>{expandedSections.has('debug-image') ? '▼' : '▶'}</span>
          </button>
          {expandedSections.has('debug-image') && (
            <div className="section-content">
              <img
                src={`data:image/png;base64,${response.debug_image_b64}`}
                alt="Debug visualization"
                className="debug-image"
              />
            </div>
          )}
        </div>
      )}

      {response.raw_sam3_json && (
        <div className="result-section">
          <button
            className="section-header"
            onClick={() => toggleSection('raw-json')}
          >
            <span>Raw SAM3 JSON</span>
            <span>{expandedSections.has('raw-json') ? '▼' : '▶'}</span>
          </button>
          {expandedSections.has('raw-json') && (
            <div className="section-content">
              <pre className="json-viewer">
                {JSON.stringify(response.raw_sam3_json, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

