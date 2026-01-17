import { useState } from 'react';
import { SegmentResponse } from '../utils/api';

interface CommunicationLogProps {
  response: SegmentResponse | null;
}

export default function CommunicationLog({ response }: CommunicationLogProps) {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  if (!response || response.status !== 'success' || !response.raw_sam3_json) {
    return (
      <div className="communication-log">
        <h3>Internal Data</h3>
        <p className="no-data">No internal data available yet.</p>
      </div>
    );
  }

  const rawData = response.raw_sam3_json;

  return (
    <div className="communication-log">
      <h3>Internal Data</h3>

      <div className="log-section">
        <button
          className="section-header"
          onClick={() => toggleSection('dimensions')}
        >
          <span>Image Dimensions</span>
          <span>{expandedSections.has('dimensions') ? '▼' : '▶'}</span>
        </button>
        {expandedSections.has('dimensions') && (
          <div className="section-content">
            <p><strong>Width:</strong> {rawData.orig_img_w}px</p>
            <p><strong>Height:</strong> {rawData.orig_img_h}px</p>
          </div>
        )}
      </div>

      {rawData.pred_boxes && rawData.pred_boxes.length > 0 && (
        <div className="log-section">
          <button
            className="section-header"
            onClick={() => toggleSection('boxes')}
          >
            <span>Bounding Boxes ({rawData.pred_boxes.length})</span>
            <span>{expandedSections.has('boxes') ? '▼' : '▶'}</span>
          </button>
          {expandedSections.has('boxes') && (
            <div className="section-content">
              <p className="info-text">Normalized coordinates [x, y, w, h] in range [0, 1]</p>
              {rawData.pred_boxes.map((box, idx) => (
                <div key={idx} className="data-item">
                  <strong>Box {idx + 1}:</strong> [
                  {box.map((v, i) => (
                    <span key={i}>
                      {v.toFixed(4)}
                      {i < box.length - 1 ? ', ' : ''}
                    </span>
                  ))}
                  ]
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {rawData.pred_masks && rawData.pred_masks.length > 0 && (
        <div className="log-section">
          <button
            className="section-header"
            onClick={() => toggleSection('masks')}
          >
            <span>Masks ({rawData.pred_masks.length})</span>
            <span>{expandedSections.has('masks') ? '▼' : '▶'}</span>
          </button>
          {expandedSections.has('masks') && (
            <div className="section-content">
              <p className="info-text">RLE (Run-Length Encoded) format</p>
              {rawData.pred_masks.map((mask, idx) => (
                <div key={idx} className="data-item">
                  <strong>Mask {idx + 1}:</strong>
                  <p className="mask-info">
                    Size: [{mask.size.join(', ')}]
                  </p>
                  <p className="mask-info">
                    Counts type: {typeof mask.counts === 'string' ? 'String' : 'Array'}
                    {typeof mask.counts === 'string' && mask.counts.length > 100
                      ? ` (${mask.counts.length} chars, truncated)`
                      : ''}
                  </p>
                  {typeof mask.counts === 'string' && mask.counts.length <= 200 && (
                    <pre className="mask-preview">{mask.counts}</pre>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {rawData.pred_scores && rawData.pred_scores.length > 0 && (
        <div className="log-section">
          <button
            className="section-header"
            onClick={() => toggleSection('scores')}
          >
            <span>Scores ({rawData.pred_scores.length})</span>
            <span>{expandedSections.has('scores') ? '▼' : '▶'}</span>
          </button>
          {expandedSections.has('scores') && (
            <div className="section-content">
              {rawData.pred_scores.map((score, idx) => (
                <div key={idx} className="data-item">
                  <strong>Mask {idx + 1}:</strong> {score.toFixed(4)}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="log-section">
        <button
          className="section-header"
          onClick={() => toggleSection('full-json')}
        >
          <span>Full JSON</span>
          <span>{expandedSections.has('full-json') ? '▼' : '▶'}</span>
        </button>
        {expandedSections.has('full-json') && (
          <div className="section-content">
            <pre className="json-viewer">
              {JSON.stringify(rawData, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

