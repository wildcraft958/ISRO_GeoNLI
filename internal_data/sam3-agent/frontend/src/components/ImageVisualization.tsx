import { useEffect, useRef, useState } from 'react';
import { drawMasksBatch, drawBbox, drawLabel, generateColors } from '../utils/visualization';
import { RLEMask } from '../utils/maskUtils';

interface Region {
  bbox?: number[];
  mask?: RLEMask;
  score?: number;
}

interface ImageVisualizationProps {
  imageUrl: string | null;
  regions?: Region[];
  rawData?: {
    orig_img_h: number;
    orig_img_w: number;
    pred_boxes: number[][];
    pred_masks: RLEMask[];
    pred_scores: number[];
  };
  showMasks?: boolean;
  showBboxes?: boolean;
  showLabels?: boolean;
  maxMasks?: number; // Limit number of masks to render for performance
}

export default function ImageVisualization({
  imageUrl,
  regions,
  rawData,
  showMasks = true,
  showBboxes = true,
  showLabels = true,
  maxMasks = 50, // Default limit to prevent performance issues
}: ImageVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [isRendering, setIsRendering] = useState(false);

  useEffect(() => {
    if (!imageUrl || !canvasRef.current) {
      if (!imageUrl) {
        console.log('[Viz] No imageUrl provided, skipping render');
      }
      if (!canvasRef.current) {
        console.warn('[Viz] Canvas ref not available');
      }
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.error('[Viz] Failed to get 2D canvas context. Canvas may not be supported in this browser.');
      return;
    }

    console.log('[Viz] Starting image visualization render', {
      hasRegions: !!regions,
      hasRawData: !!rawData,
      regionsCount: regions?.length || 0,
    });

    const img = new Image();
    img.crossOrigin = 'anonymous';
    
    img.onload = async () => {
      console.log('[Viz] Image loaded', {
        width: img.width,
        height: img.height,
        imageUrl: imageUrl.substring(0, 100) + '...',
      });

      setIsRendering(true);
      imageRef.current = img;
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw base image
      try {
        ctx.drawImage(img, 0, 0);
        console.log('[Viz] Base image drawn to canvas');
      } catch (error) {
        console.error('[Viz] Error drawing base image:', error);
        setIsRendering(false);
        return;
      }

      // Determine data source
      let masks: RLEMask[] = [];
      let boxes: number[][] = [];
      let scores: number[] = [];

      if (rawData) {
        console.log('[Viz] Using rawData source', {
          masksCount: rawData.pred_masks?.length || 0,
          boxesCount: rawData.pred_boxes?.length || 0,
          scoresCount: rawData.pred_scores?.length || 0,
        });
        masks = rawData.pred_masks || [];
        boxes = rawData.pred_boxes || [];
        scores = rawData.pred_scores || [];
      } else if (regions) {
        console.log('[Viz] Using regions source', {
          regionsCount: regions.length,
        });
        masks = regions.map(r => r.mask!).filter(m => m);
        boxes = regions.map(r => r.bbox!).filter(b => b);
        scores = regions.map(r => r.score || 0);
      }

      console.log('[Viz] Extracted data', {
        masksCount: masks.length,
        boxesCount: boxes.length,
        scoresCount: scores.length,
      });

      if (masks.length === 0 && boxes.length === 0) {
        console.warn('[Viz] No masks or boxes to render');
        setIsRendering(false);
        return;
      }

      // Sort by score (descending) and limit number of masks for performance
      if (masks.length > maxMasks) {
        console.warn(`[Viz] Limiting masks from ${masks.length} to ${maxMasks} for performance`);
        const indices = scores
          .map((score, idx) => ({ score, idx }))
          .sort((a, b) => b.score - a.score)
          .slice(0, maxMasks)
          .map(item => item.idx);
        
        masks = indices.map(idx => masks[idx]);
        boxes = indices.map(idx => boxes[idx]);
        scores = indices.map(idx => scores[idx]);
      }

      const colors = generateColors(Math.max(masks.length, boxes.length));
      console.log('[Viz] Generated colors', { colorCount: colors.length });

      // Use requestAnimationFrame to keep UI responsive
      requestAnimationFrame(() => {
        const renderStartTime = Date.now();
        try {
          // Draw masks using batch processing for better performance
          if (showMasks && masks.length > 0) {
            const validMasks = masks.filter(m => m);
            if (validMasks.length > 0) {
              console.log('[Viz] Drawing masks', { count: validMasks.length });
              try {
                drawMasksBatch(ctx, validMasks, img.width, img.height, colors, 0.3);
                console.log('[Viz] Masks drawn successfully');
              } catch (maskError) {
                console.error('[Viz] Error drawing masks:', maskError);
                throw maskError;
              }
            } else {
              console.warn('[Viz] No valid masks to draw after filtering');
            }
          }

          // Draw bounding boxes
          if (showBboxes) {
            console.log('[Viz] Drawing bounding boxes', { count: boxes.length });
            boxes.forEach((box, idx) => {
              if (box) {
                try {
                  drawBbox(ctx, box, img.width, img.height, colors[idx], 2);
                } catch (bboxError) {
                  console.error(`[Viz] Error drawing bbox ${idx}:`, bboxError, { box });
                }
              }
            });
          }

          // Draw labels
          if (showLabels) {
            console.log('[Viz] Drawing labels', { count: boxes.length });
            boxes.forEach((box, idx) => {
              if (box) {
                try {
                  const score = scores[idx];
                  drawLabel(ctx, box, img.width, img.height, `${idx + 1}`, colors[idx], score);
                } catch (labelError) {
                  console.error(`[Viz] Error drawing label ${idx}:`, labelError);
                }
              }
            });
          }

          const renderDuration = Date.now() - renderStartTime;
          console.log('[Viz] Visualization rendering completed', {
            duration: `${renderDuration}ms`,
            masksDrawn: showMasks ? masks.length : 0,
            boxesDrawn: showBboxes ? boxes.length : 0,
            labelsDrawn: showLabels ? boxes.length : 0,
          });
        } catch (error) {
          console.error('[Viz] Error rendering visualization:', error);
          if (error instanceof Error) {
            console.error('[Viz] Error details:', {
              message: error.message,
              stack: error.stack,
              name: error.name,
            });
          }
        } finally {
          setIsRendering(false);
        }
      });
    };

    img.onerror = (error) => {
      console.error('[Viz] Failed to load image', {
        imageUrl: imageUrl.substring(0, 100) + '...',
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      
      // Check for CORS error
      if (imageUrl.startsWith('http') && !imageUrl.startsWith(window.location.origin)) {
        console.error('[Viz] Possible CORS issue - image is from different origin', {
          imageOrigin: new URL(imageUrl).origin,
          currentOrigin: window.location.origin,
        });
      }
      
      setIsRendering(false);
    };

    img.src = imageUrl;
  }, [imageUrl, regions, rawData, showMasks, showBboxes, showLabels, maxMasks]);

  if (!imageUrl) {
    return (
      <div className="image-visualization-placeholder">
        <p>Upload an image to see visualization</p>
      </div>
    );
  }

  return (
    <div className="image-visualization-container">
      {isRendering && (
        <div className="rendering-overlay">
          <div className="spinner"></div>
          <p>Rendering masks...</p>
        </div>
      )}
      <canvas ref={canvasRef} className="visualization-canvas" />
    </div>
  );
}

