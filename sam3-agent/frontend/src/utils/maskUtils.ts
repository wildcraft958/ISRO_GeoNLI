/**
 * RLE (Run-Length Encoding) mask decoding utilities
 * Supports both string and dict formats compatible with pycocotools
 */

export interface RLEMask {
  counts: string | number[];
  size: number[];
}

/**
 * Decode RLE mask to binary array
 * Supports both string counts and array counts formats
 * OPTIMIZED: Uses fill instead of nested loops for much better performance
 */
export function decodeRLE(rle: RLEMask, width?: number, height?: number): Uint8Array {
  const [h, w] = rle.size;
  const targetHeight = height || h;
  const targetWidth = width || w;
  const mask = new Uint8Array(targetHeight * targetWidth);

  let counts: number[];
  if (typeof rle.counts === 'string') {
    // String format: "1 2 3 4..." - decode from string
    counts = rle.counts.split(/\s+/).map(Number).filter(n => !isNaN(n));
  } else {
    // Array format: [1, 2, 3, 4...]
    counts = rle.counts;
  }

  let pos = 0;
  let value = 0;

  // OPTIMIZED: Use fill() instead of nested loop for much better performance
  for (let i = 0; i < counts.length; i++) {
    const count = counts[i];
    if (value === 1 && pos + count <= mask.length) {
      // Only fill when value is 1 (mask pixels), skip when 0 (already initialized to 0)
      mask.fill(1, pos, pos + count);
    }
    pos += count;
    value = 1 - value; // Toggle between 0 and 1
  }

  // Resize if needed
  if (targetWidth !== w || targetHeight !== h) {
    return resizeMask(mask, w, h, targetWidth, targetHeight);
  }

  return mask;
}

/**
 * Resize a mask array from source dimensions to target dimensions
 */
function resizeMask(
  mask: Uint8Array,
  srcWidth: number,
  srcHeight: number,
  targetWidth: number,
  targetHeight: number
): Uint8Array {
  const resized = new Uint8Array(targetHeight * targetWidth);
  const scaleX = srcWidth / targetWidth;
  const scaleY = srcHeight / targetHeight;

  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const srcX = Math.floor(x * scaleX);
      const srcY = Math.floor(y * scaleY);
      const srcIdx = srcY * srcWidth + srcX;
      const targetIdx = y * targetWidth + x;
      if (srcIdx < mask.length) {
        resized[targetIdx] = mask[srcIdx];
      }
    }
  }

  return resized;
}

/**
 * Convert normalized bbox to pixel coordinates
 * Format: [x1, y1, x2, y2] - corner-based (xyxy format)
 * 
 * Input: normalized coordinates in range [0, 1]
 * Output: pixel coordinates [x1, y1, x2, y2]
 */
export function normalizedBboxToPixels(
  bbox: number[],
  imgWidth: number,
  imgHeight: number
): [number, number, number, number] {
  if (bbox.length < 4) {
    console.warn('Invalid bbox format, expected 4 values');
    return [0, 0, 0, 0];
  }

  const [x1_norm, y1_norm, x2_norm, y2_norm] = bbox;
  
  // Convert normalized [x1, y1, x2, y2] to pixel coordinates
  const x1 = Math.floor(x1_norm * imgWidth);
  const y1 = Math.floor(y1_norm * imgHeight);
  const x2 = Math.floor(x2_norm * imgWidth);
  const y2 = Math.floor(y2_norm * imgHeight);

  // Clamp to image bounds
  return [
    Math.max(0, Math.min(x1, imgWidth - 1)),
    Math.max(0, Math.min(y1, imgHeight - 1)),
    Math.max(0, Math.min(x2, imgWidth - 1)),
    Math.max(0, Math.min(y2, imgHeight - 1)),
  ];
}

/**
 * Convert mask array to 2D array for easier manipulation
 */
export function maskTo2D(mask: Uint8Array, width: number, height: number): Uint8Array[][] {
  const result: Uint8Array[][] = [];
  for (let y = 0; y < height; y++) {
    const row: Uint8Array[] = [];
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      row.push(new Uint8Array([mask[idx]]));
    }
    result.push(row);
  }
  return result;
}

