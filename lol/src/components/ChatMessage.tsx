import { Satellite, User } from "lucide-react";
import { useEffect, useRef, useState } from "react";

interface Message {
  id: string;
  type: "user" | "ai";
  content: string;
  image_url?: string;
  aiImage?: string;
  boxes?: Array<{ x1: number; y1: number; x2: number; y2: number; label?: string; confidence?: number }>;
  imageWidth?: number;
  imageHeight?: number;
}

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isAI = message.type === "ai";

  return (
    <div className={`flex items-start gap-3 ${isAI ? "" : "flex-row-reverse"}`}>
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center shadow-lg ${
          isAI
            ? "bg-gradient-to-br from-blue-500 to-cyan-500 shadow-blue-500/50"
            : "bg-gradient-to-br from-blue-500 to-cyan-500 shadow-blue-500/50"
        }`}
      >
        <span className="text-xs">{isAI ? <Satellite /> : <User />}</span>
      </div>

      {/* Message Content */}
      <div className={`flex-1 max-w-3xl ${isAI ? "" : "flex flex-col items-end"}`}>
        {/* Image if present (user uploaded - shown for first query with parent_id = null) */}
        {message.image_url && (
          <div className="mb-3">
            <img
              src={message.image_url}
              alt="Uploaded satellite"
              className="rounded-xl max-w-sm w-full border border-white/10 shadow-lg"
            />
          </div>
        )}

        {/* Text Bubble */}
        {message.content && message.content.trim() !== "" && (
          <div
            className={`px-5 py-3.5 rounded-2xl shadow-lg ${
              isAI
                ? " border border-white/10 bg-blue-500/20 text-gray-200"
                : "  bg-blue-500/20 text-white"
            }`}
          >
            <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
          </div>
        )}

        {/* AI Image if present (AI generated annotated image) */}
        {message.aiImage && (
          <div className="mt-3 relative">
            {message.boxes && message.boxes.length > 0 ? (
              <AnnotatedImage
                imageUrl={message.aiImage}
                boxes={message.boxes}
                imageWidth={message.imageWidth}
                imageHeight={message.imageHeight}
              />
            ) : (
              <img
                src={message.aiImage}
                alt="AI annotated result"
                className="rounded-xl max-w-sm w-full border border-emerald-500/30 shadow-lg shadow-emerald-500/20"
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function AnnotatedImage({
  imageUrl,
  boxes,
  imageWidth,
  imageHeight,
}: {
  imageUrl: string;
  boxes: Array<{ x1: number; y1: number; x2: number; y2: number; label?: string; confidence?: number }>;
  imageWidth?: number;
  imageHeight?: number;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [, setImageSize] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    const image = imageRef.current;
    const canvas = canvasRef.current;
    const container = containerRef.current;
    
    if (!image || !canvas || !container) return;

    const handleImageLoad = () => {
      // Get the actual displayed image size
      const displayedWidth = image.offsetWidth;
      const displayedHeight = image.offsetHeight;
      setImageSize({ width: displayedWidth, height: displayedHeight });

      // Set canvas size to match displayed image
      canvas.width = displayedWidth;
      canvas.height = displayedHeight;

      // Get the original image dimensions (from props or actual image)
      const originalWidth = imageWidth || image.naturalWidth;
      const originalHeight = imageHeight || image.naturalHeight;

      // Calculate scale factors
      const scaleX = displayedWidth / originalWidth;
      const scaleY = displayedHeight / originalHeight;

      // Draw boxes on canvas
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "red";
      ctx.lineWidth = 3;

      boxes.forEach((box) => {
        // Scale coordinates to match displayed image size
        const x1 = box.x1 * scaleX;
        const y1 = box.y1 * scaleY;
        const x2 = box.x2 * scaleX;
        const y2 = box.y2 * scaleY;

        // Draw rectangle
        ctx.beginPath();
        ctx.rect(x1, y1, x2 - x1, y2 - y1);
        ctx.stroke();

        // Draw label if available
        if (box.label) {
          ctx.fillStyle = "red";
          ctx.font = "14px Arial";
          ctx.fillText(
            `${box.label}${box.confidence ? ` (${(box.confidence * 100).toFixed(0)}%)` : ""}`,
            x1,
            y1 - 5
          );
        }
      });
    };

    if (image.complete) {
      handleImageLoad();
    } else {
      image.addEventListener("load", handleImageLoad);
    }

    // Handle window resize
    const handleResize = () => {
      if (image.complete) {
        handleImageLoad();
      }
    };

    window.addEventListener("resize", handleResize);

    return () => {
      image.removeEventListener("load", handleImageLoad);
      window.removeEventListener("resize", handleResize);
    };
  }, [imageUrl, boxes, imageWidth, imageHeight]);

  return (
    <div ref={containerRef} className="relative inline-block">
      <img
        ref={imageRef}
        src={imageUrl}
        alt="AI annotated result"
        className="rounded-xl max-w-sm w-full border border-emerald-500/30 shadow-lg shadow-emerald-500/20"
        style={{ display: "block" }}
      />
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 pointer-events-none rounded-xl"
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
