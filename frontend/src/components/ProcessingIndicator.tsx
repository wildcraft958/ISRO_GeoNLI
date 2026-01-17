import { Brain, Satellite } from "lucide-react";

interface ProcessingIndicatorProps {
  mode: "caption" | "question" | "grounding";
}

export function ProcessingIndicator({ mode }: ProcessingIndicatorProps) {
  const getModeText = () => {
    switch (mode) {
      case "caption":
        return "Caption Generation";
      case "grounding":
        return "Object Detection";
      case "question":
        return "Visual Q&A";
      default:
        return "Analysis";
    }
  };

  return (
    <div className="flex items-start gap-4 pl-4 animate-in fade-in slide-in-from-bottom-2 duration-500">
      {/* Icon */}
      <div className="relative w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500/20 to-blue-500/20 flex items-center justify-center flex-shrink-0 mt-1">
        <Brain className="w-4 h-4 text-cyan-400 animate-spin" />
        <div className="absolute inset-0 bg-cyan-400/20 rounded-full blur-md animate-pulse" />
      </div>

      {/* Content Container */}
      <div className="flex-1 space-y-3 mt-0.5">
        {/* Header */}
        <div className="flex items-center gap-2">
          <Satellite className="w-4 h-4 text-cyan-400" />
          <span className="text-sm font-medium text-cyan-300">{getModeText()}</span>
        </div>

        {/* Loading Spinner */}
        <div className="flex items-center gap-2">
          <div className="flex gap-1">
            <div
              className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce"
              style={{ animationDelay: "0ms" }}
            />
            <div
              className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce"
              style={{ animationDelay: "150ms" }}
            />
            <div
              className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce"
              style={{ animationDelay: "300ms" }}
            />
          </div>
          <span className="text-xs text-cyan-400/70">Processing...</span>
        </div>
      </div>

      {/* Inject keyframes for animations */}
      <style>{`
        @keyframes bounce {
          0%, 100% { transform: translateY(0); opacity: 1; }
          50% { transform: translateY(-8px); opacity: 0.6; }
        }
        .animate-bounce {
          animation: bounce 1.4s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
}
