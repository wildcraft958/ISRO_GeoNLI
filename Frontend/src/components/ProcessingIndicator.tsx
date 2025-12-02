import { useEffect, useState } from "react";
import { Brain, Satellite } from "lucide-react";

interface ProcessingIndicatorProps {
  mode: "caption" | "question" | "grounding";
}

export function ProcessingIndicator({ mode }: ProcessingIndicatorProps) {
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState("Initializing");

  useEffect(() => {
    // Reset progress
    setProgress(0);

    // Animate progress from 0 to 100 over 5 seconds
    const duration = 5000;
    const intervalTime = 50; // Update every 50ms
    const increment = (100 / duration) * intervalTime;

    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        const newProgress = prev + increment;
        if (newProgress >= 100) {
          clearInterval(progressInterval);
          return 100;
        }
        return newProgress;
      });
    }, intervalTime);

    // Update stage text at different progress points
    const stages = [
      { time: 0, text: "Initializing analysis..." },
      { time: 1000, text: "Processing imagery..." },
      { time: 2500, text: "Analyzing features..." },
      { time: 4000, text: "Generating response..." },
    ];

    const timeouts: NodeJS.Timeout[] = [];
    stages.forEach(({ time, text }) => {
      const timeout = setTimeout(() => setStage(text), time);
      timeouts.push(timeout);
    });

    return () => {
      clearInterval(progressInterval);
      timeouts.forEach(clearTimeout);
    };
  }, []);

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
        <Brain className="w-4 h-4 text-cyan-400 animate-pulse" />
        <div className="absolute inset-0 bg-cyan-400/20 rounded-full blur-md animate-pulse" />
      </div>

      {/* Progress Container */}
      <div className="flex-1 space-y-3 mt-0.5">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Satellite className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-medium text-cyan-300">{getModeText()}</span>
          </div>
          <span className="text-xs text-cyan-500/60 tabular-nums">{Math.round(progress)}%</span>
        </div>

        {/* Progress Bar */}
        <div className="relative w-full h-2 bg-cyan-950/50 rounded-full overflow-hidden border border-cyan-500/20">
          {/* Background shimmer effect */}
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-500/10 to-transparent animate-shimmer" />

          {/* Progress fill */}
          <div
            className="h-full bg-gradient-to-r from-cyan-500 via-blue-500 to-cyan-400 rounded-full transition-all duration-300 ease-out relative overflow-hidden"
            style={{ width: `${progress}%` }}
          >
            {/* Animated shine effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shine" />
          </div>
        </div>

        {/* Stage Text */}
        <p className="text-xs text-cyan-400/70 animate-pulse">{stage}</p>
      </div>

      {/* Inject keyframes for animations */}
      <style>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        @keyframes shine {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(200%); }
        }
        .animate-shimmer {
          animation: shimmer 2s infinite;
        }
        .animate-shine {
          animation: shine 1.5s infinite;
        }
      `}</style>
    </div>
  );
}
