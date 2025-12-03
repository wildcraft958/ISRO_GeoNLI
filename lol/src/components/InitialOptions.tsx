import { Button } from "./ui/button";
import { Sparkles, MessageSquare, Crosshair } from "lucide-react";

interface InitialOptionsProps {
  onModeSelect: (mode: "caption" | "question" | "grounding") => void;
}

export function InitialOptions({ onModeSelect }: InitialOptionsProps) {
  // --- CONFIGURATION ---

  // 1. GRADIENT BASE: Blends your Cyan (#06B6D4) into a Sky Blue for depth
  // This looks much more premium than a flat color.
  const GRADIENT_BG = "bg-gradient-to-br from-[#06B6D4] to-[#0ea5e9]";

  // 2. BORDER: Subtle cyan border
  const BOX_BORDER = "border-cyan-400";

  // 3. HOVER: Brighter, slightly lighter gradient
  const HOVER_EFFECTS =
    "hover:scale-[1.02] hover:shadow-[0_0_25px_rgba(6,182,212,0.5)] hover:brightness-110";

  const modeOptions = [
    {
      mode: "caption" as const,
      icon: Sparkles,
      title: "Generate Caption",
      description: "Upload satellite imagery for AI-generated captions.",
      // Colors: Dark Navy/Blue text looks best on bright Cyan/Sky
      iconColor: "text-cyan-950",
      titleColor: "text-cyan-950",
      descColor: "text-cyan-900",
    },
    {
      mode: "question" as const,
      icon: MessageSquare,
      title: "Ask Question",
      description: "Upload imagery and ask detailed questions.",
      iconColor: "text-cyan-950",
      titleColor: "text-cyan-950",
      descColor: "text-cyan-900",
    },
    {
      mode: "grounding" as const,
      icon: Crosshair,
      title: "Perform Grounding",
      description: "Detect and locate objects in satellite imagery.",
      iconColor: "text-cyan-950",
      titleColor: "text-cyan-950",
      descColor: "text-cyan-900",
    },
  ];

  return (
    <div className="sticky bottom-0 w-full z-30 pb-6 pt-2">
      <div className="max-w-4xl mx-auto px-4 md:px-8">
        {/* Grid Container */}
        <div className="grid md:grid-cols-3 gap-4">
          {modeOptions.map((option) => (
            <button
              key={option.mode}
              onClick={() => onModeSelect(option.mode)}
              className={`
                group relative text-left
                /* GRADIENT BACKGROUND APPLIED HERE */
                ${GRADIENT_BG}
                border ${BOX_BORDER}
                backdrop-filter-none 
                
                rounded-3xl p-5 transition-all duration-300 ease-out
                active:scale-[0.98] shadow-lg
                ${HOVER_EFFECTS}
              `}
            >
              {/* Optional: Subtle inner white glow for a "glossy" feel */}
              <div className="absolute inset-0 rounded-3xl bg-gradient-to-b from-white/20 to-transparent pointer-events-none" />

              <div className="relative z-10">
                {/* Title with Icon */}
                <div className="flex items-center space-x-3 mb-2">
                  <option.icon className={`w-6 h-6 ${option.iconColor} transition-colors`} />
                  <h3 className={`text-lg font-bold ${option.titleColor}`}>{option.title}</h3>
                </div>

                {/* Description */}
                <p className={`text-sm font-semibold ${option.descColor}`}>{option.description}</p>
              </div>
            </button>
          ))}
        </div>

        <p className="text-center text-xs text-zinc-500 mt-4 font-medium"></p>
      </div>
    </div>
  );
}
