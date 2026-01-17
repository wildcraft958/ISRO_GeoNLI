import { Globe2, Sparkles, Target, MessageSquare, Search, Filter } from "lucide-react";
import { useState } from "react";
import groundingOriginal from "../assets/sat_demo.jpg";

interface Sample {
  id: string;
  title: string;
  description: string;
  category: "urban" | "agriculture" | "forest" | "coastal" | "desert" | "disaster";
  imageUrl: string;
  analysis: string;
  mode: "caption" | "question" | "grounding";
}

export interface SampleData {
  id: string;
  title: string;
  description: string;
  category: "urban" | "agriculture" | "forest" | "coastal" | "desert" | "disaster";
  imageUrl: string;
  analysis: string;
  mode: "caption" | "question" | "grounding";
}

interface ExploreSamplesPageProps {
  onTrySample: (sample: SampleData) => void;
}

export function ExploreSamplesPage({ onTrySample }: ExploreSamplesPageProps) {
  const [selectedMode, setSelectedMode] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");

  const samples: Sample[] = [
    {
      id: "1",
      title: "Aircraft Detection",
      description: "Airport runway with aircraft",
      category: "urban",
      imageUrl: groundingOriginal,
      analysis: "Find all aircraft in this image",
      mode: "grounding",
    },
    {
      id: "2",
      title: "Agricultural Field Patterns",
      description: "Organized crop cultivation with irrigation systems",
      category: "agriculture",
      imageUrl:
        "https://images.unsplash.com/photo-1720386063956-00296002a701?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxhZ3JpY3VsdHVyYWwlMjBmaWVsZHMlMjBhZXJpYWx8ZW58MXx8fHwxNzY0MTkxODc1fDA&ixlib=rb-4.1.0&q=80&w=1080",
      analysis: "Agricultural area showing systematic crop rotation with center-pivot irrigation",
      mode: "caption",
    },
    {
      id: "3",
      title: "Forest Coverage Assessment",
      description: "Dense vegetation and natural forest ecosystem",
      category: "forest",
      imageUrl:
        "https://images.unsplash.com/photo-1585644013005-a8028ecd0bb6?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxmb3Jlc3QlMjBsYW5kc2NhcGUlMjBhZXJpYWx8ZW58MXx8fHwxNzY0MTkxODc0fDA&ixlib=rb-4.1.0&q=80&w=1080",
      analysis: "What is the estimated forest density and health status in this region?",
      mode: "question",
    },
    {
      id: "4",
      title: "Coastal Region Monitoring",
      description: "Shoreline and coastal water analysis",
      category: "coastal",
      imageUrl:
        "https://images.unsplash.com/photo-1598674654570-039c170d184a?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjb2FzdGFsJTIwb2NlYW4lMjBhZXJpYWx8ZW58MXx8fHwxNzY0MTkxODc1fDA&ixlib=rb-4.1.0&q=80&w=1080",
      analysis: "Coastal area with clear water, visible reef structures, and minimal development",
      mode: "caption",
    },
    {
      id: "5",
      title: "Urban Infrastructure",
      description: "City infrastructure with buildings and roads",
      category: "urban",
      imageUrl:
        "https://images.unsplash.com/photo-1760459477099-ad81fd11d7c6?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx1cmJhbiUyMGNpdHklMjBhZXJpYWx8ZW58MXx8fHwxNzY0MTkxODc0fDA&ixlib=rb-4.1.0&q=80&w=1080",
      analysis: "Detect and locate all buildings and major roads in this image",
      mode: "grounding",
    },
    {
      id: "6",
      title: "Global Earth View",
      description: "Wide-angle satellite perspective",
      category: "urban",
      imageUrl:
        "https://images.unsplash.com/photo-1574786198374-9461cb650c23?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxzYXRlbGxpdGUlMjBlYXJ0aCUyMHZpZXd8ZW58MXx8fHwxNzY0MDg1OTA4fDA&ixlib=rb-4.1.0&q=80&w=1080",
      analysis:
        "What patterns of human activity and natural features are visible from this perspective?",
      mode: "question",
    },
  ];

  const categories = [
    { id: "all", label: "All Samples", icon: Globe2 },
    { id: "caption", label: "Caption", icon: Sparkles },
    { id: "question", label: "Question", icon: MessageSquare },
    { id: "grounding", label: "Grounding", icon: Target },
  ];

  const getModeIcon = (mode: string) => {
    switch (mode) {
      case "caption":
        return <Sparkles className="w-4 h-4" />;
      case "question":
        return <MessageSquare className="w-4 h-4" />;
      case "grounding":
        return <Target className="w-4 h-4" />;
      default:
        return null;
    }
  };

  const getModeColor = (mode: string) => {
    switch (mode) {
      case "caption":
        return "bg-purple-500/20 text-purple-400 border-purple-500/30";
      case "question":
        return "bg-blue-500/20 text-blue-400 border-blue-500/30";
      case "grounding":
        return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
      default:
        return "bg-gray-500/20 text-gray-400 border-gray-500/30";
    }
  };

  const filteredSamples = samples.filter((sample) => {
    const matchesMode = selectedMode === "all" || sample.mode === selectedMode;
    const matchesSearch =
      sample.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      sample.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesMode && matchesSearch;
  });

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-cyan-500/20 bg-gradient-to-b from-cyan-500/5 to-transparent">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-cyan-500/20 rounded-xl">
              <Globe2 className="w-6 h-6 text-cyan-400" />
            </div>
            <div>
              <h1 className="text-2xl text-white">Explore Samples</h1>
              <p className="text-sm text-gray-400">
                Discover pre-analyzed satellite imagery examples
              </p>
            </div>
          </div>

          {/* Search Bar */}
          <div className="relative mb-4">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-cyan-500/50" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search samples..."
              className="w-full bg-white/5 border border-cyan-500/30 rounded-xl pl-12 pr-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-500/20 transition-all"
            />
          </div>

          {/* Category Filter */}
          <div className="flex items-center gap-2 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-cyan-500/20 scrollbar-track-transparent">
            {categories.map((category) => (
              <button
                key={category.id}
                onClick={() => setSelectedMode(category.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all duration-300 whitespace-nowrap border ${
                  selectedMode === category.id
                    ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/40"
                    : "bg-white/5 text-gray-400 border-transparent hover:bg-white/10 hover:text-cyan-400"
                }`}
              >
                <category.icon className="w-4 h-4" />
                {category.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Sample Grid */}
      <div className="flex-1 overflow-y-auto p-6 pb-24 scrollbar-thin scrollbar-thumb-cyan-500/20 scrollbar-track-transparent">
        <div className="max-w-6xl mx-auto">
          {filteredSamples.length === 0 ? (
            <div className="text-center py-16">
              <Globe2 className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-400">No samples found</p>
              <p className="text-sm text-gray-500 mt-2">Try adjusting your search or filter</p>
            </div>
          ) : (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredSamples.map((sample) => (
                <div
                  key={sample.id}
                  className="group bg-white/5 border border-cyan-500/20 rounded-2xl overflow-hidden hover:bg-white/10 hover:border-cyan-500/40 transition-all duration-300"
                >
                  {/* Image */}
                  <div className="relative aspect-[4/3] overflow-hidden bg-black">
                    <img
                      src={sample.imageUrl}
                      alt={sample.title}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                    <div className="absolute top-3 right-3">
                      <span
                        className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs border backdrop-blur-sm ${getModeColor(sample.mode)}`}
                      >
                        {getModeIcon(sample.mode)}
                        {sample.mode}
                      </span>
                    </div>
                  </div>

                  {/* Content */}
                  <div className="p-5">
                    <h3 className="text-white font-medium mb-2">{sample.title}</h3>
                    <p className="text-sm text-gray-400 mb-3 line-clamp-2">{sample.description}</p>

                    {/* Analysis Preview */}
                    <div className="bg-black/30 border border-cyan-500/20 rounded-lg p-3 mb-4">
                      <p className="text-xs text-cyan-300/80 italic line-clamp-2">
                        {sample.analysis}
                      </p>
                    </div>

                    {/* Try Button */}
                    <button
                      onClick={() => onTrySample(sample)}
                      className="w-full py-2.5 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 border border-cyan-500/30 text-cyan-300 rounded-lg hover:from-cyan-500/30 hover:to-blue-500/30 hover:border-cyan-400/50 transition-all duration-300 text-sm font-medium"
                    >
                      Try This Sample
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
