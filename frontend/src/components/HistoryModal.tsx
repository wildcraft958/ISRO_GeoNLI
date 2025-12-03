import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import { ScrollArea } from "./ui/scroll-area";
import { Trash2, Clock, MessageSquare } from "lucide-react";
import { ChatSession, Analysis, Sample } from "../utils/api";

interface HistoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  type: "chats" | "analyses" | "samples";
  items: ChatSession[] | Analysis[] | Sample[];
  onItemClick: (item: any) => void;
  onDelete?: (itemId: string) => void;
}

export function HistoryModal({
  isOpen,
  onClose,
  type,
  items,
  onItemClick,
  onDelete,
}: HistoryModalProps) {
  const getTitle = () => {
    switch (type) {
      case "chats":
        return "Chat History";
      case "analyses":
        return "Past Analyses";
      case "samples":
        return "Explore Samples";
      default:
        return "";
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl bg-[#0a0a0a]/95 backdrop-blur-2xl border-white/10 text-white">
        <DialogHeader>
          <DialogTitle className="text-2xl bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            {getTitle()}
          </DialogTitle>
        </DialogHeader>

        <ScrollArea className="h-[500px] pr-4">
          {items.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-zinc-500">
              <MessageSquare className="w-16 h-16 mb-4 opacity-30" />
              <p>No {type} found</p>
              <p className="text-sm mt-2">Start creating analyses to see them here</p>
            </div>
          ) : (
            <div className="space-y-3">
              {type === "chats" &&
                (items as ChatSession[]).map((chat) => (
                  <div
                    key={chat.id}
                    className="group p-4 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-blue-500/30 rounded-xl cursor-pointer transition-all duration-300"
                    onClick={() => onItemClick(chat)}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <h3 className="font-medium text-white truncate mb-1">{chat.title}</h3>
                        <div className="flex items-center gap-2 text-xs text-zinc-400">
                          <Clock className="w-3 h-3" />
                          <span>{formatDate(chat.createdAt)}</span>
                          <span>•</span>
                          <span className="capitalize">{chat.mode} mode</span>
                          <span>•</span>
                          <span>{chat.messages.length} messages</span>
                        </div>
                      </div>
                      {onDelete && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 hover:bg-red-500/10"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDelete(chat.id);
                          }}
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                  </div>
                ))}

              {type === "analyses" &&
                (items as Analysis[]).map((analysis) => (
                  <div
                    key={analysis.id}
                    className="group p-4 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-cyan-500/30 rounded-xl cursor-pointer transition-all duration-300"
                    onClick={() => onItemClick(analysis)}
                  >
                    <div className="flex items-start gap-4">
                      {analysis.imageUrl && (
                        <div className="w-20 h-20 flex-shrink-0 rounded-lg overflow-hidden bg-white/5">
                          <img
                            src={analysis.imageUrl}
                            alt={analysis.title}
                            className="w-full h-full object-cover"
                          />
                        </div>
                      )}
                      <div className="flex-1 min-w-0">
                        <h3 className="font-medium text-white truncate mb-1">{analysis.title}</h3>
                        <p className="text-sm text-zinc-400 line-clamp-2 mb-2">{analysis.result}</p>
                        <div className="flex items-center gap-2 text-xs text-zinc-500">
                          <Clock className="w-3 h-3" />
                          <span>{formatDate(analysis.createdAt)}</span>
                          <span>•</span>
                          <span className="capitalize">{analysis.type}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}

              {type === "samples" &&
                (items as Sample[]).map((sample) => (
                  <div
                    key={sample.id}
                    className="group p-4 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-purple-500/30 rounded-xl cursor-pointer transition-all duration-300"
                    onClick={() => onItemClick(sample)}
                  >
                    <div className="flex items-start gap-4">
                      <div className="w-24 h-24 flex-shrink-0 rounded-lg overflow-hidden bg-white/5">
                        <img
                          src={sample.imageUrl}
                          alt={sample.title}
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-medium text-white mb-1">{sample.title}</h3>
                        <p className="text-sm text-zinc-400 line-clamp-2 mb-2">
                          {sample.description}
                        </p>
                        <div className="flex items-center gap-2 text-xs">
                          <span className="px-2 py-1 bg-blue-500/20 text-blue-400 rounded-md capitalize">
                            {sample.type}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          )}
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
