
interface Message {
  id: string;
  type: "user" | "ai";
  content: string;
  image?: string;
  aiImage?: string;
}

import { Satellite, User } from "lucide-react";

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isAI = message.type === "ai";
console.log("hiiiiii this is ",message.image)
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

        {/* USER UPLOADED IMAGE – LARGE */}
        {message.image && (
          <div className="mb-3">
            <img
              src={message.image}
              alt="Uploaded satellite"
              className="rounded-xl w-full max-w-3xl max-h-[600px] object-contain border border-white/10 shadow-xl"
            />
          </div>
        )}

        {/* TEXT */}
        {message.content && (
          <div
            className={`px-5 py-3.5 rounded-2xl shadow-lg ${
              isAI
                ? "border border-white/10 bg-blue-500/20 text-gray-200"
                : "bg-blue-500/20 text-white"
            }`}
          >
            <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
          </div>
        )}

        {/* AI-GENERATED IMAGE – LARGE */}
        {message.aiImage && (
          <div className="mt-3">
            <img
              src={message.aiImage}
              alt="AI annotated result"
              className="rounded-xl w-full max-w-3xl max-h-[600px] object-contain border border-emerald-500/30 shadow-xl shadow-emerald-500/20"
            />
          </div>
        )}
      </div>
    </div>
  );
}
