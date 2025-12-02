interface Message {
  id: string;
  type: "user" | "ai";
  content: string;
  image?: string;
  aiImage?: string;
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
        <span className="text-xs">{isAI ? "🧠" : "👤"}</span>
      </div>

      {/* Message Content */}
      <div className={`flex-1 max-w-3xl ${isAI ? "" : "flex flex-col items-end"}`}>
        {/* Image if present (user uploaded) */}
        {message.image && (
          <div className="mb-3">
            <img
              src={message.image}
              alt="Uploaded satellite"
              className="rounded-xl max-w-sm w-full border border-white/10 shadow-lg"
            />
          </div>
        )}

        {/* Text Bubble */}
        {message.content && (
          <div
            className={`px-5 py-3.5 rounded-2xl shadow-lg ${
              isAI
                ? "bg-white/[0.05] border border-white/10 backdrop-blur-sm text-gray-200"
                : "bg-gradient-to-br from-blue-600 to-cyan-600 text-white"
            }`}
          >
            <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
          </div>
        )}

        {/* AI Image if present (AI generated annotated image) */}
        {message.aiImage && (
          <div className="mt-3">
            <img
              src={message.aiImage}
              alt="AI annotated result"
              className="rounded-xl max-w-sm w-full border border-emerald-500/30 shadow-lg shadow-emerald-500/20"
            />
          </div>
        )}
      </div>
    </div>
  );
}
