import { MessageSquare, Clock, Trash2, Search, Calendar } from "lucide-react";
import { useState } from "react";
import { useUser } from "@clerk/clerk-react";
interface ChatHistoryItem {
  id: string;
  title: string;
  preview: string;
  date: string;
  mode: "caption" | "question" | "grounding";
  messageCount: number;
}

interface ChatHistoryPageProps {
  onSelectChat: (chatId: string) => void;
}

export function ChatHistoryPage({ onSelectChat }: ChatHistoryPageProps) {
  const [searchQuery, setSearchQuery] = useState("");
   // const [chats,setChats]=useState<ChatType[]>([])
    // const { user } = useUser();
  
  //   useEffect(() => {
  //     try {
  //       const fetchChats = async () => {
  //         const response = await apiClient.get(routes.GET_ALL_CHATS,{
  //           user_id:user?.id
  //         });
  //         //getting array of chats
  //         setChats(response.data)
  //       };
  //       fetchChats();
  //     } catch (err) {
  //       console.log(err);
  //     }
  //   }, []);
  // Mock chat history data
  const chatHistory: ChatHistoryItem[] = [
    {
      id: "1",
      title: "Urban Development Analysis",
      preview: "What are the changes in urban infrastructure over the past 5 years?",
      date: "2025-11-26",
      mode: "question",
      messageCount: 12,
    },
    {
      id: "2",
      title: "Agricultural Land Detection",
      preview: "Detect all agricultural areas in this satellite image",
      date: "2025-11-25",
      mode: "grounding",
      messageCount: 8,
    },
    {
      id: "3",
      title: "Coastal Region Caption",
      preview: "Generate caption for coastal satellite imagery",
      date: "2025-11-24",
      mode: "caption",
      messageCount: 5,
    },
    {
      id: "4",
      title: "Forest Coverage Study",
      preview: "Analyze forest density and vegetation patterns",
      date: "2025-11-23",
      mode: "question",
      messageCount: 15,
    },
    {
      id: "5",
      title: "Infrastructure Mapping",
      preview: "Identify roads, buildings, and water bodies",
      date: "2025-11-22",
      mode: "grounding",
      messageCount: 10,
    },
    {
      id: "6",
      title: "Disaster Assessment",
      preview: "Caption the damage extent from recent flooding",
      date: "2025-11-21",
      mode: "caption",
      messageCount: 7,
    },
  ];

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

  const filteredHistory = chatHistory.filter(
    (chat) =>
      chat.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      chat.preview.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-cyan-500/20 bg-gradient-to-b from-cyan-500/5 to-transparent">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-cyan-500/20 rounded-xl">
              <MessageSquare className="w-6 h-6 text-cyan-400" />
            </div>
            <div>
              <h1 className="text-2xl text-white">Chat History</h1>
              <p className="text-sm text-gray-400">View and manage your past conversations</p>
            </div>
          </div>

          {/* Search Bar */}
          <div className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-cyan-500/50" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search chat history..."
              className="w-full bg-white/5 border border-cyan-500/30 rounded-xl pl-12 pr-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-500/20 transition-all"
            />
          </div>
        </div>
      </div>

      {/* Chat List */}
      <div className="flex-1 overflow-y-auto p-6 pb-24 scrollbar-thin scrollbar-thumb-cyan-500/20 scrollbar-track-transparent">
        <div className="max-w-4xl mx-auto space-y-4">
          {filteredHistory.length === 0 ? (
            <div className="text-center py-16">
              <MessageSquare className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-400">No chat history found</p>
              <p className="text-sm text-gray-500 mt-2">Start a new conversation to see it here</p>
            </div>
          ) : (
            filteredHistory.map((chat) => (
              <div
                key={chat.id}
                className="group bg-white/5 border border-cyan-500/20 rounded-2xl p-5 hover:bg-white/10 hover:border-cyan-500/40 transition-all duration-300 cursor-pointer"
                onClick={() => onSelectChat(chat.id)}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="text-white font-medium truncate">{chat.title}</h3>
                      <span
                        className={`px-2.5 py-0.5 rounded-full text-xs border ${getModeColor(chat.mode)}`}
                      >
                        {chat.mode}
                      </span>
                    </div>
                    <p className="text-gray-400 text-sm mb-3 line-clamp-2">{chat.preview}</p>
                    <div className="flex items-center gap-4 text-xs text-gray-500">
                      <span className="flex items-center gap-1.5">
                        <Calendar className="w-3.5 h-3.5" />
                        {new Date(chat.date).toLocaleDateString("en-US", {
                          month: "short",
                          day: "numeric",
                          year: "numeric",
                        })}
                      </span>
                      <span className="flex items-center gap-1.5">
                        <MessageSquare className="w-3.5 h-3.5" />
                        {chat.messageCount} messages
                      </span>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        // Handle delete
                      }}
                      className="p-2 bg-red-500/10 text-red-400 rounded-lg hover:bg-red-500/20 transition-colors"
                      aria-label="Delete chat"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
