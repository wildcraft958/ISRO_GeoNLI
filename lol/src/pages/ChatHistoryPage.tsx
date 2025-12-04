import { MessageSquare, Trash2, Search, Calendar, Loader } from "lucide-react";
import { useState, useEffect } from "react";
import { useUser } from "@clerk/clerk-react";
import { chatService } from "@/services/chatService";

interface ChatHistoryItem {
  id: string;
  image_url?: string;
  created_at: string;
  user_id: string;
  messageCount: number;
  preview: string;
}

interface ChatHistoryPageProps {
  onSelectChat: (chatId: string) => void;
}

export function ChatHistoryPage({ onSelectChat }: ChatHistoryPageProps) {
  const { user } = useUser();
  const [searchQuery, setSearchQuery] = useState("");
  const [chats, setChats] = useState<ChatHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const fetchChats = async () => {
    if (!user?.id) return;
    try {
      setLoading(true);
      setError(null);
      const chatList = await chatService.getChats(user.id);
      // Handle empty chat list
      if (!chatList || chatList.length === 0) {
        setChats([]);
        setLoading(false);
        return;
      }

      // Fetch query count for each chat
      const chatsWithMessages = await Promise.all(
        chatList.map(async (chat: any) => {
          try {
            const queries = await chatService.getChatQueries(chat.id);
            const preview =
              queries.length > 0 && queries[0].request
                ? queries[0].request.substring(0, 80)
                : "No queries yet";

            return {
              id: chat.id || chat.chat_id,
              image_url: chat.image_url,
              created_at: chat.created_at,
              user_id: chat.user_id,
              messageCount: queries.length,
              preview: preview,
            };
          } catch (err) {
            console.error(`Failed to fetch queries for chat ${chat.id}:`, err);
            return {
              id: chat.id || chat.chat_id,
              image_url: chat.image_url,
              created_at: chat.created_at,
              user_id: chat.user_id,
              messageCount: 0,
              preview: "No queries available",
            };
          }
        })
      );

      setChats(chatsWithMessages);
      setError(null);
    } catch (err: any) {
      console.error("Failed to fetch chats:", err);
      setError(err?.response?.data?.detail || err?.message || "Failed to load chat history");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchChats();
  }, [user?.id]);

  const handleDeleteChat = async (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation();

    // Confirm deletion
    const confirmed = window.confirm(
      "Are you sure you want to delete this chat? This action cannot be undone."
    );
    if (!confirmed) return;

    try {
      setDeletingId(chatId);
      await chatService.deleteChat(chatId);

      // Remove from local state
      setChats((prev) => prev.filter((chat) => chat.id !== chatId));
    } catch (err) {
      console.error("Failed to delete chat:", err);
      alert("Failed to delete chat. Please try again.");
    } finally {
      setDeletingId(null);
    }
  };

  const filteredHistory = chats.filter(
    (chat) =>
      chat.preview.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (chat.image_url && chat.image_url.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-cyan-500/20 bg-linear-to-b from-cyan-500/5 to-transparent">
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
          {loading ? (
            <div className="text-center py-16">
              <Loader className="w-8 h-8 text-cyan-400 mx-auto mb-4 animate-spin" />
              <p className="text-gray-400">Loading chat history...</p>
            </div>
          ) : error ? (
            <div className="text-center py-16">
              <MessageSquare className="w-16 h-16 text-red-600/40 mx-auto mb-4" />
              <p className="text-red-400">{error}</p>
            </div>
          ) : filteredHistory.length === 0 ? (
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
                  <div className="flex-1 min-w-0 flex gap-4">
                    {/* Chat Image Thumbnail */}
                    {chat.image_url && (
                      <div className="shrink-0">
                        <img
                          src={chat.image_url}
                          alt="Chat image"
                          className="w-20 h-20 object-cover rounded-lg border border-cyan-500/30 shadow-lg"
                        />
                      </div>
                    )}
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="text-white font-medium truncate">
                          {chat.image_url ? "📸 Image Chat" : "💬 Text Chat"}
                        </h3>
                      </div>
                      <p className="text-gray-400 text-sm mb-3 line-clamp-2">{chat.preview}</p>
                      <div className="flex items-center gap-4 text-xs text-gray-500">
                        <span className="flex items-center gap-1.5">
                          <Calendar className="w-3.5 h-3.5" />
                          {formatDate(chat.created_at)}
                        </span>
                        <span className="flex items-center gap-1.5">
                          <MessageSquare className="w-3.5 h-3.5" />
                          {chat.messageCount} messages
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => handleDeleteChat(chat.id, e)}
                      disabled={deletingId === chat.id}
                      className="p-2 bg-red-500/10 text-red-400 rounded-lg hover:bg-red-500/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      aria-label="Delete chat"
                      title="Delete this chat"
                    >
                      {deletingId === chat.id ? (
                        <Loader className="w-4 h-4 animate-spin" />
                      ) : (
                        <Trash2 className="w-4 h-4" />
                      )}
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
