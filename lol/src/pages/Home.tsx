import { useState, useRef, useEffect } from "react";
import { useUser } from "@clerk/clerk-react";
import type { messageType } from "@/types/message";
import { Sidebar } from "../components/Sidebar";
import { TopBar } from "../components/TopBar";
import { ChatMessage } from "../components/ChatMessage";
import { ChatInput } from "../components/ChatInput";
import Starfield from "react-starfield";
import { ChatHistoryPage } from "./ChatHistoryPage";
import { chatService } from "@/services/chatService";
import { apiClient } from "@/lib/api";
import type { SampleData } from "./ExploreSamplesPage";
import { ExploreSamplesPage } from "./ExploreSamplesPage";
import Lottie from "lottie-react";
import { ProcessingIndicator } from "../components/ProcessingIndicator";
import groundingAnnotated from "../assets/sat_demo.jpg";
import radar from "@/assets/radar.json";
import { routes } from "@/lib/api";
import DropZone from "@/components/Dropzone";

export interface Message {
  id: string;
  type: "user" | "ai";
  content: string;
  image_url?: string;
  aiImage?: string;
}

export type Mode = "captioning" | "vqa" | "grounding";

export default function Home() {
  const { user } = useUser();
  const [currentPage, setCurrentPage] = useState<string>("chat");
    const [imageUploaded, setImageUploaded] = useState<boolean>(false);
  const [imgFile, setImgFile] = useState<File | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedImage,setSelectedImage]=useState<string|null>(null)
  const [mode, setMode] = useState<Mode>("vqa");
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const handleNavigation = (page: string) => {
    setCurrentPage(page);
  };

  const handleSendMessage = async (data: messageType) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: data.text,
      image_url: data.image_url,
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsProcessing(true);

    try {
      let sessionId = currentChatId;

      // Create a new session if one doesn't exist
      if (!sessionId) {
        if (!user?.id) {
          throw new Error("User must be authenticated to create a chat");
        }
        // Generate session ID (can be UUID from backend)
        sessionId = `session_${Date.now()}`;
        setCurrentChatId(sessionId);
      }

      // Send message to orchestrator endpoint
      const response = await chatService.sendMessage(
        sessionId,
        user.id,
        data.text,
        data.image_url || "", // Ensure image_url is always a string
        mode,
        true, // modalityDetectionEnabled
        false, // needsIr2rgb
        [], // ir2rgbChannels
        "B" // ir2rgbSynthesize
      );

      // Create AI message from response
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "ai",
        content: typeof response.content === "string" ? response.content : "No response from model",
        aiImage:
          mode === "grounding" &&
          typeof response.content !== "string" &&
          response.content?.boxes &&
          response.content.boxes.length > 0
            ? groundingAnnotated
            : undefined,
      };

      // Save AI message to state
      setMessages((prev) => [...prev, aiMessage]);

      // Backend persists messages via the orchestrator endpoint
    } catch (error) {
      console.error("Failed to send message:", error);
      const errorMessage: Message = {
        id: (Date.now() + 2).toString(),
        type: "ai",
        content: "Sorry, I encountered an error processing your request. Please try again.",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleNewChat = () => {
    setMessages([]);
    setMode("vqa"); // Reset to default
    setCurrentPage("chat");
    setCurrentChatId(null);
  };

  const handleSelectChat = async (sessionId: string) => {
    try {
      setIsProcessing(true);

      // Fetch messages for this session
      const historyResponse = await apiClient.get(`${routes.SESSION_HISTORY}/${sessionId}/history`);
      const messages = historyResponse.data?.messages || [];

      // Convert messages to Message format
      const loadedMessages: Message[] = messages.map((msg: any, index: number) => ({
        id: `${msg.id || index}`,
        type: msg.type === "user" ? "user" : "ai",
        content: msg.content,
        image_url: msg.image_url,
      }));

      // Set the loaded chat state
      setCurrentChatId(sessionId);
      setMessages(loadedMessages);
      setCurrentPage("chat");
    } catch (error) {
      console.error("Failed to load chat:", error);
      alert("Failed to load chat history");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleTrySample = (sample: SampleData) => {
    // Set the mode to match the sample
    if (sample.mode === "caption") {
      setMode("captioning");
    } else if (sample.mode === "question") {
      setMode("vqa");
    } else {
      setMode(sample.mode);
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: sample.mode === "caption" ? "" : sample.analysis,
      image_url: sample.imageUrl,
    };

    setMessages([userMessage]);
    setIsProcessing(true);
    setCurrentPage("chat");

    // Generate AI response after delay
    setTimeout(() => {
      let aiResponse = "";
      let aiImage: string | undefined;

      if (sample.mode === "caption") {
        aiResponse = `Caption Analysis:\n\n${sample.analysis}\n\nThis satellite imagery captures ${sample.description.toLowerCase()}. The visual patterns indicate distinct characteristics typical of ${sample.category} terrain, with clear evidence of the described features.`;
      } else if (sample.mode === "grounding") {
        // For grounding mode, respond to the detection/find query with annotated image
        aiResponse = `Object Detection Results:**\n\n**Query:** ${sample.analysis}\n\n**Detection Summary:**\n- Total objects detected: 2\n- Primary features: Aircraft\n- Detection confidence: 96.3%\n- Bounding boxes generated: ✓\n\n**Analysis:** Successfully located and mapped all requested objects in the satellite imagery. See annotated image below with detected objects highlighted.`;
        aiImage = groundingAnnotated;
      } else {
        // question mode
        aiResponse = ` **Visual Question Answering:**\n\n**Question:** ${sample.analysis}\n\n**Answer:** Based on the satellite imagery analysis, this ${sample.category} region shows ${sample.description.toLowerCase()}. The visual data suggests healthy patterns with characteristic features clearly visible in the spectral bands. The density and distribution indicate normal conditions for this terrain type.`;
      }

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "ai",
        content: aiResponse,
        aiImage: aiImage,
      };

      setMessages((prev) => [...prev, aiMessage]);
      setIsProcessing(false);
    }, 5000);
  };

  return (
    <div className="relative h-screen w-full overflow-hidden bg-[#000814]">
      {/*<SpaceBackground />*/}
      {/*<Stars/>*/}

      <Starfield
        starCount={4000}
        starColor={[255, 255, 255]}
        speedFactor={0.03}
        backgroundColor="black"
      />
      <div className="relative z-10 flex h-full">
        <Sidebar
          isOpen={isSidebarOpen}
          onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
          onLogoClick={handleNewChat}
          onNavigate={handleNavigation}
        />

        <div className="flex-1 flex flex-col min-w-0">
          <TopBar onMenuClick={() => setIsSidebarOpen(!isSidebarOpen)} />

          {currentPage === "chat-history" && <ChatHistoryPage onSelectChat={handleSelectChat} />}

          {currentPage === "explore" && <ExploreSamplesPage onTrySample={handleTrySample} />}

          {currentPage === "chat" && (
            <>
              {/* Main Chat Area - Dynamic height based on keyboard state */}
              <div
                className="flex-1 overflow-y-auto px-4 md:px-8 py-6 scrollbar-thin scrollbar-thumb-orange-500/20 scrollbar-track-transparent transition-all duration-300"
                style={{}}
              >
                <div className="max-w-3xl mx-auto h-full flex flex-col space-y-6 pb-4">
                  {/* Empty State: Show Logo Center */}
                  {messages.length === 0 ? (
                    <div className="flex-auto flex flex-col items-center justify-center ">
                      <div className="relative ">
                        <Lottie
                          animationData={radar}
                          loop={true}
                          className=" w-100 pointer-events-none"
                        />
                      </div>
                      <p className="text-cyan-400/80 text-center max-w-lg mt-4 animate-in fade-in slide-in-from-bottom-4 duration-700">
                        Unlock the language of satellite imagery. <br />
                        <span className="text-sm opacity-60">
                          Type{" "}
                          <span className="bg-cyan-900/40 border border-cyan-700/50 px-1.5 rounded text-cyan-200">
                            /
                          </span>{" "}
                          to switch modes instantly.
                        </span>
                      </p>
                    </div>
                  ) : (
                    /* Message List */
                    <>
                      {messages.map((message) => (
                        <ChatMessage key={message.id} message={message} />
                      ))}
                      {isProcessing && <ProcessingIndicator mode={mode} />}
                      {/* Scroll anchor */}
                      <div ref={messagesEndRef} />
                    </>
                  )}
                </div>
              </div>

              <div
                className="fixed bottom-0 left-0 right-0 md:relative md:bottom-auto p-4 md:p-6 bg-linear-to-t from-[#000814] via-[#000814] to-transparent z-30 transition-all duration-300"
                style={{
                  bottom: "0",
                }}
              >
                {!imageUploaded && <DropZone setimageUploaded={setImageUploaded} selectedImage={selectedImage} setSelectedImage={setSelectedImage} setImgFile={setImgFile} />}
                <div
                  className="max-w-3xl mx-auto"
                  style={{ marginLeft: isSidebarOpen ? "auto" : "auto" }}
                >
                  <ChatInput
                    onSend={handleSendMessage}
                    isProcessing={isProcessing}
                    mode={mode}
                    setMode={setMode}
                    selectedImage={selectedImage}
                    setSelectedImage={setSelectedImage}
                    onNewChat={handleNewChat}
                    setImageUploaded={setImageUploaded}
                  />
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
