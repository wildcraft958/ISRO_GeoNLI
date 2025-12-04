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
import type { SampleData } from "./ExploreSamplesPage";
import { ExploreSamplesPage } from "./ExploreSamplesPage";
import Lottie from "lottie-react";
import { ProcessingIndicator } from "../components/ProcessingIndicator";
import groundingAnnotated from "../assets/sat_demo.jpg";
import radar from "@/assets/radar.json";
import DropZone from "@/components/Dropzone";

export interface Message {
  id: string;
  type: "user" | "ai";
  content: any;
  image_url?: string;
  aiImage?: string;
  boxes?: Array<{ x1: number; y1: number; x2: number; y2: number; label?: string; confidence?: number }>;
  imageWidth?: number;
  imageHeight?: number;
}

export type Mode = "captioning" | "vqa" | "grounding"|"auto";

export default function Home() {
  const { user } = useUser();
  const [currentPage, setCurrentPage] = useState<string>("chat");
    const [imageUploaded, setImageUploaded] = useState<boolean>(false);
  const [imgFile, setImgFile] = useState<File | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedImage,setSelectedImage]=useState<string|null>(null)
  const [mode, setMode] = useState<Mode>("auto");
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [lastQueryId, setLastQueryId] = useState<string | null>(null); // New state for last query ID
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const handleNavigation = (page: string) => {
    setCurrentPage(page);
  };

  const handleSendMessage = async (
    data: messageType,
    userId?: string, // Added userId param from ChatInput
    messageCategory?: "chat" | "query" // Added messageCategory param from ChatInput
  ) => {
    const isNewChat = !currentChatId; // Check if this is a new chat (parent_id will be null)
    
    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: data.text,
      // Only show image for the first query (when creating a new chat, parent_id will be null)
      image_url: isNewChat ? data.image_url : undefined,
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsProcessing(true);

    try {
      let sessionId = currentChatId;
      let queryId: string | null = lastQueryId;

      if (!user?.id) {
        throw new Error("User must be authenticated to create a chat/query");
      }

      // If no session exists, create a new one and an initial query
      if (!sessionId) {
        // Always create a chat session first, whether it's image or text based.
        const createChatResponse = await chatService.createChat(
          user.id,
          data.image_url || undefined // Pass image_url if present, else undefined
        );
        const chatId = createChatResponse.id; // Backend returns 'id' not 'chat_id'
        sessionId = chatId; // Use chat_id as session_id for orchestrator
        setCurrentChatId(chatId);

        // Create the initial query for the new chat
        const createQueryResponse = await chatService.createQuery(
          null, // parent_id is null for the first query in a chat
          chatId, // Pass the chat_id to the query
          data.text,
          null, // response is null initially for the user's query
          messageCategory === "chat" ? "image_query" : "text_query", // Type based on presence of image
          mode // Mode of the current chat
        );
        console.log("createqueryeresponse", createQueryResponse);
        queryId = createQueryResponse.id; // Backend returns id as string
        setLastQueryId(queryId);
      } else {
        // If it's a subsequent text-only query in an existing session
        // sessionId is the chat_id in this context
        const createQueryResponse = await chatService.createQuery(
          lastQueryId, // parent_id is the previous query ID
          sessionId, // Pass the chat_id (sessionId) to the query
          data.text,
          null,
          "auto",
          mode
        );
        queryId = createQueryResponse.id; // Backend returns id as string
        setLastQueryId(queryId);
      }

      // Send message to orchestrator endpoint (this is the main AI processing)
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
      const aiResponseContent = response.content ?? "No response from model";
      
      // Check if response is a string or JSON with boxes
      let displayContent: string;
      if (typeof aiResponseContent === "string") {
        displayContent = aiResponseContent;
      } else if (aiResponseContent && typeof aiResponseContent === "object" && "boxes" in aiResponseContent) {
        // If it's JSON with boxes, console.log it instead of displaying
        console.log("Response with boxes:", aiResponseContent);
        displayContent = ""; // Don't display anything for boxes response
      } else {
        // Fallback for other object types
        displayContent = JSON.stringify(aiResponseContent);
      }
      
      // Extract boxes and image dimensions if present
      let boxes: Array<{ x1: number; y1: number; x2: number; y2: number; label?: string; confidence?: number }> | undefined;
      let imageWidth: number | undefined;
      let imageHeight: number | undefined;
      let aiImageUrl: string | undefined;
      
      if (mode === "grounding" &&
          typeof response.content !== "string" &&
          response.content?.boxes &&
          response.content.boxes.length > 0) {
        boxes = response.content.boxes;
        imageWidth = response.content.image_width;
        imageHeight = response.content.image_height;
        // Use the original image URL if available, otherwise use the grounding annotated image
        aiImageUrl = response.original_image_url || groundingAnnotated;
      }
      
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "ai",
        content: displayContent,
        aiImage: aiImageUrl,
        boxes: boxes,
        imageWidth: imageWidth,
        imageHeight: imageHeight,
      };

      // Save AI message to state
      setMessages((prev) => [...prev, aiMessage]);

      // Update the query with the AI response
      if (queryId) {
        await chatService.updateQueryResponse(queryId, response.content);
      }
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
    setLastQueryId(null); // Reset last query ID
    // Reset image state to prompt for image upload
    setSelectedImage(null);
    setImageUploaded(false);
    setImgFile(null);
  };

  const handleSelectChat = async (chatId: string) => {
    try {
      setIsProcessing(true);

      // Fetch the chat details to get image_url
      const chats = await chatService.getChats(user?.id || "");
      const selectedChat = chats.find((chat: any) => chat.id === chatId);
      
      // When loading a chat from history, don't set selectedImage for sending
      // The image will be displayed in the first message (where parent_id = null)
      // but won't be sent with new messages
      setSelectedImage(null);
      setImageUploaded(true); // Enable input for continuing the conversation

      // Fetch queries for this chat to build messages
      const queries = await chatService.getChatQueries(chatId);
      
      // Ensure queries is an array
      if (!Array.isArray(queries)) {
        console.error("Expected array but got:", queries);
        throw new Error("Invalid response format: queries is not an array");
      }
      
      // Build messages from queries
      // Each query represents a user question (request) and AI response (response)
      const loadedMessages: Message[] = [];
      
      queries.forEach((query: any) => {
        // Check if this is the first query (parent_id is null or empty)
        const isFirstQuery = !query.parent_id || query.parent_id === "" || query.parent_id === null;
        
        // Add user message (request)
        if (query.request) {
          loadedMessages.push({
            id: `${query.id}_user`,
            type: "user",
            content: query.request,
            image_url: isFirstQuery && selectedChat?.image_url ? selectedChat.image_url : undefined, // Only first query (parent_id = null) gets the image
          });
        }
        
        // Add AI message (response) if it exists
        if (query.response) {
          // Check if response is a string or object with boxes
          let responseContent: any = query.response;
          
          // Try to parse if it's a stringified JSON
          if (typeof responseContent === "string") {
            try {
              responseContent = JSON.parse(responseContent);
            } catch (e) {
              // If parsing fails, it's a regular string
              responseContent = query.response;
            }
          }
          
          // Extract display content and boxes data
          let displayContent: string;
          let boxes: Array<{ x1: number; y1: number; x2: number; y2: number; label?: string; confidence?: number }> | undefined;
          let imageWidth: number | undefined;
          let imageHeight: number | undefined;
          let aiImageUrl: string | undefined;
          
          if (typeof responseContent === "string") {
            displayContent = responseContent;
          } else if (responseContent && typeof responseContent === "object" && "boxes" in responseContent) {
            // Response has boxes (grounding response)
            console.log("Loading response with boxes from history:", responseContent);
            displayContent = ""; // Don't display text for boxes response
            boxes = responseContent.boxes;
            imageWidth = responseContent.image_width;
            imageHeight = responseContent.image_height;
            // Use the chat's image_url for displaying boxes
            aiImageUrl = selectedChat?.image_url || undefined;
          } else {
            // Other object types
            displayContent = JSON.stringify(responseContent);
          }
          
          loadedMessages.push({
            id: `${query.id}_ai`,
            type: "ai",
            content: displayContent,
            aiImage: aiImageUrl,
            boxes: boxes,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
          });
        }
      });

      // Set the last query ID for continuing the conversation
      if (queries.length > 0) {
        const lastQuery = queries[queries.length - 1];
        setLastQueryId(lastQuery.id);
      } else {
        setLastQueryId(null);
      }

      // Set the loaded chat state
      setCurrentChatId(chatId);
      setMessages(loadedMessages);
      setCurrentPage("chat");
    } catch (error) {
      console.error("Failed to load chat:", error);
      alert("Failed to load chat history");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleTrySample = async (sample: SampleData) => {
    // Reset chat state to start a new chat
    setCurrentChatId(null);
    setLastQueryId(null);
    setMessages([]);
    
    // Set the mode to match the sample
    let sampleMode: Mode = "vqa";
    if (sample.mode === "caption") {
      sampleMode = "captioning";
    } else if (sample.mode === "question") {
      sampleMode = "vqa";
    } else if (sample.mode === "grounding") {
      sampleMode = "grounding";
    }
    else if (sample.mode === "auto") {
      sampleMode = "auto";
    }
    setMode(sampleMode);
    
    // Set the page to chat
    setCurrentPage("chat");
    
    // Prepare the message data
    const messageText = sample.mode === "caption" ? "" : sample.analysis;
    
    // Create user message for display (with image for first query)
    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: messageText,
      image_url: sample.imageUrl, // Image will be shown in first message
    };
    
    setMessages([userMessage]);
    setIsProcessing(true);

    try {
      if (!user?.id) {
        throw new Error("User must be authenticated to create a chat/query");
      }

      // Create a new chat with the sample image
      const createChatResponse = await chatService.createChat(
        user.id,
        sample.imageUrl
      );
      const chatId = createChatResponse.id;
      setCurrentChatId(chatId);

      // Create the initial query for the new chat (parent_id = null for first query)
      const createQueryResponse = await chatService.createQuery(
        null, // parent_id is null for the first query
        chatId,
        messageText,
        null, // response is null initially
        sample.mode === "caption" ? "image_query" : "text_query",
        sampleMode
      );
      const queryId = createQueryResponse.id;
      setLastQueryId(queryId);

      // Send message to orchestrator endpoint
      const response = await chatService.sendMessage(
        chatId,
        user.id,
        messageText,
        sample.imageUrl,
        sampleMode,
        true, // modalityDetectionEnabled
        false, // needsIr2rgb
        [], // ir2rgbChannels
        "B" // ir2rgbSynthesize
      );

      console.log("response", response);

      // Create AI message from response
      const aiResponseContent = response.content ?? "No response from model";
      
      // Check if response is a string or JSON with boxes
      let displayContent: string;
      if (typeof aiResponseContent === "string") {
        displayContent = aiResponseContent;
      } else if (aiResponseContent && typeof aiResponseContent === "object" && "boxes" in aiResponseContent) {
        // If it's JSON with boxes, console.log it instead of displaying
        console.log("Response with boxes:", aiResponseContent);
        displayContent = ""; // Don't display anything for boxes response
      } else {
        // Fallback for other object types
        displayContent = JSON.stringify(aiResponseContent);
      }
      
      // Extract boxes and image dimensions if present
      let boxes: Array<{ x1: number; y1: number; x2: number; y2: number; label?: string; confidence?: number }> | undefined;
      let imageWidth: number | undefined;
      let imageHeight: number | undefined;
      let aiImageUrl: string | undefined;
      
      if (sampleMode === "grounding" &&
          typeof response.content !== "string" &&
          response.content?.boxes &&
          response.content.boxes.length > 0) {
        boxes = response.content.boxes;
        imageWidth = response.content.image_width;
        imageHeight = response.content.image_height;
        // Use the original image URL if available, otherwise use the grounding annotated image
        aiImageUrl = response.original_image_url || sample.imageUrl || groundingAnnotated;
      }
      
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "ai",
        content: displayContent,
        aiImage: aiImageUrl,
        boxes: boxes,
        imageWidth: imageWidth,
        imageHeight: imageHeight,
      };

      // Save AI message to state
      setMessages((prev) => [...prev, aiMessage]);

      // Update the query with the AI response
      if (queryId) {
        await chatService.updateQueryResponse(queryId, response.content);
      }
      
      // Clear image state after first query is sent
      // This ensures subsequent messages won't include the image
      setSelectedImage(null);
      setImageUploaded(true); // Keep input enabled for continuing the conversation
    } catch (error) {
      console.error("Failed to process sample:", error);
      const errorMessage: Message = {
        id: (Date.now() + 2).toString(),
        type: "ai",
        content: "Sorry, I encountered an error processing the sample. Please try again.",
      };
      setMessages((prev) => [...prev, errorMessage]);
      // Clear image state even on error
      setSelectedImage(null);
      setImageUploaded(true);
    } finally {
      setIsProcessing(false);
    }
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
                    imageUploaded={imageUploaded}
                    imgFile={imgFile}
                    setImgFile={setImgFile}
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
