import { useState, useRef, useEffect } from "react";
import { useUser } from "@clerk/clerk-react";
import {
  Send,
  Image as ImageIcon,
  Sparkles,
  Target,
  MessageSquare,
  X,
  Mic,
  MicOff,
  Upload,
} from "lucide-react";
import type { Mode } from "../pages/Home";
import { BACKEND_URL, ROUTES } from "@/lib/constant";

// TypeScript definition for the Web Speech API
declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}

interface ChatInputProps {
  onSend: (data: { text: string; image_url?: string }, userId?: string) => void;
  isProcessing: boolean;
  mode: Mode;
  setMode: (mode: Mode) => void;
  onFocusChange?: (isFocused: boolean) => void;
  onNewChat?: () => void;
}

export function ChatInput({
  onSend,
  isProcessing,
  mode,
  setMode,
  onFocusChange,
  onNewChat,
}: ChatInputProps) {
  const { user } = useUser();
  const [content, setContent] = useState("");
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [imgFile, setImgFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showSlashMenu, setShowSlashMenu] = useState(false);
  const [isListening, setIsListening] = useState(false);

  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<any>(null);

  // --- Voice Input Logic ---
  const toggleVoiceInput = () => {
    if (isListening) {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
        setIsListening(false);
      }
    } else {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) {
        alert("Your browser does not support voice input.");
        return;
      }
      const recognition = new SpeechRecognition();
      recognition.lang = "en-US";
      recognition.continuous = false;
      recognition.interimResults = false;

      recognition.onstart = () => setIsListening(true);
      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setContent((prev) => (prev ? `${prev} ${transcript}` : transcript));
        setTimeout(() => {
          if (inputRef.current) {
            inputRef.current.style.height = "auto";
            inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 200)}px`;
          }
        }, 10);
      };
      recognition.onerror = () => setIsListening(false);
      recognition.onend = () => setIsListening(false);

      recognitionRef.current = recognition;
      recognition.start();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setContent(value);

    if (value === "/") {
      setShowSlashMenu(true);
    } else {
      setShowSlashMenu(false);
    }

    e.target.style.height = "auto";
    e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];

      // Only accept images
      if (!file.type.startsWith("image/")) {
        console.warn("Selected file is not an image.");
        e.target.value = "";
        return;
      }

      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result as string);
        setImgFile(file); // <--- make sure we set the File object used by uploadImage
      };
      reader.readAsDataURL(file);

      // clear input so same-file re-selects will trigger onChange again
      e.target.value = "";
    }
  };

  const readFileAsDataUrl = (file: File) => {
    return new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const handleFileObject = async (file: File) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) return;
    try {
      const data = await readFileAsDataUrl(file);
      setSelectedImage(data);
      setImgFile(file);
    } catch (err) {
      console.error("Failed to read dropped file", err);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = "copy";
    setIsDragging(true);
  };

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileObject(files[0]);
    }
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const clipboard = e.clipboardData;
    if (!clipboard) return;
    const items = clipboard.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) {
          e.preventDefault();
          handleFileObject(file);
          break;
        }
      }
    }
  };

  const handleRemoveImage = () => {
    setSelectedImage(null);
  };

  const uploadImage = async (userId?: string) => {
    if (!imgFile) {
      return;
    }

    const formData = new FormData();
    formData.append("file", imgFile);

    try {
      const url = new URL(`${BACKEND_URL}${ROUTES.IMAGE_UPLOAD}`);
      if (userId) {
        url.searchParams.append("user_id", userId);
      }
      const res = await fetch(url.toString(), {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (data.image_url) {
        return data.image_url;
      }
      if (data.file_url) {
        return data.file_url;
      }
    } catch (error) {
      console.error("Image upload failed:", error);
    }
  };
  const handleSendClick = async () => {
    if ((content.trim() || selectedImage) && !isProcessing) {
      const image_url = selectedImage ? await uploadImage(user?.id) : undefined;
      console.log(image_url);
      const data = {
        text: content,
        image_url: image_url,
      };
      onSend(data, user?.id);

      setContent("");
      setSelectedImage(null);
      setImgFile(null);
      setShowSlashMenu(false);
      if (inputRef.current) inputRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendClick();
    }
    if (e.key === "Escape") {
      setShowSlashMenu(false);
    }
  };

  const selectMode = (newMode: Mode, e?: React.MouseEvent) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    setMode(newMode);
    setShowSlashMenu(false);
    setContent("");
    inputRef.current?.focus();
  };

  const handleSlashUpload = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setShowSlashMenu(false);
    setContent("");
    fileInputRef.current?.click();
  };

  useEffect(() => {
    const currentRef = inputRef.current;
    if (currentRef) {
      const handleFocus = () => {
        if (onFocusChange) {
          onFocusChange(true);
        }
      };
      const handleBlur = () => {
        if (onFocusChange) {
          onFocusChange(false);
        }
      };
      currentRef.addEventListener("focus", handleFocus);
      currentRef.addEventListener("blur", handleBlur);
      return () => {
        currentRef.removeEventListener("focus", handleFocus);
        currentRef.removeEventListener("blur", handleBlur);
      };
    }
  }, [onFocusChange]);

  return (
    <div
      className="relative group"
      onDragOver={handleDragOver}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* --- Inject Styles for Wave Animation --- */}
      <style>{`
        @keyframes sound-wave {\n          0%, 100% { height: 4px; opacity: 0.5; }
          50% { height: 16px; opacity: 1; }
        }
        .animate-wave {
          animation: sound-wave 1s ease-in-out infinite;
        }
      `}</style>

      {/* Hidden File Input */}
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        accept="image/*"
        onChange={handleFileSelect}
      />

      {/* Slash Menu */}
      {showSlashMenu && (
        <>
          <div
            className="fixed inset-0 z-40 bg-transparent"
            onClick={() => setShowSlashMenu(false)}
          />
          <div className="absolute bottom-full left-0 mb-2 w-64 bg-[#0a101f] border border-cyan-800/30 rounded-lg shadow-2xl shadow-cyan-900/40 overflow-hidden backdrop-blur-xl z-50 animate-in slide-in-from-bottom-2 fade-in duration-200">
            <div className="px-3 py-2 text-xs font-semibold text-blue-400 group-hover/item:text-blue-300">
              Options
            </div>
            <button
              onMouseDown={(e) => selectMode("question", e)}
              className="w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-cyan-500/10 transition-colors group/item cursor-pointer"
            >
              <div className="p-1.5 rounded bg-blue-500/20 text-blue-400 group-hover/item:text-blue-300">
                <MessageSquare size={16} />
              </div>
              <div>
                <div className="text-cyan-100 text-sm font-medium">Visual Q</div>
                <div className="text-cyan-500/50 text-xs">General chat & analysis</div>
              </div>
            </button>
            <button
              onMouseDown={(e) => selectMode("caption", e)}
              className="w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-cyan-500/10 transition-colors group/item cursor-pointer"
            >
              <div className="p-1.5 rounded bg-blue-500/20 text-blue-400 group-hover/item:text-blue-300">
                <ImageIcon size={16} />
              </div>
              <div>
                <div className="text-cyan-100 text-sm font-medium">Caption</div>
                <div className="text-cyan-500/50 text-xs">Describe imagery</div>
              </div>
            </button>
            <button
              onMouseDown={(e) => selectMode("grounding", e)}
              className="w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-cyan-500/10 transition-colors group/item cursor-pointer"
            >
              <div className="p-1.5 rounded bg-blue-500/20 text-blue-400 group-hover/item:text-blue-300">
                <Target size={16} />
              </div>
              <div>
                <div className="text-cyan-100 text-sm font-medium">Grounding</div>
                <div className="text-cyan-500/50 text-xs">Locate specific objects</div>
              </div>
            </button>
            <div className="h-px bg-cyan-800/30 my-1 mx-2"></div>
            <button
              onMouseDown={handleSlashUpload}
              className="w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-cyan-500/10 transition-colors group/item cursor-pointer"
            >
              <div className="p-1.5 rounded bg-blue-500/20 text-blue-400 group-hover/item:text-blue-300">
                <Upload size={16} />
              </div>
              <div>
                <div className="text-cyan-100 text-sm font-medium">Upload Image</div>
                <div className="text-cyan-500/50 text-xs">Attach file from device</div>
              </div>
            </button>
          </div>
        </>
      )}

      {/* Main Input Container */}
      <div
        className={`
        relative bg-[#0d1b2e]/95 backdrop-blur-md border transition-all duration-300 rounded-2xl overflow-hidden z-30 shadow-xl shadow-black/50
        ${isProcessing ? "border-cyan-500/50 opacity-80" : "border-cyan-700/50 hover:border-cyan-500/70 focus-within:border-cyan-400/90 focus-within:ring-2 focus-within:ring-cyan-500/30"}
        ${isListening ? "ring-2 ring-cyan-500/60 border-cyan-500/70 shadow-[0_0_25px_rgba(34,211,238,0.25)]" : ""}
      `}
      >
        <div className="flex flex-col">
          {/* Image Preview */}
          {selectedImage && (
            <div className="px-4 pt-4 pb-0">
              <div className="relative inline-block">
                <img
                  src={selectedImage}
                  alt="Selected"
                  className="h-16 w-16 object-cover rounded-lg border border-cyan-500/30 shadow-lg"
                />
                <button
                  onClick={handleRemoveImage}
                  className="absolute -top-2 -right-2 bg-black/80 text-white rounded-full p-0.5 border border-cyan-500/50 hover:bg-red-500/20 hover:border-red-500 transition-colors"
                >
                  <X size={12} />
                </button>
              </div>
            </div>
          )}

          <div className="relative">
            {/* --- Textarea --- */}
            <textarea
              ref={inputRef}
              value={content}
              onChange={handleInput}
              onPaste={handlePaste}
              onKeyDown={handleKeyDown}
              placeholder={
                isListening
                  ? ""
                  : mode === "grounding"
                    ? "Describe objects to locate..."
                    : "Ask anything about the satellite data..."
              }
              rows={1}
              className="w-full bg-transparent text-cyan-100 placeholder-cyan-700/50 px-4 py-4 focus:outline-none resize-none scrollbar-none min-h-14 text-[15px] relative z-10"
              disabled={isProcessing}
            />

            {/* --- Listening Animation Overlay (Cyan Theme) --- */}
            {isListening && !content && (
              <div className="absolute top-0 left-0 w-full h-full flex items-center px-4 pointer-events-none z-0">
                <div className="flex items-center gap-2">
                  {/* The Wave Animation Bars (Cyan) */}
                  <div className="flex items-center gap-0.5 h-4">
                    <div
                      className="w-1 bg-cyan-400 rounded-full animate-wave"
                      style={{ animationDelay: "0ms" }}
                    ></div>
                    <div
                      className="w-1 bg-cyan-400 rounded-full animate-wave"
                      style={{ animationDelay: "100ms" }}
                    ></div>
                    <div
                      className="w-1 bg-cyan-400 rounded-full animate-wave"
                      style={{ animationDelay: "200ms" }}
                    ></div>
                    <div
                      className="w-1 bg-cyan-400 rounded-full animate-wave"
                      style={{ animationDelay: "150ms" }}
                    ></div>
                    <div
                      className="w-1 bg-cyan-400 rounded-full animate-wave"
                      style={{ animationDelay: "50ms" }}
                    ></div>
                  </div>
                  <span className="text-cyan-400/80 text-sm font-medium animate-pulse">
                    Listening...
                  </span>
                </div>
              </div>
            )}

            {/* Drag & Drop Overlay */}
            {isDragging && (
              <div className="absolute inset-0 z-50 flex items-center justify-center pointer-events-none">
                <div className="w-full h-full rounded-2xl bg-black/40 backdrop-blur-sm border-2 border-dashed border-cyan-400/60 flex items-center justify-center">
                  <div className="text-center text-cyan-200/90">
                    <div className="text-lg text-white font-semibold">Drop image to upload</div>
                    <div className="text-sm text-cyan-300/80">Or paste an image from clipboard</div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Action Buttons Row */}
          <div className="flex justify-between items-center px-3 pb-3 pt-1">
            {/* Mode Tabs - Visible only on larger screens (md and up) */}
            <div className="hidden md:flex items-center gap-1.5">
              <ModePill
                active={mode === "question"}
                icon={<MessageSquare size={13} />}
                label="VQA"
                onClick={() => setMode("question")}
              />
              <ModePill
                active={mode === "caption"}
                icon={<Sparkles size={13} />}
                label="Caption"
                onClick={() => setMode("caption")}
              />
              <ModePill
                active={mode === "grounding"}
                icon={<Target size={13} />}
                label="Ground"
                onClick={() => setMode("grounding")}
              />
            </div>

            {/* Action Buttons - Record, Image, New Chat, Send */}
            <div className="flex items-center gap-2 md:ml-auto w-full md:w-auto justify-end">
              {/* Voice Input Button (Cyan Theme active state) */}
              <button
                onClick={toggleVoiceInput}
                className={`
                            p-2 transition-all duration-300 rounded-lg
                            ${
                              isListening
                                ? "bg-cyan-500/20 text-cyan-400 animate-pulse shadow-[0_0_10px_rgba(34,211,238,0.3)]"
                                : "text-cyan-600 hover:text-cyan-300 hover:bg-cyan-900/20"
                            }
                        `}
                title="Voice Input"
              >
                {isListening ? <MicOff size={18} /> : <Mic size={18} />}
              </button>

              <button
                onClick={() => fileInputRef.current?.click()}
                className={`p-2 transition-colors rounded-lg hover:bg-cyan-900/20 ${selectedImage ? "text-cyan-400" : "text-cyan-600 hover:text-cyan-300"}`}
                title="Upload Image"
              >
                <ImageIcon size={18} />
              </button>

              {/* New Chat Button */}
              {/*{onNewChat && (
                <button
                  onClick={onNewChat}
                  className="px-3 py-2 text-xs font-medium text-cyan-300 bg-cyan-900/30 hover:bg-cyan-900/50 border border-cyan-700/50 hover:border-cyan-500/70 rounded-lg transition-all duration-300"
                  title="Start a new chat"
                >
                  New Chat
                </button>
              )}*/}

              <button
                onClick={handleSendClick}
                disabled={(!content.trim() && !selectedImage) || isProcessing}
                className={`
                        p-2 rounded-lg transition-all duration-300
                        ${
                          (content.trim() || selectedImage) && !isProcessing
                            ? "bg-cyan-500 text-black shadow-lg shadow-cyan-500/20 hover:bg-cyan-400 transform hover:scale-105"
                            : "bg-cyan-900/20 text-cyan-800 cursor-not-allowed"
                        }
                        `}
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Mode Tabs - Floating below chat box, visible only on smaller screens (below md) */}
      <div className="mt-4 flex justify-center md:hidden">
        <div className="inline-flex items-center gap-1.5 bg-linear-to-r from-cyan-500/20 to-blue-600/20 backdrop-blur-md border border-cyan-500/40 rounded-full p-1 shadow-lg shadow-cyan-500/30">
          <ModePill
            active={mode === "question"}
            icon={<MessageSquare size={13} />}
            label="VQA"
            onClick={() => setMode("question")}
          />
          <ModePill
            active={mode === "caption"}
            icon={<Sparkles size={13} />}
            label="Caption"
            onClick={() => setMode("caption")}
          />
          <ModePill
            active={mode === "grounding"}
            icon={<Target size={13} />}
            label="Ground"
            onClick={() => setMode("grounding")}
          />
        </div>
      </div>
    </div>
  );
}

function ModePill({
  active,
  icon,
  label,
  onClick,
}: {
  active: boolean;
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`
                flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-200 border
                ${
                  active
                    ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/30 shadow-[0_0_10px_rgba(34,211,238,0.15)]"
                    : "bg-transparent text-cyan-600/70 border-transparent hover:bg-cyan-900/20 hover:text-cyan-400"
                }
            `}
    >
      {icon}
      {label}
    </button>
  );
}
