import {
  ImageUp,
  History,
  Globe,
  Settings,
  ChevronLeft,
  LogOut,
  MessageSquare,
  Info,
  PanelLeftClose,
  PanelLeft,
  Sparkles,
} from "lucide-react";
import { Button } from "./ui/button";
import { OrbitLogo } from "./OrbitLogo";
import logo from "@/assets/logo.png";
interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  onLogoClick: () => void;
  onLogout?: () => void;
  onNavigate?: (page: string) => void;
  className?: string;
}

export function Sidebar({ isOpen, onToggle, onLogoClick, onNavigate, className }: SidebarProps) {
  const menuItems = [
    // PRIMARY INTERACTION
    { icon: MessageSquare, label: "New chat", page: "chat", action: "new-chat" },
    { icon: History, label: "Chat History", page: "chat-history" },

    // UTILITY
    { icon: Sparkles, label: "Explore Samples", page: "explore" },
  ];

  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-md z-40 md:hidden"
          onClick={onToggle}
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed md:relative z-50 h-full transition-all duration-500 cubic-bezier(0.32, 0.72, 0, 1) ${
          isOpen ? "translate-x-full" : "-translate-x-full md:translate-x-0"
        } ${isOpen ? "w-[14rem]" : "w-0 md:w-[4rem]"} ${className || ""}`}
      >
        {/* Main Glass Layer */}

        <div className="h-full flex flex-col relative overflow-hidden bg-[#0d1b2e]/95 backdrop-blur-3xl border-r border-cyan-700/40 shadow-2xl shadow-black/50">
          {/* --- AMBIENT GLOW EFFECTS --- */}
          <img
            src={logo}
            className={`${isOpen ? "w-28" : "w-0"} ml-4 my-4 p-8 transition-all duration-300 z-20`}
            alt="Logo"
          />
          <div className="absolute -top-24 -left-24 w-64 h-64 bg-cyan-500/20 rounded-full blur-[80px] pointer-events-none" />
          <div className="absolute -bottom-24 -right-24 w-64 h-64 bg-blue-600/15 rounded-full blur-[80px] pointer-events-none" />
          <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-cyan-400/30 to-transparent opacity-70" />

          {/* --- HEADER --- */}
          <div className="pt-[5px] pb-0 px-3 relative z-10">
            {/* Desktop Toggle Button - Top Right when open */}
            {isOpen && (
              <button
                onClick={onToggle}
                className="hidden md:flex absolute right-4 top-4 w-8 h-8 bg-white/5 hover:bg-cyan-500/20 border border-white/10 hover:border-cyan-500/40 rounded-lg items-center justify-center text-gray-400 hover:text-cyan-400 transition-all duration-300 group z-50"
                title="Collapse sidebar"
              >
                <PanelLeftClose className="w-4 h-4" />
              </button>
            )}

            <div
              className={`flex items-center justify-start transition-all duration-500 ${isOpen ? "h-[80px]" : "h-[80px]"}`}
            >
              {/* LOGO LINK or Toggle when collapsed */}
              {isOpen ? (
                <button
                  onClick={onLogoClick}
                  className="hidden sm:block
 cursor-pointer  transition-transform duration-150 hover:scale-[0.9] focus:outline-none"
                >
                  <OrbitLogo />
                </button>
              ) : (
                <button
                  onClick={onToggle}
                  className="hidden md:flex w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-600/20 border border-cyan-500/30 hover:from-cyan-500/30 hover:to-blue-600/30 hover:border-cyan-400/50 items-center justify-center text-cyan-400 hover:text-cyan-300 transition-all duration-300 hover:scale-110 shadow-lg shadow-cyan-500/20 hover:shadow-cyan-500/40"
                  title="Expand sidebar"
                >
                  <PanelLeft className="w-5 h-5" />
                </button>
              )}

              {/* Mobile Close Button */}
              <Button
                variant="ghost"
                size="icon"
                onClick={onToggle}
                className="md:hidden absolute right-3 text-white/50 hover:bg-white/10 hover:text-white rounded-full transition-colors"
              >
                <ChevronLeft className="w-5 h-5" />
              </Button>
            </div>
          </div>

          {/* --- NAVIGATION --- */}
          <nav className="flex-1 px-4 mt-0.5 space-y-2 overflow-y-auto z-10">
            <div className="flex flex-col gap-2">
              {menuItems.map((item, index) => (
                <button
                  key={index}
                  className={`
                    cursor-pointer
                    group relative flex items-center ${isOpen ? "gap-4 px-4" : "justify-center px-0"} py-3.5 rounded-2xl
                    text-sm font-medium tracking-wide text-zinc-400
                    border border-transparent
                    transition-all duration-300 ease-out
                    hover:text-white
                    hover:bg-white/[0.05]
                    hover:border-blue-500/30
                    hover:shadow-[0_0_20px_-5px_rgba(59,130,246,0.4)]
                    active:scale-[0.98]
                  `}
                  onClick={() => {
                    if (item.action === "new-chat") {
                      onLogoClick();
                    } else {
                      onNavigate && onNavigate(item.page);
                    }
                  }}
                  title={!isOpen ? item.label : ""}
                >
                  <div className="relative flex items-center justify-center">
                    <item.icon className="w-5 h-5 text-cyan-600 group-hover:text-cyan-300 transition-colors duration-300" />
                    <div className="absolute inset-0 bg-blue-400/20 blur-md rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                  </div>

                  {isOpen && (
                    <>
                      <span className="relative">{item.label}</span>
                    </>
                  )}
                </button>
              ))}
            </div>
          </nav>
        </div>
      </aside>
    </>
  );
}
