import { Search, Menu, User } from "lucide-react";
import { Button } from "./ui/button";
import { Avatar, AvatarFallback } from "../components/ui/avatar";
import { SignIn, SignInButton, UserButton } from "@clerk/clerk-react";

interface TopBarProps {
  onMenuClick: () => void;
}

export function TopBar({ onMenuClick }: TopBarProps) {
  return (
    <header className="sticky top-0 z-20 w-full border-b border-blue-500/10 bg-[#001020]/95 backdrop-blur-md">
      <div className="flex items-center justify-between px-4 md:px-8 py-4">
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={onMenuClick}
            className="md:hidden text-blue-400 hover:bg-blue-500/10"
          >
            <Menu className="w-5 h-5" />
          </Button>

          {/* Animated Satellite Indicator */}
          <div className="hidden md:flex items-center gap-3">
            <div className="relative">
              <div className="w-8 h-8 relative">
                {/* Orbit ring */}
                <div
                  className="absolute inset-0 border-2 border-cyan-400/30 rounded-full animate-spin"
                  style={{ animationDuration: "3s" }}
                >
                  <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-cyan-400 rounded-full shadow-lg shadow-cyan-400/50"></div>
                </div>
                {/* Center Earth */}
                <div className="absolute inset-2 bg-gradient-to-br from-blue-500 to-green-500 rounded-full"></div>
              </div>
            </div>
            <div></div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <Avatar className="w-9 h-9 border-2 border-blue-500/50 shadow-lg shadow-blue-500/20">
            <AvatarFallback className="bg-gradient-to-br from-blue-500 to-cyan-500 text-white">
              <SignInButton>
                <UserButton />
              </SignInButton>
            </AvatarFallback>
          </Avatar>
        </div>
      </div>
    </header>
  );
}
