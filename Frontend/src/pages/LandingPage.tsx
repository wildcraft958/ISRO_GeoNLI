import {
  ArrowRight,
  Satellite,
  Brain,
  Zap,
  Shield,
  Globe2,
  Sparkles,
  Target,
  MessageSquare,
} from "lucide-react";
import { OrbitLogo } from "../components/OrbitLogo";
import { SignedIn, SignedOut, SignInButton, UserButton } from "@clerk/clerk-react";
import { useNavigate } from "react-router-dom";
import satImage from "../assets/sat_demo.jpg";
  import Lottie from "lottie-react";
import { SyncUserToBackend } from "@/utils/SyncToBackend";
import { SpaceBackground } from "../components/SpaceBackground";
import satellite from "@/assets/satellite.json"
export function LandingPage() {
  const navigate = useNavigate();
  return (
    <div className="relative min-h-screen w-full overflow-x-hidden bg-[#000814]">
      <SpaceBackground/>
      <div className="relative z-10">
        {/* Navigation */}
        <nav className="fixed top-0 w-full z-50 h-16 bg-red-900/80 backdrop-blur-md border-b border-cyan-500/10 flex justify-center overflow-hidden">
          <div className="w-full h-full flex items-center justify-between">
            <div className="flex items-center gap-3 h-full">
              <div className="relative h-full flex justify-start items-center ml-9">
                <div className="scale-[0.4] origin-left pl-12">
                  <OrbitLogo />
                </div>
              </div>
            </div>

            <SignedOut>
              <div className="px-5 py-1 text-cyan-400 hover:text-cyan-300 transition-colors cursor-pointer">
                <SignInButton />
              </div>
            </SignedOut>
            <SignedIn>
              <SyncUserToBackend /> {/* Sync user data to mongodb backend */}
              <div className="px-5 py-1 text-cyan-400 hover:text-cyan-300 transition-colors cursor-pointer">
                <UserButton />
              </div>
            </SignedIn>
          </div>
        </nav>

        {/* Hero Section */}
        <section className="pt-32 pb-20 px-6">
          <div className="max-w-7xl mx-auto">
            <div className="grid lg:grid-cols-2 gap-12 items-center">
              {/* Left Column */}
              <div className="space-y-8">
                <h1 className="text-5xl md:text-6xl lg:text-7xl text-white leading-tight">
                  Unlock the Language of{" "}
                  <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                    Satellite Imagery
                  </span>
                </h1>

                <p className="text-xl text-gray-400 leading-relaxed">
                  See, ask, and know your world in seconds with Aksha Drishti's advanced AI vision
                  platform. Transform satellite data into actionable insights instantly.
                </p>

                <div className="flex flex-wrap gap-4">
                  <button
                    className="group cursor-pointer px-8 py-4 bg-cyan-600 text-white rounded-lg hover:shadow-2xl hover:shadow-cyan-500/50 transition-all duration-300 hover:scale-105 flex items-center gap-2"
                    onClick={() => {
                      navigate("/geo_nli");
                    }}
                  >
                    Start Analyzing
                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                  </button>
                </div>
              </div>

              {/* Right Column - Visual */}
              <div className="">
                <div className="max-md:hidden">
                            <Lottie animationData={satellite} loop={true} className="w-100 pointer-events-none" />
                          </div>
              </div>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-20 px-6">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-16">
              <h2 className="text-4xl md:text-5xl text-white mb-4">Three Powerful Modes</h2>
              <p className="text-xl text-gray-400">
                Comprehensive satellite imagery analysis at your fingertips
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              {/* Caption Mode */}
              <div className="group relative bg-white/5 border border-cyan-500/20 rounded-3xl p-8 hover:bg-white/10 hover:border-cyan-500/50 transition-all duration-300 hover:scale-105">
                <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-transparent rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <div className="relative z-10">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                    <Sparkles className="w-8 h-8 text-blue-400" />
                  </div>
                  <h3 className="text-2xl text-white mb-4">Generate Caption</h3>
                  <p className="text-gray-400 leading-relaxed">
                    Upload satellite imagery and receive AI-generated descriptive captions
                    instantly. Perfect for quick insights and documentation.
                  </p>
                </div>
              </div>

              {/* Question Mode */}
              <div className="group relative bg-white/5 border border-cyan-500/20 rounded-3xl p-8 hover:bg-white/10 hover:border-cyan-500/50 transition-all duration-300 hover:scale-105">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-transparent rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <div className="relative z-10">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                    <MessageSquare className="w-8 h-8 text-blue-400" />
                  </div>
                  <h3 className="text-2xl text-white mb-4">Visual Question Answering</h3>
                  <p className="text-gray-400 leading-relaxed">
                    Upload imagery and ask detailed questions. Get comprehensive answers about
                    features, changes, and patterns in your data.
                  </p>
                </div>
              </div>

              {/* Grounding Mode */}
              <div className="group relative bg-white/5 border border-cyan-500/20 rounded-3xl p-8 hover:bg-white/10 hover:border-cyan-500/50 transition-all duration-300 hover:scale-105">
                <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/10 to-transparent rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <div className="relative z-10">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                    <Target className="w-8 h-8 text-blue-400" />
                  </div>
                  <h3 className="text-2xl text-white mb-4">Perform Grounding</h3>
                  <p className="text-gray-400 leading-relaxed">
                    Detect and locate specific objects in satellite imagery with bounding boxes.
                    Precise spatial analysis powered by AI.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Benefits Section */}
        <section className="py-20 px-6 bg-gradient-to-b from-transparent via-cyan-500/5 to-transparent">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-16">
              <h2 className="text-4xl md:text-5xl text-white mb-4">Why Choose Aksha Drishti?</h2>
              <p className="text-xl text-gray-400">
                Built with cutting-edge technology and ISRO-inspired precision
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-white/5 border border-white/10 rounded-2xl p-6 backdrop-blur-sm">
                <Brain className="w-10 h-10 text-cyan-400 mb-4" />
                <h4 className="text-lg text-white mb-2">AI-Powered</h4>
                <p className="text-sm text-gray-400">
                  Advanced neural networks trained on millions of satellite images
                </p>
              </div>

              <div className="bg-white/5 border border-white/10 rounded-2xl p-6 backdrop-blur-sm">
                <Zap className="w-10 h-10 text-cyan-400 mb-4" />
                <h4 className="text-lg text-white mb-2">Lightning Fast</h4>
                <p className="text-sm text-gray-400">
                  Get results in seconds, not hours. Real-time analysis at scale
                </p>
              </div>

              <div className="bg-white/5 border border-white/10 rounded-2xl p-6 backdrop-blur-sm">
                <Shield className="w-10 h-10 text-cyan-400 mb-4" />
                <h4 className="text-lg text-white mb-2">Secure & Private</h4>
                <p className="text-sm text-gray-400">
                  Enterprise-grade security with end-to-end encryption
                </p>
              </div>

              <div className="bg-white/5 border border-white/10 rounded-2xl p-6 backdrop-blur-sm">
                <Globe2 className="w-10 h-10 text-cyan-400 mb-4" />
                <h4 className="text-lg text-white mb-2">Global Coverage</h4>
                <p className="text-sm text-gray-400">Analyze imagery from any location on Earth</p>
              </div>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="border-t border-white/10 py-4 px-6">
          <div className="max-w-7xl mx-auto">
            <div className="grid md:grid-cols-3 gap-8 mb-8">
              <div className="col-span-2">
                <div className="scale-[0.4] origin-top-left mb-4">
                  <OrbitLogo />
                </div>
                <p className="text-gray-400 text-sm max-w-md">
                  Advanced AI-powered satellite imagery analysis platform. See, ask, and know your
                  world in seconds.
                </p>
              </div>

              <div>
                <h5 className="text-white mb-4">Product</h5>
                <ul className="space-y-2 text-sm text-gray-400">
                  <li>
                    <a href="#" className="hover:text-cyan-400 transition-colors">
                      Documentation
                    </a>
                  </li>
                </ul>
              </div>
            </div>

            <div className="border-t border-white/10 pt-4 text-center text-sm text-gray-500">
              <p>
                &copy; 2025 Aksha Drishti. All rights reserved. Built with precision and inspired by
                ISRO.
              </p>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
