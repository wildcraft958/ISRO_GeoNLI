import React, { useEffect, useRef } from "react";

export function SpaceBackground() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleMouseMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Update CSS variables for high performance
      container.style.setProperty("--mouse-x", `${x}px`);
      container.style.setProperty("--mouse-y", `${y}px`);
    };

    window.addEventListener("mousemove", handleMouseMove);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
    };
  }, []);

  return (
    <div ref={containerRef} className=" fixed inset-0 overflow-hidden font-vendsans bg-[#000510]">
      {/* 1. Base Gradient (Deep Space) */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#0a0a1f] via-[#000814] to-[#020210] z-0" />

      {/* 2. Static Stars (Subtle background texture) */}
      <div
        className="absolute inset-0 opacity-20 z-0"
        style={{
          backgroundImage: `radial-gradient(1.5px 1.5px at 20px 30px, white, transparent),
                           radial-gradient(1.5px 1.5px at 150px 80px, white, transparent)`,
          backgroundSize: "300px 300px",
        }}
      />

      {/* 3. The Interactive Grid ("The Cool Tiles") */}
      <div
        className="absolute inset-0 z-10 opacity-40 pointer-events-none"
        style={{
          // Create the grid lines using CSS gradients
          backgroundImage: `
            linear-gradient(to right, rgba(255, 255, 255, 0.13) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(255, 255, 255, 0.13) 1px, transparent 1px)
          `,
          backgroundSize: "40px 40px", // Size of the tiles

          // The Magic: A mask that follows the mouse to reveal the grid
          maskImage: `radial-gradient(
            300px circle at var(--mouse-x, 0px) var(--mouse-y, 0px), 
            black 0%, 
            transparent 100%
          )`,
          WebkitMaskImage: `radial-gradient(
            300px circle at var(--mouse-x, 0px) var(--mouse-y, 0px), 
            black 0%, 
            transparent 100%
          )`,
        }}
      />

      {/* 4. Mouse Glow (The soft blue ambient light following cursor) */}
      <div
        className="absolute inset-0 z-0 pointer-events-none transition-opacity duration-300"
        style={{
          background: `radial-gradient(
            600px circle at var(--mouse-x, 0px) var(--mouse-y, 0px), 
            rgba(29, 78, 216, 0.195), 
            transparent 60%
          )`,
        }}
      />
    </div>
  );
}
