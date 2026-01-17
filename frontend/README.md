# Frontend — DRISHTI Web Interface

React-based user interface for natural language interaction with satellite imagery, featuring a unified multimodal chat panel for image captioning, visual question answering, and object grounding.

<p align="center">
  <img src="../docs/ui_screenshot.png" alt="DRISHTI UI" width="700"/>
</p>

## Features

| Feature | Description |
|---------|-------------|
| 🖼️ **Image Upload** | Drag-and-drop support for RGB, SAR, IR, and FCC imagery |
| 💬 **Chat Interface** | Conversational multimodal interaction with context memory |
| 📊 **Result Visualization** | Bounding boxes, masks, and annotated responses |
| 🎨 **Dark Theme** | Space-inspired UI with smooth animations |
| 📱 **Responsive** | Works on desktop and tablet |
| 🔄 **Sample Gallery** | Pre-loaded examples for quick exploration |

---

## Capabilities

The frontend interfaces with the DRISHTI backend to provide three core capabilities:

### 1. Image Captioning
> *"Describe this satellite image."*

Generates semantically dense descriptions covering object types, spatial relationships, and scene context.

### 2. Visual Question Answering (VQA)

| Question Type | Example | Backend Routing |
|---------------|---------|-----------------|
| **Semantic** | "What type of buildings are visible?" | VQA Model |
| **Binary** | "Is there a river in the image?" | VQA Model |
| **Numeric** | "How many ships are in the harbor?" | SAM3 + Pyramidal Tiling |

### 3. Visual Grounding
> *"Locate the leftmost storage tank."*

Returns bounding boxes for objects matching natural language queries, handling spatial relationships like "to the left of," "below," "northernmost," etc.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Framework** | React 18 + TypeScript |
| **Build Tool** | Vite |
| **Styling** | TailwindCSS 4 |
| **UI Components** | Radix UI primitives |
| **State Management** | React Context |
| **Routing** | React Router v7 |
| **API Client** | Axios |

---

## Getting Started

```bash
# Install dependencies
npm install

# Start development server
npm run dev    # → http://localhost:3000

# Production build
npm run build
```

---

## Project Structure

```
src/
├── components/
│   ├── ChatInput.tsx       # Message input with image upload
│   ├── ChatMessage.tsx     # Message rendering (text, boxes, masks)
│   ├── Dropzone.tsx        # Image upload zone
│   ├── Sidebar.tsx         # Navigation and chat history
│   ├── BoundingBoxOverlay.tsx  # Grounding result visualization
│   └── ui/                 # Radix-based UI primitives
│       ├── button.tsx
│       ├── card.tsx
│       ├── dialog.tsx
│       └── ...
├── pages/
│   ├── Home.tsx            # Main chat interface
│   ├── LandingPage.tsx     # Landing/onboarding
│   └── ExploreSamplesPage.tsx  # Sample image gallery
├── services/
│   └── chatService.ts      # API client for backend
├── types/
│   └── chat.ts             # TypeScript interfaces
└── lib/
    └── utils.ts            # Utility functions
```

---

## Key Components

### `ChatMessage.tsx`

Renders different response types:
- **Text** — Standard caption or answer
- **Bounding Boxes** — Annotated image with detection overlays
- **Masks** — SAM3 segmentation visualization
- **Numeric** — Count/area results with confidence scores

### `Dropzone.tsx`

Handles multi-modal image uploads:
- Supports JPEG, PNG, TIFF (including 16-bit for SAR)
- Validates file size and format
- Generates base64 for API transmission

### `ExploreSamplesPage.tsx`

Pre-loaded sample images for demonstration:
- Valley terrain (grounding queries)
- Road infrastructure (captioning)
- Urban areas (counting/VQA)
- Harbor scenes (multi-object detection)

---

## Environment Variables

Create a `.env.local` file:

```bash
# API Configuration
VITE_API_URL=http://localhost:8000

# Optional: Authentication
VITE_CLERK_PUBLISHABLE_KEY=pk_...
```

---

## API Integration

The frontend communicates with the backend via `chatService.ts`:

```typescript
// Send multimodal chat request
const response = await chatService.sendMessage({
  sessionId: "abc123",
  imageB64: base64Image,
  query: "How many cars are parked?",
  mode: "auto"  // auto | captioning | vqa | grounding
});

// Response structure
interface ChatResponse {
  response: string;
  taskType: "captioning" | "vqa_semantic" | "vqa_binary" | "vqa_numeric" | "grounding";
  confidence: number;
  boundingBoxes?: BBox[];
  masks?: string[];  // Base64 mask images
  metadata?: {
    detectedModality: string;
    sam3Count?: number;
    sam3Area?: number;
  };
}
```

---

## Design System

The UI follows a space-inspired dark theme:

| Element | Value |
|---------|-------|
| **Background** | `#0a0a0f` (near-black) |
| **Surface** | `#1a1a2e` (dark slate) |
| **Primary** | `#00d9ff` (cyan accent) |
| **Text** | `#e0e0e0` (light gray) |
| **Border** | `#2a2a4a` (subtle purple) |

Animations use CSS transitions for smooth interactions and loading states.
