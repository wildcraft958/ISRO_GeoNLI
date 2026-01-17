# SAM3 Agent Frontend

A React + Vite frontend application for visualizing SAM3 segmentation results with masks, bounding boxes, and internal communications.

## Features

- 🖼️ **Image Upload**: Drag-and-drop or click to upload images
- 🎨 **Visualization**: Canvas-based rendering of masks and bounding boxes
- 📊 **Results Display**: View segmentation results, regions, and scores
- 🔍 **Internal Data**: Explore raw SAM3 JSON data and communication logs
- ⚙️ **LLM Configuration**: Configure LLM settings (OpenAI, Anthropic, vLLM, etc.)
- 🚀 **Dual Mode**: Full agent mode (with LLM) or pure SAM3 inference mode

## Setup

### Prerequisites

- Node.js 18+ and npm

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

The app will be available at `http://localhost:5173`

### Build

```bash
npm run build
```

Output will be in the `dist/` directory.

## Deployment to Vercel

### Option 1: Vercel CLI

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Deploy:
```bash
cd frontend
vercel
```

### Option 2: Vercel Dashboard

1. Push your code to GitHub/GitLab/Bitbucket
2. Import the project in Vercel dashboard
3. Vercel will auto-detect Vite and configure build settings
4. The `vercel.json` file is already configured

### Environment Variables (Optional)

You can set these at build time or use runtime configuration:

#### Build-Time Configuration

Set these environment variables before building:

- `VITE_API_ENDPOINT`: Custom API endpoint URL (defaults to Modal endpoint)
- `VITE_INFER_ENDPOINT`: Custom inference endpoint URL

**For local development:**
```bash
# Create .env file in frontend directory
VITE_API_ENDPOINT=https://your-api-endpoint.com
VITE_INFER_ENDPOINT=https://your-infer-endpoint.com
npm run dev
```

**For Vercel deployment:**
Set these in Vercel dashboard under Project Settings > Environment Variables

#### Runtime Configuration

You can override endpoints at runtime without rebuilding:
1. Open the app in your browser
2. Click "Diagnostics" button in the header
3. Use the "Override Endpoints" section to set custom endpoints
4. Endpoints are saved in browser localStorage and persist across sessions

## Usage

1. **Upload Image**: Click or drag an image into the upload area
2. **Enter Prompt**: Type your segmentation prompt (e.g., "segment all objects")
3. **Configure LLM** (if using full agent mode):
   - Base URL: API endpoint (e.g., `https://api.openai.com/v1`)
   - Model: Model name (e.g., `gpt-4o`)
   - API Key: Your API key
4. **Run Segmentation**: Click "Run Segmentation"
5. **View Results**: 
   - See visualization with masks and bboxes
   - Check results panel for summary and regions
   - Explore internal data in communication log

## Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── ImageUpload.tsx
│   │   ├── ImageVisualization.tsx
│   │   ├── ResultsPanel.tsx
│   │   ├── CommunicationLog.tsx
│   │   └── LLMConfigForm.tsx
│   ├── utils/              # Utility functions
│   │   ├── api.ts          # API client
│   │   ├── maskUtils.ts    # RLE decoding
│   │   └── visualization.ts # Canvas drawing
│   ├── App.tsx             # Main app component
│   ├── main.tsx            # Entry point
│   └── index.css           # Global styles
├── package.json
├── vite.config.ts
├── tsconfig.json
├── vercel.json             # Vercel configuration
└── index.html
```

## API Endpoints

The frontend connects to Modal endpoints:

- **Full Agent**: `https://srinjoy59--sam3-agent-sam3-segment.modal.run`
- **Pure SAM3**: `https://srinjoy59--sam3-agent-sam3-infer.modal.run`

These can be overridden with environment variables.

## Technologies

- **React 18**: UI framework
- **TypeScript**: Type safety
- **Vite**: Build tool and dev server
- **Axios**: HTTP client
- **Canvas API**: Image rendering and mask visualization

## Troubleshooting

### Visualizations Not Showing

If visualizations work on one PC but not another, check the following:

1. **Check Browser Console**
   - Open Developer Tools (F12)
   - Look for error messages in the Console tab
   - Check the Network tab for failed API requests

2. **Use Diagnostic Page**
   - Click "Diagnostics" button in the app header
   - Run "Run All Diagnostics" to test:
     - Browser compatibility
     - Canvas 2D context support
     - API endpoint connectivity
     - Network configuration

3. **Common Issues**

   **CORS Errors:**
   - Error message: "CORS Error: Cannot connect to..."
   - **Solution**: The backend server needs to allow cross-origin requests from your frontend domain
   - Check backend CORS configuration to include your frontend URL

   **Network Errors:**
   - Error message: "Network Error: Cannot reach..."
   - **Solutions**:
     - Verify endpoint URLs are correct (check Diagnostic Page)
     - Check if endpoints are accessible from your network
     - Verify firewall/network settings allow connections
     - Try accessing endpoints directly in browser

   **Endpoint Configuration:**
   - Endpoints may be different between environments
   - **Solution**: Use Diagnostic Page to:
     - View current endpoint configuration
     - Override endpoints at runtime (stored in localStorage)
     - Reset to defaults if needed

   **Canvas Not Supported:**
   - Error: "Canvas 2D context is not supported"
   - **Solution**: Update to a modern browser that supports Canvas API

   **Outdated Build:**
   - If endpoints were changed, you may need to rebuild
   - **Solution**: Rebuild the frontend with updated environment variables

4. **Debug Information**
   - All API calls and errors are logged to browser console with `[API]` prefix
   - Visualization rendering is logged with `[Viz]` prefix
   - Check console for detailed error information

5. **Browser Compatibility**
   - Required: Modern browser with Canvas 2D support
   - Tested on: Chrome, Firefox, Safari, Edge (latest versions)
   - Check Diagnostic Page for browser compatibility tests

### API Endpoint Issues

**Different endpoints on different machines:**
- Use the Diagnostic Page to configure endpoints at runtime
- Endpoints are stored in browser localStorage, so each browser/machine has its own configuration
- You can override the default endpoints without rebuilding

**Environment variables not working:**
- Vite environment variables must be prefixed with `VITE_`
- Environment variables are only available at build time
- For runtime configuration, use the Diagnostic Page

### Development vs Production

**Development mode:**
- Endpoints can be configured via `.env` file
- Hot reload available
- More detailed console logging

**Production build:**
- Endpoints configured at build time via environment variables
- Or use runtime configuration via Diagnostic Page
- Optimized and minified code

## License

Same as parent project.

