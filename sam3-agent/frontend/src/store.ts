import { create } from 'zustand';
import { LLMConfig, SegmentResponse, SAM3Config } from './utils/api';

interface AppState {
  // Input State
  imageBase64: string | null;
  imageFile: File | null;
  imageUrl: string | null;
  prompt: string;

  // Config State
  llmConfig: LLMConfig;
  sam3Config: SAM3Config;
  useInfer: boolean; // Toggle between full agent and fast counting

  // Result State
  response: SegmentResponse | null;
  loading: boolean;
  loadingStage: 'starting' | 'encoding_text' | 'encoding_images' | 'processing_tiles' | 'verification' | 'finalizing' | null;
  error: string | null;

  // Actions
  setImage: (base64: string | null, file: File | null, url: string | null) => void;
  setPrompt: (prompt: string) => void;
  setLlmConfig: (config: LLMConfig) => void;
  setSam3Config: (config: SAM3Config) => void;
  setUseInfer: (useInfer: boolean) => void;
  setResponse: (response: SegmentResponse | null) => void;
  setLoading: (loading: boolean) => void;
  setLoadingStage: (stage: 'starting' | 'encoding_text' | 'encoding_images' | 'processing_tiles' | 'verification' | 'finalizing' | null) => void;
  setError: (error: string | null) => void;
  resetState: () => void;
}

export const useStore = create<AppState>((set) => ({
  // Initial State
  imageBase64: null,
  imageFile: null,
  imageUrl: null,
  prompt: 'segment all objects',

  llmConfig: {
    base_url: import.meta.env.VITE_LLM_BASE_URL || 'https://api.openai.com/v1',
    model: 'gpt-4o',
    api_key: '',
    name: 'default',
    max_tokens: 2048,
  },

  sam3Config: {
    confidence_threshold: 0.4,
    max_retries: 2,
    include_obb: false,
    obb_as_polygon: false,
    pyramidal_config: {
      tile_size: 512,
      overlap_ratio: 0.15,
      scales: [1.0],
      batch_size: 16,
      iou_threshold: 0.5
    }
  },

  useInfer: false,

  response: null,
  loading: false,
  loadingStage: null,
  error: null,

  // Actions
  setImage: (base64, file, url) => set({ imageBase64: base64, imageFile: file, imageUrl: url, response: null, error: null }),
  setPrompt: (prompt) => set({ prompt }),
  setLlmConfig: (config) => set({ llmConfig: config }),
  setSam3Config: (config) => set({ sam3Config: config }),
  setUseInfer: (useInfer) => set({ useInfer }),
  setResponse: (response) => set({ response }),
  setLoading: (loading) => set({ loading }),
  setLoadingStage: (stage) => set({ loadingStage: stage }),
  setError: (error) => set({ error }),

  resetState: () => set({
    imageBase64: null,
    imageFile: null,
    imageUrl: null,
    response: null,
    error: null,
    loading: false
  })
}));
