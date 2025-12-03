import { apiClient, routes } from "@/lib/api";

export interface ChatResponse {
  session_id: string;
  response_type: "caption" | "answer" | "boxes";
  content: string | { boxes: any[]; image_width: number; image_height: number };
  execution_log: string[];
  message_id: string;
  detected_modality: string;
  modality_confidence: number;
  resnet_classification_used: boolean;
  vqa_type: string | null;
  vqa_type_confidence: number | null;
  converted_image_url: string | null;
  original_image_url: string | null;
  buffer_token_count: number;
  buffer_summarized: boolean;
}

export interface ImageUploadResponse {
  chat_id: string;
  image_url: string;
  filename: string;
}

export interface CreateChatResponse {
  status: string;
  chat_id: string;
  message: string;
}

export interface CreateQueryResponse {
  status: string;
  query_id: number;
  chat_id: string;
  message: string;
}

export interface QueryData {
  id: number;
  chat_id: string;
  question_text: string;
  response_text: string | null;
  query_type: string;
}

export const chatService = {
  // ===== Chat Operations =====
  async createChat(userId: string, imageUrl?: string): Promise<CreateChatResponse> {
    const response = await apiClient.post(routes.CHAT_CREATE, {
      user_id: userId,
      image_url: imageUrl || null,
    });
    return response.data;
  },

  async sendMessage(
    sessionId: string,
    userId: string,
    query: string,
    imageUrl: string = "",
    mode: "auto" | "grounding" | "vqa" | "captioning" = "auto",
    modalityDetectionEnabled: boolean = true,
    needsIr2rgb: boolean = false,
    ir2rgbChannels: string[] = [],
    ir2rgbSynthesize: "B" | "G" | "R" | null = "B"
  ): Promise<ChatResponse> {
    const payload = {
      session_id: sessionId,
      user_id: userId,
      image_url: imageUrl,
      query: query,
      mode: mode,
      modality_detection_enabled: modalityDetectionEnabled,
      needs_ir2rgb: needsIr2rgb,
      ir2rgb_channels: ir2rgbChannels,
      ir2rgb_synthesize: ir2rgbSynthesize,
    };

    const response = await apiClient.post(routes.ORCHESTRATOR_CHAT, payload);
    return response.data;
  },

  async uploadImage(file: File, userId: string): Promise<ImageUploadResponse> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await apiClient.post(routes.IMAGE_UPLOAD, formData, {
      params: { user_id: userId },
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    console.log(response.data);
    return response.data;
  },

  async getChats(userId: string): Promise<any[]> {
    const response = await apiClient.get(routes.GET_CHATS, {
      params: { user_id: userId },
    });
    return response.data;
  },

  async deleteChat(chatId: string): Promise<any> {
    const response = await apiClient.delete(`/chat/${chatId}`);
    return response.data;
  },

  // ===== Query Operations =====
  async createQuery(
    chatId: string,
    questionText: string,
    queryType?: string
  ): Promise<CreateQueryResponse> {
    const response = await apiClient.post(routes.QUERY_CREATE, {
      chat_id: chatId,
      question_text: questionText,
      query_type: queryType || "general",
    });
    return response.data;
  },

  async getQuery(queryId: number): Promise<QueryData> {
    const response = await apiClient.get(`${routes.QUERY_GET}/${queryId}`);
    return response.data;
  },

  async getChatQueries(chatId: string): Promise<QueryData[]> {
    const response = await apiClient.get(`${routes.QUERY_LIST}/${chatId}`);
    return response.data;
  },

  async updateQueryResponse(queryId: number, responseText: string): Promise<any> {
    const response = await apiClient.put(`${routes.QUERY_GET}/${queryId}`, {
      response_text: responseText,
    });
    return response.data;
  },
};
