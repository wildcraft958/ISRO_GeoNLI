import { apiClient, routes } from "@/lib/api";

export interface ChatResponse {
  status: string;
  intent: string;
  response: {
    content: string;
    suggestions?: string[];
    boxes?: any[];
  };
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

  async sendMessage(sessionId: string, query: string, imageUrl?: string, mode: string = "auto"): Promise<ChatResponse> {
    const response = await apiClient.post(routes.ORCHESTRATOR_CHAT, {
      image_url: imageUrl,
      query: query,
      mode: mode,
      session_id: sessionId,
    });
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
