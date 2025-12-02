import { projectId, publicAnonKey } from "./supabase/info";

const API_BASE = `https://${projectId}.supabase.co/functions/v1/make-server-1ac2e09a`;

export interface ChatSession {
  id: string;
  userId: string;
  messages: any[];
  mode: string;
  title: string;
  createdAt: string;
  updatedAt: string;
}

export interface Analysis {
  id: string;
  userId: string;
  type: string;
  result: string;
  imageUrl?: string;
  title: string;
  createdAt: string;
}

export interface Sample {
  id: string;
  title: string;
  type: string;
  description: string;
  imageUrl: string;
  result: string;
}

// Helper function to make API calls
async function apiCall<T>(
  endpoint: string,
  options: RequestInit = {},
  accessToken?: string
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };

  // Use access token if provided, otherwise use public anon key
  headers["Authorization"] = `Bearer ${accessToken || publicAnonKey}`;

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: "Unknown error" }));
    throw new Error(error.error || `API call failed: ${response.statusText}`);
  }

  return response.json();
}

// ============================================
// AUTH API
// ============================================

export async function signUp(email: string, password: string, name: string) {
  return apiCall<{ user: any }>("/signup", {
    method: "POST",
    body: JSON.stringify({ email, password, name }),
  });
}

// ============================================
// CHAT HISTORY API
// ============================================

export async function fetchChats(accessToken: string) {
  return apiCall<{ chats: ChatSession[] }>("/chats", {}, accessToken);
}

export async function saveChat(
  accessToken: string,
  sessionId: string,
  messages: any[],
  mode: string,
  title: string
) {
  return apiCall<{ success: boolean; session: ChatSession }>(
    "/chats",
    {
      method: "POST",
      body: JSON.stringify({ sessionId, messages, mode, title }),
    },
    accessToken
  );
}

export async function deleteChat(accessToken: string, sessionId: string) {
  return apiCall<{ success: boolean }>(`/chats/${sessionId}`, { method: "DELETE" }, accessToken);
}

// ============================================
// ANALYSES API
// ============================================

export async function fetchAnalyses(accessToken: string) {
  return apiCall<{ analyses: Analysis[] }>("/analyses", {}, accessToken);
}

export async function saveAnalysis(
  accessToken: string,
  analysisId: string,
  type: string,
  result: string,
  title: string,
  imageUrl?: string
) {
  return apiCall<{ success: boolean; analysis: Analysis }>(
    "/analyses",
    {
      method: "POST",
      body: JSON.stringify({ analysisId, type, result, title, imageUrl }),
    },
    accessToken
  );
}

// ============================================
// SAMPLES API
// ============================================

export async function fetchSamples() {
  return apiCall<{ samples: Sample[] }>("/samples");
}
