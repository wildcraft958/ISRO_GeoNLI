import axios from "axios";
export const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

const apiClient = axios.create({
  baseURL: BACKEND_URL,
  headers: {
    "Content-Type": "application/json",
  },
});
apiClient.defaults.withCredentials = true;

export * as routes from "./routes";
export { apiClient };
