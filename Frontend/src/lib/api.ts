import axios from "axios";
export const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;

const apiClient = axios.create({
  baseURL: BACKEND_URL,
});
apiClient.defaults.withCredentials = true;

export * as routes from "./routes";
export { apiClient };