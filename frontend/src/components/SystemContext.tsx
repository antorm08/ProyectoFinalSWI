import { createContext, useContext, type ReactNode } from "react";
import { api, type Health } from "../api/client";
import { useApiQuery } from "../hooks/useApi";

interface SystemState { data: Health | null; loading: boolean; error: string; retry: () => void }
const SystemContext = createContext<SystemState | null>(null);
const loadHealth = (signal: AbortSignal) => api.health(signal);

export function SystemProvider({ children }: { children: ReactNode }) {
  const state = useApiQuery(loadHealth);
  return <SystemContext.Provider value={state}>{children}</SystemContext.Provider>;
}
export function useSystem() {
  const state = useContext(SystemContext);
  if (!state) throw new Error("SystemProvider no está disponible");
  return state;
}
