import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./index.css";
import { SystemProvider } from "./components/SystemContext";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <SystemProvider><App /></SystemProvider>
    </BrowserRouter>
  </StrictMode>,
);
