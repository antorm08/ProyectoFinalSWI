import { useCallback, useEffect, useRef, useState } from "react";

export function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : "No se pudo completar la solicitud.";
}

export function useApiQuery<T>(load: (signal: AbortSignal) => Promise<T>) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  const retry = useCallback(() => setRevision((v) => v + 1), []);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError(""); setData(null);
    load(controller.signal).then((value) => {
      if (!controller.signal.aborted) setData(value);
    }).catch((err: unknown) => {
      if (!controller.signal.aborted) setError(errorMessage(err));
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false);
    });
    return () => controller.abort();
  }, [load, revision]);
  return { data, error, loading, retry };
}

export function useApiAction<T>() {
  const controller = useRef<AbortController | null>(null);
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const reset = useCallback(() => {
    controller.current?.abort();
    setData(null); setError(""); setLoading(false);
  }, []);
  const run = useCallback(async (execute: (signal: AbortSignal) => Promise<T>) => {
    controller.current?.abort();
    const current = new AbortController();
    controller.current = current;
    setLoading(true); setData(null); setError("");
    try {
      const result = await execute(current.signal);
      if (!current.signal.aborted) setData(result);
    } catch (err: unknown) {
      if (!current.signal.aborted) setError(errorMessage(err));
    } finally {
      if (!current.signal.aborted) setLoading(false);
    }
  }, []);
  useEffect(() => () => controller.current?.abort(), []);
  return { data, error, loading, reset, run };
}
