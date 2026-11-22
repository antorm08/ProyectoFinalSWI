import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

// Integración con la API local real: no se sustituyen resultados por métricas ficticias.
const routes = [
  ["/", "Dashboard ejecutivo"],
  ["/retrasos", "Riesgo de entregas"],
  ["/demanda", "Pronóstico de demanda"],
  ["/productos", "Clasificación visual"],
  ["/vendedores", "Segmentación de vendedores"],
  ["/resenas", "Resumen de reseñas"],
] as const;

for (const [path, title] of routes) {
  test(`${title}: ruta, responsive y accesibilidad`, async ({ page }, info) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(path);
    await expect(page.getByRole("heading", { name: title, exact: true, level: 1 })).toBeVisible();
    await expect(page.getByText("Entorno académico", { exact: true })).toHaveCount(0);
    await page.waitForLoadState("networkidle");
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
    expect(errors).toEqual([]);
    const audit = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
    expect(audit.violations.map((v) => ({ id: v.id, nodes: v.nodes.map((node) => node.target) }))).toEqual([]);
    await page.screenshot({ path: info.outputPath(`${title.replaceAll(" ", "-")}.png`), fullPage: true });
  });
}

test("navegación por teclado y cierre del menú móvil", async ({ page }) => {
  await page.goto("/productos");
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Saltar al contenido" })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("#contenido")).toBeFocused();
  const menu = page.getByRole("button", { name: "Abrir navegación" });
  if (await menu.isVisible()) {
    await menu.click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await expect(page.getByRole("button", { name: "Cerrar", exact: true })).toBeFocused();
    const audit = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
    expect(audit.violations).toEqual([]);
    await page.keyboard.press("Escape");
    await expect(page.getByRole("dialog")).toHaveCount(0);
    await expect(menu).toBeFocused();
    await menu.click();
    await page.getByRole("dialog").getByRole("link", { name: "Resumen de reseñas" }).click();
    await expect(page).toHaveURL("/resenas");
    await expect(page.getByRole("dialog")).toHaveCount(0);
  }
});

test("retrasos conserva el contrato y retira resultados al editar", async ({ page }) => {
  await page.goto("/retrasos");
  await page.getByLabel("Precio total (BRL)").fill("175");
  const response = page.waitForResponse((res) => res.url().endsWith("/api/delay/predict") && res.request().method() === "POST");
  await page.getByRole("button", { name: "Evaluar riesgo de entrega" }).click();
  const res = await response;
  expect(res.status()).toBe(200);
  expect(res.request().postDataJSON()).toEqual({ precio: 175, flete: 20, peso_g: 1500, n_items: 1, dias_estimados: 20, estado_cliente: "RJ", estado_vendedor: "SP", mes: 11, dia_semana: 0 });
  if (!(await res.json()).modelo_entrenado) {
    await expect(page.getByText("Estimación orientativa", { exact: true })).toHaveCount(1);
    await expect(page.getByText(/no una predicción validada/)).toBeVisible();
  }
  await expect(page.getByText("Probabilidad de retraso", { exact: true })).toBeVisible();
  await page.getByLabel("Precio total (BRL)").fill("200");
  await expect(page.getByText("Listo para evaluar", { exact: true })).toBeVisible();
  await expect(page.getByText("Probabilidad de retraso", { exact: true })).toHaveCount(0);
});

test("dashboard mantiene los indicadores técnicos en una sección secundaria", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Pedidos previstos", { exact: true })).toBeVisible();
  const health = await (await page.request.get("http://localhost:8000/api/health")).json();
  if (!health.modelos.demanda && !health.datos_olist.orders) {
    await expect(page.getByText("Datos sintéticos · solo demostración", { exact: true })).toBeVisible();
    await expect(page.getByText("Datos sintéticos · solo demostración", { exact: true })).toHaveCount(1);
  }
  await expect(page.getByText("Modelos detectados", { exact: true })).toBeHidden();
  await page.getByText("Estado del sistema", { exact: true }).click();
  await expect(page.getByText("Modelos detectados", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Métodos y modelos" })).toBeVisible();
  const audit = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
  expect(audit.violations.map((v) => v.id)).toEqual([]);
});

test("copiar resumen conserva el texto y comunica errores de portapapeles", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"], { origin: "http://localhost:5173" });
  await page.goto("/resenas");
  const original = "El producto tiene buena calidad. La entrega fue puntual y segura. El embalaje protege bien el producto. Recomiendo comprar este producto.";
  await page.getByLabel("Reseñas", { exact: true }).fill(original);
  await page.getByRole("button", { name: "Generar resumen" }).click();
  await expect(page.locator("#summary-text")).toBeVisible();
  await expect(page.getByLabel("Reseñas", { exact: true })).toHaveValue(original);
  const summary = await page.locator("#summary-text").innerText();
  await page.getByRole("button", { name: "Copiar resumen" }).click();
  await expect(page.getByText("Resumen copiado.", { exact: true })).toBeVisible();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(summary);
  await page.evaluate(() => {
    Object.defineProperty(navigator.clipboard, "writeText", {
      configurable: true,
      value: () => Promise.reject(new Error("Permiso denegado en prueba")),
    });
  });
  await page.getByRole("button", { name: "Copiar resumen" }).click();
  await expect(page.getByRole("alert")).toContainText("No se pudo copiar");
  await expect(page.locator("#summary-text")).toHaveText(summary);
  await page.getByLabel("Reseñas", { exact: true }).fill(`${original} Otra opinión completa.`);
  await expect(page.getByRole("button", { name: "Copiar resumen" })).toHaveCount(0);
  await expect(page.getByText("Resumen copiado.", { exact: true })).toHaveCount(0);
});

test("demanda aplica horizonte sin solicitudes por cada movimiento", async ({ page }) => {
  await page.goto("/demanda");
  await expect(page.getByText("Pedidos previstos", { exact: true })).toBeVisible();
  const requests: string[] = [];
  page.on("request", (req) => { if (req.url().includes("/api/demand/forecast")) requests.push(req.url()); });
  await page.getByLabel(/Horizonte de pronóstico/).fill("21");
  await expect(page.getByText(/Cambio pendiente/)).toBeVisible();
  expect(requests).toHaveLength(0);
  const response = page.waitForResponse((res) => res.url().includes("/api/demand/forecast?horizonte=21&historico=90"));
  await page.getByRole("button", { name: "Aplicar horizonte" }).click();
  expect((await response).status()).toBe(200);
  await expect(page.getByText(/Cambio pendiente/)).toHaveCount(0);
  await expect(page.getByText("Pedidos previstos", { exact: true })).toBeVisible();
  await page.getByText("Ver datos del gráfico en tabla", { exact: true }).click();
  await expect(page.getByRole("table")).toBeVisible();
});

test("imágenes conserva multipart imagen y ejecuta inferencia", async ({ page }) => {
  await page.goto("/productos");
  const response = page.waitForResponse((res) => res.url().endsWith("/api/products/classify") && res.request().method() === "POST");
  await page.getByLabel("Seleccionar imagen", { exact: true }).setInputFiles({
    name: "imagen-diagnostico.png", mimeType: "image/png",
    buffer: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a53cAAAAASUVORK5CYII=", "base64"),
  });
  const res = await response;
  expect(res.status()).toBe(200);
  expect(res.request().headers()["content-type"]).toContain("multipart/form-data");
  // Chromium no siempre expone el cuerpo multipart. Un 200 de este endpoint
  // confirma que FastAPI recibió el campo obligatorio `imagen` (otro nombre da 422).
  expect(await res.json()).toMatchObject({ modelo_entrenado: true });
  await expect(page.getByText("Categoría propuesta", { exact: true })).toBeVisible();
  const audit = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
  expect(audit.violations.map((v) => ({ id: v.id, nodes: v.nodes.map((node) => node.target) }))).toEqual([]);
});

test("reseñas conserva texto, extensión y cambio de modalidad", async ({ page }) => {
  await page.goto("/resenas");
  const text = "El producto tiene buena calidad. La entrega del producto fue puntual. El embalaje protege bien el producto. Recomiendo comprar este producto.";
  await page.getByLabel("Reseñas", { exact: true }).fill(text);
  await page.getByLabel("Oraciones del resumen", { exact: true }).fill("2");
  const response = page.waitForResponse((res) => res.url().endsWith("/api/reviews/summarize") && res.request().method() === "POST");
  await page.getByRole("button", { name: "Generar resumen" }).click();
  const res = await response;
  expect(res.status()).toBe(200);
  expect(res.request().postDataJSON()).toEqual({ texto: text, n_oraciones: 2 });
  await expect(page.getByText("Oraciones originales", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Por vendedor", exact: true }).click();
  await expect(page.getByText("Oraciones originales", { exact: true })).toHaveCount(0);
  await page.getByLabel("Identificador del vendedor", { exact: true }).fill("diagnostico-sin-datos");
  const missing = page.waitForResponse((result) => result.url().includes("/api/reviews/seller/diagnostico-sin-datos?n_oraciones=2"));
  await page.getByRole("button", { name: "Generar resumen" }).click();
  expect((await missing).status()).toBe(404);
  await expect(page.getByRole("alert")).toContainText(/Sin reseñas/);
});

test("errores de conexión son visibles y se puede reintentar", async ({ page }) => {
  await page.route("**/api/health", (route) => route.abort("failed"));
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("No se pudo conectar");
  await page.getByText("Estado del sistema", { exact: true }).click();
  await expect(page.getByText("Estado no confirmado").first()).toBeVisible();
  await page.unroute("**/api/health");
  await page.getByRole("button", { name: "Reintentar", exact: true }).click();
  await expect(page.getByText("Sistema conectado").first()).toBeAttached();
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("una respuesta tardía no restaura un resultado de un escenario editado", async ({ page }) => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  let reached!: () => void;
  const fetched = new Promise<void>((resolve) => { reached = resolve; });
  await page.route("**/api/delay/predict", async (route) => {
    const response = await route.fetch();
    reached();
    await gate;
    try { await route.fulfill({ response }); } catch { /* La edición cancela la solicitud. */ }
  });
  await page.goto("/retrasos");
  await page.getByRole("button", { name: "Evaluar riesgo de entrega" }).click();
  await fetched;
  await page.getByLabel("Precio total (BRL)").fill("300");
  release();
  await expect(page.getByText("Listo para evaluar", { exact: true })).toBeVisible();
  await expect(page.getByText("Probabilidad de retraso", { exact: true })).toHaveCount(0);
});
