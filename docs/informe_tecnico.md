# Informe técnico – CommerceAI

> Plantilla con los apartados exigidos por el enunciado. Completar en cada sección.

## 1. Descripción de funcionalidades de la aplicación
- Panel general
- Predicción del riesgo de retraso
- Pronóstico de demanda
- Clasificación de productos por imagen
- Segmentación de vendedores
- Resumen de reseñas

## 2. Arquitectura del software inteligente
### 2.1 Visión general (diagrama de capas)
### 2.2 Frontend (React + TypeScript)
### 2.3 Backend / API (FastAPI)
### 2.4 Componentes de IA
#### 2.4.1 Predicción de retraso – datos, variables, modelo, métricas (ROC-AUC, F1)
#### 2.4.2 Series temporales – serie diaria, SARIMA, validación (MAE, MAPE)
#### 2.4.3 Clasificación de imágenes – CNN, Fashion-MNIST, accuracy, matriz de confusión

**Problema.** En un marketplace, cada vendedor publica productos con una categoría que a menudo falta o es incorrecta.
El componente propone automáticamente la categoría de un producto a partir de su foto, para agilizar el alta en el
catálogo y detectar productos mal categorizados.

**Datos.** Fashion-MNIST (ver §2.5): 70 000 imágenes en gris de 28 × 28 px de productos de Zalando, en 10 categorías
equilibradas: camiseta/top, pantalón, jersey, vestido, abrigo, sandalia, camisa, zapatilla, bolso y botín.

**Modelo.** Red neuronal convolucional (CNN) implementada en PyTorch:

| Capa | Salida | Función |
|---|---|---|
| Entrada | 1 × 28 × 28 | Imagen en gris normalizada (media 0,2860, desviación 0,3530) |
| Conv 3×3 (32) + BatchNorm + ReLU + MaxPool | 32 × 14 × 14 | Bordes y texturas |
| Conv 3×3 (64) + BatchNorm + ReLU + MaxPool | 64 × 7 × 7 | Formas: mangas, cuellos, suelas, asas |
| Flatten + Dropout (0,3) + Dense (128) + ReLU | 128 | Combinación de rasgos; el dropout reduce el sobreajuste |
| Dense (10) + softmax | 10 | Probabilidad de cada categoría |

En total, 421 834 parámetros entrenables. La arquitectura es deliberadamente compacta: entrena en CPU en ~15 min y clasifica
una imagen en ~2 ms (preprocesado incluido, CPU Ryzen 5 3600), suficiente para la resolución de 28 × 28 px.

**Entrenamiento y validación** (`ml/train_products.py`):

- Del conjunto oficial de entrenamiento (60 000) se separa una **validación estratificada de 6 000** imágenes (semilla 42).
  El **test oficial (10 000)** solo se usa una vez, al final, para que la métrica publicada no esté sesgada.
- Optimizador Adam (tasa 0,001, lote 128), con reducción de la tasa a la mitad cuando la validación se estanca.
  **Early stopping**: se conserva la época con mejor accuracy de validación (máximo 20; paciencia 4).
- Aumento de datos: volteo horizontal y pequeñas traslaciones y cambios de escala, para tolerar prendas no centradas.
- El modelo se exporta como TorchScript (`backend/app/models/product_cnn.pt`), con verificación de que el archivo recargado
  reproduce las predicciones. Protocolo, historial y métricas: `docs/clasificacion_productos_metricas.json`.

**Resultados en test** (10 000 imágenes no vistas):

| Métrica | Valor |
|---|---:|
| **Accuracy** | **92,46 %** (objetivo ≥ 91 %) |
| F1 macro | 0,924 |
| Accuracy en validación (época 20 de 20) | 92,77 % |

Las mejores categorías son pantalón (F1 0,994), bolso (0,989) y sandalia (0,986). La más difícil es **camisa** (F1 0,773):
la matriz de confusión muestra que los errores se concentran entre prendas de torso, cuya silueta a 28 × 28 px apenas
difiere (camiseta → camisa: 100 casos; camisa → camiseta: 87; camisa → abrigo: 72). Entre los errores con más confianza
aparecen además etiquetas dudosas del propio dataset.

![Matriz de confusión en test](resultados_clasificacion_productos/matriz_confusion.png)

**Confianza y aviso de revisión.** La probabilidad máxima no está calibrada, pero ordena bien los casos dudosos (confianza
media 0,949 en aciertos frente a 0,688 en errores). Con un umbral de **0,7, elegido en validación**, el 90 % de las
predicciones se muestran sin aviso con un 97 % de acierto, y el resto se marca para revisión manual; así se señalan casi dos
tercios de los errores.

**Integración en la aplicación.** El endpoint `POST /api/products/classify` recibe la imagen (multipart, campo `imagen`,
hasta 5 MB) y devuelve la categoría, la confianza, las tres categorías más probables y el indicador `baja_confianza`.
Antes de clasificar, `product_service.preprocess` adapta cualquier foto al formato de Fashion-MNIST:

1. Corrige la orientación EXIF, compone la transparencia sobre blanco y convierte a escala de grises.
2. Estima el fondo con los bordes de la imagen; si es claro la invierte (Fashion-MNIST usa prenda clara sobre negro).
3. **Recorta la imagen a la prenda**, la rellena a cuadrado y la escala a 28 × 28 px, igual que se construyó el dataset.

Los archivos que no son imágenes o están dañados se rechazan (400), igual que los que superan el tamaño (413). Si no se
distingue ninguna prenda (imagen lisa), la API responde 422 en lugar de inventar una categoría.

**Robustez.** El recorte fue decisivo. Con 2 000 imágenes de test simuladas como fotos (`ml/evaluate_products.py`):

| Escenario | Sin recorte | Con recorte |
|---|---:|---:|
| Foto de catálogo: 280 px, color, fondo blanco | 91,6 % | 92,0 % |
| Prenda ocupando el 33 % del encuadre | 15,2 % | 92,5 % |
| Fondo gris con ruido, prenda al 50 % | 17,5 % | 90,2 % |

**Limitaciones.** El modelo solo conoce siluetas en gris sobre fondo uniforme: fotos con fondos con textura, personas o
varias prendas degradan el resultado. Es un clasificador de conjunto cerrado (cualquier objeto recibe una de las 10
categorías) y no distingue detalles finos como colores o estampados. Análisis completo: `docs/evaluacion_productos.md`.

**Pruebas.** `ml/tests/test_products.py` (24 pruebas: preprocesado, servicio, API, coherencia con el entrenamiento y
modelo real) y pruebas E2E con Playwright de la página de clasificación en cuatro tamaños de pantalla, con auditoría de
accesibilidad.

#### 2.4.4 Clustering – métricas por vendedor, K-Means, elección de k (codo, silhouette)
#### 2.4.5 Resumen de texto – TextRank, ejemplos de resúmenes

**Problema.** Un marketplace acumula miles de reseñas y nadie puede leerlas todas. El componente resume **qué reclaman y
qué valoran los clientes** de un vendedor, de una categoría o de toda la plataforma, para que el equipo comercial sepa
dónde actuar (por ejemplo, un vendedor cuyos reclamos hablan sobre todo de productos distintos al anuncio).

**Datos.** `olist_order_reviews_dataset.csv` (Olist, §2.5): 99 224 reseñas reales, de las cuales 40 950 tienen comentario
en portugués. Son textos cortos (mediana de 53 caracteres, 1,2 oraciones útiles por comentario) y muy repetitivos: el
12 % de las oraciones aparecen literalmente en otras reseñas. No hace falta extraer datos de la web: el dataset ya trae
las reseñas, su nota (1–5) y el pedido al que pertenecen, que se une con vendedores y categorías.

**Método** (`backend/app/services/summary_service.py`, sin entrenamiento):

1. **Limpieza y segmentación.** Se corrigen saltos de línea y puntuación pegada (“iguais.Paguei”), se divide en oraciones,
   se descartan las de menos de 3 palabras (“Muito bom”) y se agrupan las que son iguales salvo tildes, mayúsculas o
   puntuación. Cada oración conserva **cuántas reseñas la contienen**.
2. **Reclamos y elogios.** Se separan por la nota real del cliente: reclamos (1–2) y elogios (4–5).
3. **Selección de frases: TextRank ponderado + MMR.** Las oraciones se representan con TF-IDF de n-gramas de caracteres
   (3–5), que toleran faltas de ortografía y variaciones de palabras del portugués. Sobre el grafo de similitud (coseno)
   se calcula PageRank con salto aleatorio proporcional a la frecuencia, de modo que lo que dicen muchos clientes pesa más.
   Después, **MMR** (*Maximal Marginal Relevance*, λ = 0,7) elige frases relevantes pero distintas entre sí, y un umbral
   de similitud (0,6) descarta paráfrasis. Para grupos grandes, solo entran al grafo las 400 oraciones más centrales.
4. **Temas.** Un léxico en portugués y español (insensible a tildes) marca 10 temas: plazo de entrega, pedido no recibido,
   pedido incompleto, fidelidad al anuncio, calidad, atención al cliente, cambios y devoluciones, embalaje, precio y
   satisfacción. Se calcula el porcentaje de reclamos que menciona cada tema y se compara con la media de la plataforma.
5. **Resumen ejecutivo en español**, armado con plantillas a partir de los indicadores (nota media, % de reclamos frente a
   la plataforma, temas principales y el tema que más se desvía de la media). No traduce ni inventa contenido.

**Por qué este enfoque.** Las reseñas están en portugués, son cortas y no existen resúmenes de referencia para entrenar un
modelo supervisado. Un modelo abstractivo (mT5 o un LLM) necesitaría GPU o una API externa y podría inventar afirmaciones.
El enfoque extractivo cita frases reales (con su número de repeticiones), responde en menos de medio segundo en CPU y es
auditable. El servicio mantiene la misma interfaz si en el futuro se sustituye por un modelo abstractivo.

**Evaluación** (`ml/evaluate_reviews.py` → `docs/evaluacion_resumen.md`). Al no haber resúmenes humanos de referencia, se
evaluó de dos formas reproducibles:

- **Los temas se validan con hechos de los pedidos que el texto no conoce.** Entre los reclamos, los marcados como
  *pedido no recibido* corresponden a pedidos que realmente no se habían entregado cuando se escribió la reseña en el
  67,8 % de los casos, frente al 26,1 % del resto (*lift* 2,59). Los marcados como *plazo de entrega* llegaron después
  de la fecha estimada en el 44,9 % de los casos, frente al 21,4 % (*lift* 2,10). Los controles (temas sin relación con
  la entrega) no muestran ese efecto (*lift* 0,15–0,38).
- **El método de resumen se compara con alternativas** sobre 120 grupos (reclamos y elogios de los 60 vendedores con más
  reseñas). Las métricas usan una representación por palabras distinta de la del método:

| Método (5 frases) | Cobertura de contenido ↑ | Reseñas representadas ↑ | Temas principales cubiertos ↑ | Redundancia ↓ |
|---|---:|---:|---:|---:|
| Frases al azar | 0,533 | 12,0 % | 59,9 % | 0,065 |
| Más frecuentes | 0,574 | 15,1 % | 59,3 % | 0,096 |
| Centroide TF-IDF | 0,771 | 18,9 % | 60,1 % | 0,453 |
| TextRank clásico | 0,774 | 18,8 % | 60,4 % | 0,429 |
| **TextRank ponderado + MMR (elegido)** | **0,790** | **20,0 %** | **73,6 %** | 0,305 |

El método elegido supera a TextRank clásico en todas las métricas de cobertura y reduce la redundancia: sin MMR, los cinco
reclamos “más representativos” de la plataforma son cinco formas de escribir “não recebi o produto”. El coste es que elige
frases algo más largas (13,1 frente a 10,7 palabras). Las frases al azar repiten menos solo porque no tienen relación
entre sí, pero cubren mucho menos contenido.

**Ejemplo** (toda la plataforma, 5 frases por sección):

> Se analizaron 99.224 reseñas (40.950 con comentario), con una nota media de 4,1 sobre 5. El 15% son reclamos (nota 1–2)
> y el 77% son elogios (nota 4–5). Los reclamos se concentran en: pedido no recibido (28%), atención al cliente (17%) y
> plazo de entrega (17%). Lo más valorado: plazo de entrega (42%) y calidad y estado del producto (12%).
>
> Reclamos frecuentes: “Não recebi o produto.” (×132) · “Ainda não recebi o produto...” (×73) · “Quero meu dinheiro de
> volta” (×30) · “O produto nao foi entregue.” (×27) · “Produto de péssima qualidade.” (×17)

Para un vendedor con nota media de 2,7, el resumen ejecutivo destaca que “fidelidad al anuncio aparece en el 33 % de sus
reclamos (media: 12 %)”, y las frases elegidas lo confirman (“Comprei o produto não veio o modelo que eu queria”). Más
ejemplos en `docs/evaluacion_resumen.md`, §4.

**Integración en la aplicación.**

| Endpoint | Uso |
|---|---|
| `POST /api/reviews/summarize` | Resume un texto pegado (español o portugués) y lista sus temas |
| `GET /api/reviews/seller/{seller_id}` | Indicadores, temas, reclamos y elogios representativos de un vendedor |
| `GET /api/reviews/category/{categoria}` | Lo mismo para una categoría de producto |
| `GET /api/reviews/overview` | Lo mismo para toda la plataforma |
| `GET /api/reviews/options` | Vendedores y categorías con más reseñas, para los selectores |

Todos aceptan `n_oraciones` (1–15 frases por sección). Las reseñas se preparan una vez por proceso (~2 s) y después cada
consulta de vendedor o categoría tarda menos de 0,5 s. Sin los CSV de Olist, el resumen de texto pegado sigue funcionando
y los demás endpoints responden 404 con un mensaje claro.

**Limitaciones.** Las frases se citan en portugués, el idioma original. El léxico de temas no entiende ironía ni
negaciones complejas (por eso se valida con datos de pedidos y se publica una muestra para revisión manual). Las reseñas
de nota 3 cuentan en los totales, pero no en reclamos ni en elogios.

**Pruebas.** `ml/tests/test_reviews.py` (17 pruebas: limpieza, temas en portugués y español, TextRank, MMR, indicadores
por grupo, API y datos reales de Olist) y pruebas E2E con Playwright de la página de reseñas en cuatro tamaños de
pantalla, con auditoría de accesibilidad.
### 2.5 Datos (Olist, Fashion-MNIST)

#### Fashion-MNIST

| | |
|---|---|
| Origen | Zalando Research (2017). Xiao, H., Rasul, K. y Vollgraf, R. *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. arXiv:1708.07747 |
| Licencia | MIT |
| Contenido | 70 000 imágenes de productos del catálogo de Zalando: 60 000 de entrenamiento y 10 000 de test |
| Formato | 28 × 28 px, escala de grises (0–255), producto claro y centrado sobre fondo negro |
| Clases | 10, equilibradas (7 000 imágenes cada una) |
| Obtención | Descarga automática con `torchvision.datasets.FashionMNIST` en `data/fashion_mnist/` (~30 MB, excluido de git) |
| Uso en el proyecto | Componente 3: 54 000 entrenamiento / 6 000 validación (estratificada) / 10 000 test |

Cada imagen se creó recortando la foto original del producto, escalando su lado mayor a 28 px y centrándola; el
preprocesado del backend reproduce ese proceso con las fotos que suben los usuarios (§2.4.3).

## 3. Tipo de población que puede usar el software
- Empresas de e-commerce y marketplaces
- Operadores logísticos
- Vendedores (pymes)
- Instituciones / gobierno (análisis del comercio electrónico regional)

## 4. Ventajas frente a otros softwares existentes

## 5. Conclusiones y trabajo futuro
