# Evaluación del módulo de resumen de opiniones y reclamos (Olist)

Generado con `python ml/evaluate_reviews.py`. Todos los números se recalculan con ese comando.

## 1. Datos

- Reseñas: **99 224**; con comentario: **40 950** (41.3%).
- Longitud mediana del comentario: 53 caracteres; oraciones útiles por comentario (≥ 3 palabras): 1.18.
- Reclamos (nota 1–2): 14.7% · neutras (3): 8.2% · elogios (4–5): 77.1%.
- Oraciones repetidas literalmente (misma clave normalizada): 12.4% de las apariciones. Por eso el resumen pondera cada frase por el número de reseñas que la contienen.

## 2. Detección de temas

### 2.1 Cobertura

| Grupo | Reseñas | Con algún tema |
|---|---:|---:|
| reclamos | 10 889 | 81.4% |
| neutras | 3 556 | 73.3% |
| elogios | 26 505 | 79.6% |

| Tema | % de reclamos | % de elogios |
|---|---:|---:|
| Plazo de entrega | 16.8% | 42.0% |
| Pedido no recibido | 28.1% | 0.8% |
| Pedido incompleto | 11.2% | 1.1% |
| Fidelidad al anuncio | 12.4% | 6.8% |
| Calidad y estado del producto | 13.9% | 12.2% |
| Atención al cliente | 17.4% | 3.7% |
| Cambios y devoluciones | 14.3% | 0.9% |
| Embalaje | 4.2% | 4.4% |
| Precio | 5.9% | 4.4% |
| Satisfacción y recomendación | 7.5% | 50.4% |

### 2.2 Validación con hechos de los pedidos

El detector solo lee el texto; las fechas de entrega salen de `olist_orders_dataset.csv`. *Lift* = tasa con el tema / tasa sin el tema, dentro del mismo grupo de notas (así la nota no influye).

| Tema | Grupo | Hecho comprobado | Con tema | Sin tema | Lift |
|---|---|---|---:|---:|---:|
| Pedido no recibido | reclamos | Pedido sin entregar cuando se escribió la reseña | 67.8% (n=3061) | 26.1% | 2.59 |
| Plazo de entrega | reclamos | Entregado después de la fecha estimada | 44.9% (n=1828) | 21.4% | 2.1 |
| Plazo de entrega | elogios | Entregado antes de la fecha estimada | 96.8% (n=11141) | 94.8% | 1.02 |
| Pedido incompleto | reclamos | Control: incompleto frente a sin entregar | 15.3% (n=1216) | 40.6% | 0.38 |
| Calidad y estado del producto | reclamos | Control: calidad frente a retraso | 4.4% (n=1513) | 28.7% | 0.15 |

### 2.3 Muestra para revisión manual (reclamos, semilla 42)

- **Plazo de entrega:** “APESAR DE TER CUMPRIDO PRAZO DE ENTREGA, NÃO RECEBI O PRODUTO CONFORME NF” · “ainda não obtive resposta do porquê do atraso” · “NENHUMA ESCLARECIMENTO SOBRE ATRASO DA ENTREGA.” · “Esta super atrasados meu pedido, mais estão acompanhando, entrando em contato comigo.”
- **Pedido no recibido:** “EU NAO RECEBI O PRODUTO AINDA NAO” · “Infelizmente ainda não foi entregue!  Primeira compra na loja parceira da stark e tive essa surpresa... devemos ficar mais atento a isso!” · “Não recebi as botinhas” · “não recebi”
- **Pedido incompleto:** “Comprei 2 puffs e um chromebook. A entrega foi incompleta. Os puffs nao foram entregues. Tento telefonar para reclamar e nenhum dos telefone” · “Falta entregar um dos produtos” · “Boa tarde, pedi 3 gorros mas só recebi, é agora que faço?” · “não receci todas as peças compradas, somente metade das peças compradas”
- **Fidelidad al anuncio:** “a cor do produto no site é diferente do produto recebido.” · “Comprei pensando q seria igual a foto quando chegou totalmente diferente” · “Preciso trocar os copos pois vieram errados. O produto que eu pedi foram 2 copos do liquidificador Philips Walesa código RI 2044 na cor PRET” · “A descrição do produto não condiz com o que é! Na descrição da compra está: Composição: 100% Algodão. No entanto, na embalagem do produto a ”
- **Calidad y estado del producto:** “Dia 18 de dezembro de 2017, o produto veio com uma das partes quebrada e outra mal confeccionada (com um encaixe mal feito).  Solicito a tro” · “Veio estragada, os ilhoes soltos. N indico” · “O produto chegou em 2 entregas, só que bem mal embalado, sem aviso de frágil, num papel pardo. Entregaram 2 pacotes em um dia e um pacote no” · “Produto nao e original , de mau qualidade nunca mais compro com ele , as tesouras nao fecham bem vagabunda mesmo tipo deve custar uns 10 rea”
- **Atención al cliente:** “NAO TIVE RETORNO E NAO RECEBI A MERCADORIA COM MAIS DE 20 DIAS DE PEDIDO APROVADO” · “comprei 2 toner hp ce278a e só recebi 1 e ainda emitiram a nota fiscal com 2 produtos. Estou até agora, esperando a entrega do segundo toner” · “fiz uma compra, meus pedidos vieram separados até ai tudo bem, mas ficou faltando um produto e até agora não foi entregue! e isso que paguei” · “Foi anunciado um produto e veio outro, totalmente inferior ao que eu comprei, enviei um e-mail para lannister e até agora não se resolveu na”
- **Cambios y devoluciones:** “Irresponsável, não tem nenhuma preocupação com o cliente pois cancela a venda na véspera da entrega após esperar 45dias. Fui Obrigada a colo” · “Quero devolver pois recebi algo que não pedi” · “Recebi um email relatando que o produto teria sido extraviado e fiz o cancelamento pois era a bolsa de um bebê Comprei outra e agora recebi ” · “nao responem email, ate hoje nao recebi o produto e nao dao satisfacao quero meu dinheiro de volta”
- **Embalaje:** “Muito ruim!! Embalaram ele dobrado e por isso quando enche ele é difícil desamassar!! Péssimo... Estou insatisfeito com o produto... Desapon” · “Todos os produtos chegaram quebrados e um absurdo isso eles foram muito mau embalados e não tinha especificação na caixa falando q era produ” · “O relógio é simples, está ok. A embalagem veio amassada.” · “A caixa do produto chegou amassada. Os correios informaram q o vendedor nao optou pela embalagem apropriada e quem deveria responder por iss”
- **Precio:** “Foi enviado produto Diferente daquele do anúncio. Solicitei devolver o produto e receber o valor pago, porém ainda não tive resposta. Espero” · “Achei o prazo para entrega muito longo (40 dias) e a loja ainda atrasou um dia. Pela primeira vez, comprei um produto no baratheon e esperei” · “Aguardo a entrega do que comprei, no dia 10/02/2018. Ate agora, nada de produto. Estou aguardando todo esse tempo, somente pelo atendimento ” · “Veio uma porcariaaaaa nao vale o preço que é ..”
- **Satisfacción y recomendación:** “Considerando o site stark nunca tive problema algum e recomendo ao meus amigos porém está empresa que está atendendo com a venda do produto ” · “Não gostei porque a empresa enviou um envelope vazio.” · “Comprei o produto e não recebi já vai fazer um mês, muito ruim não recomendo.” · “Não recomendo pois meu pedido veio errado não foi o produto que comprei veio um totalmente diferente”

## 3. Comparación de métodos de resumen

120 grupos (reclamos y elogios de los 60 vendedores con más reseñas y al menos 15 reclamos comentados); 5 frases por resumen.

- **Cobertura de contenido:** coseno entre las palabras del resumen y las del grupo (ponderadas por repetición).
- **Reseñas representadas:** % de oraciones del grupo cuyo vocabulario coincide en coseno ≥ 0.5 con alguna frase del resumen.
- **Cobertura de temas:** de los 3 temas principales del grupo, cuántos aparecen en el resumen.
- **Redundancia:** similitud media entre las frases del resumen (menor es mejor).
- Las métricas usan palabras; el método usa n-gramas de caracteres, para no evaluarse con su propia representación.

### Todos

| Método | Cobertura contenido ↑ | Reseñas representadas ↑ | Cobertura temas ↑ | Redundancia ↓ | Palabras/frase |
|---|---:|---:|---:|---:|---:|
| Aleatorio (5 semillas) | 0.533 | 12.0% | 59.9% | 0.065 | 10.0 |
| Más frecuentes | 0.574 | 15.1% | 59.3% | 0.096 | 8.7 |
| Centroide TF-IDF | 0.771 | 18.9% | 60.1% | 0.453 | 10.2 |
| TextRank clásico | 0.774 | 18.8% | 60.4% | 0.429 | 10.7 |
| TextRank ponderado | 0.774 | 18.9% | 60.7% | 0.433 | 10.5 |
| TextRank ponderado + MMR λ=0.5 | 0.779 | 19.9% | 76.2% | 0.217 | 14.5 |
| TextRank ponderado + MMR λ=0.7 sin umbral de paráfrasis | 0.789 | 20.0% | 72.8% | 0.309 | 13.0 |
| TextRank ponderado + MMR λ=0.7 (elegido) | 0.790 | 20.0% | 73.6% | 0.305 | 13.1 |
| TextRank ponderado + MMR λ=0.9 | 0.784 | 19.5% | 66.2% | 0.365 | 11.9 |

### Reclamos

| Método | Cobertura contenido ↑ | Reseñas representadas ↑ | Cobertura temas ↑ | Redundancia ↓ | Palabras/frase |
|---|---:|---:|---:|---:|---:|
| Aleatorio (5 semillas) | 0.505 | 10.2% | 59.2% | 0.053 | 11.6 |
| Más frecuentes | 0.531 | 11.9% | 63.3% | 0.074 | 10.6 |
| Centroide TF-IDF | 0.727 | 13.9% | 59.4% | 0.379 | 11.9 |
| TextRank clásico | 0.724 | 13.7% | 60.0% | 0.362 | 12.6 |
| TextRank ponderado | 0.725 | 13.7% | 60.0% | 0.367 | 12.4 |
| TextRank ponderado + MMR λ=0.5 | 0.719 | 14.7% | 80.0% | 0.156 | 17.8 |
| TextRank ponderado + MMR λ=0.7 sin umbral de paráfrasis | 0.729 | 14.7% | 75.6% | 0.225 | 16.1 |
| TextRank ponderado + MMR λ=0.7 (elegido) | 0.730 | 14.7% | 76.7% | 0.220 | 16.3 |
| TextRank ponderado + MMR λ=0.9 | 0.731 | 14.3% | 66.7% | 0.285 | 14.4 |

### Elogios

| Método | Cobertura contenido ↑ | Reseñas representadas ↑ | Cobertura temas ↑ | Redundancia ↓ | Palabras/frase |
|---|---:|---:|---:|---:|---:|
| Aleatorio (5 semillas) | 0.561 | 13.7% | 60.7% | 0.076 | 8.3 |
| Más frecuentes | 0.616 | 18.3% | 55.3% | 0.119 | 6.8 |
| Centroide TF-IDF | 0.815 | 23.8% | 60.8% | 0.527 | 8.4 |
| TextRank clásico | 0.824 | 23.9% | 60.8% | 0.497 | 8.8 |
| TextRank ponderado | 0.823 | 24.0% | 61.4% | 0.499 | 8.6 |
| TextRank ponderado + MMR λ=0.5 | 0.839 | 25.1% | 72.5% | 0.279 | 11.2 |
| TextRank ponderado + MMR λ=0.7 sin umbral de paráfrasis | 0.849 | 25.2% | 70.0% | 0.393 | 9.8 |
| TextRank ponderado + MMR λ=0.7 (elegido) | 0.850 | 25.2% | 70.6% | 0.390 | 9.9 |
| TextRank ponderado + MMR λ=0.9 | 0.838 | 24.7% | 65.8% | 0.444 | 9.4 |

En reseñas representadas, el método elegido iguala o supera a TextRank clásico en el 83.3% de los grupos, al centroide en el 80.0% y al azar en el 95.0%.

### Decisión

Se elige **TextRank ponderado por frecuencia + MMR (λ = 0.7)** con n-gramas de caracteres. Frente a TextRank clásico cubre más contenido (0.790 frente a 0.774), representa más reseñas (20.0% frente a 18.8%), recoge más temas principales (73.6% frente a 60.4%) y repite menos (0.305 frente a 0.429; centroide: 0.453). Sin MMR, TextRank y el centroide eligen variantes de la misma frase (“não recebi o produto” escrito de cinco formas). El coste es que las frases elegidas son algo más largas (13.1 frente a 10.7 palabras). λ = 0.5 diversifica aún más (temas 76.2%, redundancia 0.217) pero elige frases más largas (14.5 palabras) y cubre menos contenido (0.779); λ = 0.9 se acerca a TextRank sin MMR. Ponderar por frecuencia casi no cambia las métricas, porque el filtro de candidatas ya usa la frecuencia, pero hace que la frase que comparten más clientes salga primero. El umbral de paráfrasis (coseno > 0.6) mejora poco frente a no usarlo (redundancia 0.309 → 0.305, temas 72.8% → 73.6%), pero evita que en vendedores con pocas reseñas salgan dos versiones de la misma frase. Ningún método supera al azar en redundancia: el azar elige frases sin relación entre sí, a costa de cubrir mucho menos (0.533).

### Ejemplo: los 5 reclamos más representativos de toda la plataforma

| TextRank clásico | Centroide TF-IDF | TextRank ponderado + MMR (elegido) |
|---|---|---|
| nao recebi o produto ainda | Não recebi o produto que eu comprei | Não recebi o produto. |
| EU NAO RECEBI O PRODUTO AINDA NAO | AINDA NÃO RECEBI MEU PRODUTO, JÁ FAZ UM MÊS QUE COMPREI E AINDA NÃO RECEBI O PRODUTO | Ainda não recebi o produto... |
| produto que não recebi ainda | produto que não recebi ainda | Quero meu dinheiro de volta |
| Ainda não recebi meu produto | Comprei e não recebi o produto. | O produto nao foi entregue. |
| Não recebi meu produto ainda | EU NAO RECEBI O PRODUTO AINDA NAO | Produto de péssima qualidade. |

## 4. Ejemplos de resúmenes

### Toda la plataforma

> Se analizaron 99.224 reseñas (40.950 con comentario), con una nota media de 4,1 sobre 5. El 15% son reclamos (nota 1–2) y el 77% son elogios (nota 4–5). Los reclamos se concentran en: pedido no recibido (28%), atención al cliente (17%) y plazo de entrega (17%). Lo más valorado: plazo de entrega (42%) y calidad y estado del producto (12%).

**Reclamos representativos**

- Não recebi o produto. (×132)
- Ainda não recebi o produto... (×73)
- Quero meu dinheiro de volta (×30)
- O produto nao foi entregue. (×27)
- Produto de péssima qualidade. (×17)

**Elogios representativos**

- Chegou antes do prazo! (×166)
- Produto muito bom (×87)
- Entrega antes do prazo. (×98)
- Recomendo a todos. (×91)
- Entrega super rápida. (×69)

### Vendedor 2eb70248… (`2eb70248d66e0e3ef83659f71b244378`)

> Se analizaron 199 reseñas (125 con comentario), con una nota media de 2,7 sobre 5. El 50% son reclamos (nota 1–2) y el 36% son elogios (nota 4–5). La tasa de reclamos está por encima de la media de la plataforma (15%). Los reclamos se concentran en: fidelidad al anuncio (33%), calidad y estado del producto (29%) y atención al cliente (20%). Destaca frente a la plataforma: fidelidad al anuncio aparece en el 33% de sus reclamos (media: 12%). Lo más valorado: plazo de entrega (56%) y embalaje (28%).

**Reclamos representativos**

- não recebi o produto
- Comprei o produto não veio o modelo que eu queria
- Não recebi o produto que comprei, chegou outro modelo menor e diferente e não consigo entrar em contato com o atendimento das lojas lannister.
- Produto que enviaram não foi o produto comprado.
- Comprei um relógio Casio e oque me foi entregue não bate com o número de série que comprei,, pelo que eu identifique parece uma réplica gostaria muito de devolver este produto

**Elogios representativos**

- O produto chegou antes do prazo.
- Chegou relativamente rápido, antes do prazo previsto, produto lindo, condizente a foto, e veio em perfeito estado, muito bem embalado.
- Produto original, entregue antes do prazo esperado, entrega super rápida, produto bem embalado e lacrado.
- Produto muito bonito, leve e entregue antes do prazo!
- Gostei do produto e da rapidez ao ser entregue, mais não fiquei satisfeita pela embalagem que veio achei muito fuleira pois a caixa do casio veio toda amassada e de péssima qualidade.

### Vendedor fe2032da… (`fe2032dab1a61af8794248c8196565c9`)

> Se analizaron 290 reseñas (126 con comentario), con una nota media de 4,4 sobre 5. El 10% son reclamos (nota 1–2) y el 85% son elogios (nota 4–5). La tasa de reclamos está por debajo de la media de la plataforma (15%). Los reclamos se concentran en: pedido no recibido (40%), plazo de entrega (20%) y fidelidad al anuncio (10%). Lo más valorado: plazo de entrega (48%) y precio (9%).

**Reclamos representativos**

- Não recebi o produto
- Fiz um pedido com dois perfumes mas so foi entregue um produto
- aconteceu algo na entrega e eu não recebi ainda o produto.
- Consta no site que o produto foi entregue, mas não recebi.
- O perfume que recebi tem embalagem diferente da foto do site e também diferente de um perfume igual que tenho em casa.

**Elogios representativos**

- produto chegou muito bem embalado e a entrega foi rápida
- Entrega muito rápida antes do prazo
- Gostei do produto que comprei veio bem embalado o prazo de entrega nem se fala sensacional chegou muito rapido, futuramente vou fazer mais compras recomendo essa loja pode comprar sem medo.
- Trabalho de qualidade entregue no prazo produto em perfeito estado bem embalado recomendo!
- Entrega super rápida e produto perfeito.

### Cama mesa banho (`cama_mesa_banho`)

> Se analizaron 9.432 reseñas (4.320 con comentario), con una nota media de 4,0 sobre 5. El 17% son reclamos (nota 1–2) y el 73% son elogios (nota 4–5). La tasa de reclamos es similar a la media de la plataforma (15%). Los reclamos se concentran en: pedido no recibido (27%), fidelidad al anuncio (15%) y calidad y estado del producto (14%). Lo más valorado: plazo de entrega (33%) y calidad y estado del producto (15%).

**Reclamos representativos**

- Nao recebi o produto (×12)
- Ainda não recebi o produto... (×14)
- O produto não foi entregue (×4)
- o produto ainda nao chegou (×4)
- Não gostei do produto. (×5)

**Elogios representativos**

- Chegou antes do prazo. (×18)
- produto de boa qualidade (×15)
- Entrega antes do prazo (×12)
- produto muito bom. (×11)
- Produto de ótima qualidade. (×11)

## 5. Tiempo de respuesta (CPU, sin GPU)

- Carga y preparación de las reseñas (una vez por proceso): 2.39 s
- Resumen de toda la plataforma (primera vez; luego queda en caché): 4.11 s
- Resumen de un vendedor o categoría: ≤ 0.48 s
- Texto pegado (60 oraciones): 0.003 s

## 6. Limitaciones

- Es un resumen **extractivo**: las frases salen tal cual de las reseñas, en portugués. El resumen ejecutivo en español se arma con plantillas a partir de los indicadores; no traduce ni inventa contenido.
- Los temas se detectan por léxico: es transparente y auditable, pero no entiende negaciones complejas ni ironía. Los lifts de la sección 2.2 muestran que aun así capturan señales reales.
- Reclamo y elogio se definen por la nota (1–2 y 4–5). Las reseñas de nota 3 cuentan en los totales pero no en esas dos secciones.
- Un modelo abstractivo (p. ej. mT5 o un LLM) podría redactar el resumen en español, pero requiere GPU o una API externa. La interfaz del servicio permite sustituirlo sin cambiar el frontend.
