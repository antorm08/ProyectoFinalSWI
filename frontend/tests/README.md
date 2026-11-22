# Verificación del frontend

## Requisitos

- Dependencias locales: `npm ci`.
- API real en `http://localhost:8000`.
- Microsoft Edge instalado. La configuración utiliza `channel: "msedge"` y no descarga un navegador.
- Para probar la inferencia de imágenes debe existir el modelo CNN y estar disponible PyTorch.
- El caso de reseñas por vendedor sin datos espera un identificador inexistente y una respuesta 404.

## Comandos

```powershell
npm.cmd run build
npm.cmd run test:e2e
npm.cmd run test:e2e:report
```

Playwright reutiliza el frontend en el puerto 5173 o lo inicia si no está disponible. La API debe iniciarse por separado; las pruebas no entrenan modelos ni modifican datasets.

Se comprueban las seis rutas a 375, 768, 1024 y 1440 píxeles, errores JavaScript, desbordamiento horizontal, reglas automatizadas WCAG A/AA con axe, navegación por teclado, menú móvil y contratos HTTP de las interacciones.

Las solicitudes de predicción, demanda, imágenes y reseñas se envían a la API real. La imagen de diagnóstico y el texto de prueba son entradas de prueba, no registros comerciales. Solo se interceptan peticiones para simular fallos de red o retrasar una respuesta real y comprobar su cancelación.

Se guardan capturas y resultados en `test-results/`, y el informe HTML en `playwright-report/`; ambos directorios están ignorados por Git.

## Alcance

Las comprobaciones automatizadas de accesibilidad no sustituyen una evaluación manual con tecnologías de asistencia. Los estados con modelos o datasets ausentes no acreditan el funcionamiento de los estados entrenados. Si cambian los datos locales, los resultados de integración también pueden cambiar.

El dashboard no interpreta presencia de archivos como precisión comprobada. Los gráficos de vendedores se limitan a las filas devueltas por la API (máximo 300). Los datos sintéticos de demanda permanecen identificados como demostración.
