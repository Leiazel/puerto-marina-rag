# ⚽ Club Atlético Puerto Marina - Sistema RAG Híbrido (v3.0 Profesional)

Sistema integral de **Generación Aumentada por Recuperación (RAG)** de grado productivo para la gestión operativa, biomédica, táctica y logística de un club de fútbol profesional.

Arquitectura híbrida de última generación basada en **SQLite Híbrido** (**FTS5** con ranking **BM25**, filtrado de *stopwords* en español y normalización de diacríticos + almacenamiento binario en **BLOB** de vectores densos multilingües con **FastEmbed ONNX 384d** + **Caché Matricial en Memoria RAM** con producto punto BLAS/C de cero I/O + fusión de rankings mediante **Reciprocal Rank Fusion (RRF)**).

---

## 🚀 Novedades y Optimizaciones de la Versión 3.0

1. **Carga Multiformato de Documentos (`src/extractors.py` y `POST /api/upload`):**
   - **PDF:** Extracción estructurada por páginas (`## Página N`) mediante `pypdf`, con detección automática de documentos escaneados sin capa OCR.
   - **Word Moderno (.docx):** Mapeo de estilos a encabezados Markdown (`#`, `##`, `###`) y conversión automática de tablas de Word a tablas Markdown nativas mediante `python-docx`.
   - **Word Legado (.doc):** Parser binario de flujo OLE2 / WordDocument mediante `olefile` con decodificación de tablas de piezas y fallback robusto de secuencias de texto.
   - **Markdown y Texto Plano (.md, .txt):** Ingesta directa con detección y normalización UTF-8.
2. **Caché Matricial de Vectores en Memoria RAM (`src/retriever.py`):**
   - Elimina la lectura y deserialización repetitiva de miles de BLOBs desde SQLite en cada consulta.
   - La matriz de vectores normalizada se mantiene en memoria RAM protegida por locks de concurrencia y se invalida automáticamente cuando se agregan, modifican o eliminan documentos.
   - Permite búsquedas semánticas masivas en menos de **1 ms**.
3. **Blindaje de Seguridad y Prevención de Vulnerabilidades (`src/api.py` y `web/static/app.js`):**
   - **Prevención de Path Traversal:** Validación estricta de departamentos contra lista blanca, sanitización de nombres de archivo y códigos de documento (`re.sub`), y verificación de confinamiento en carpetas raíz permitidas.
   - **Validación de Firmas Binarias (Magic Bytes):** Verificación de cabeceras `%PDF-` y `PK\x03\x04` para bloquear archivos maliciosos renombrados.
   - **Protección contra Cross-Site Scripting (XSS):** Sanitización y escape HTML completo en la consola web (`escapeHtml`), con formateo seguro de fragmentos FTS (`formatSnippet`).
   - **Ajuste de Política CORS:** Eliminación de combinaciones inseguras de credenciales con comodines universales.
4. **Chunking Inteligente y Granularidad Semántica (`src/ingestion.py`):**
   - División jerárquica de párrafos excesivos mediante detección de oraciones y ventanas deslizantes con solapamiento (`overlap`).
   - Preservación de encabezados de tablas Markdown al particionar tablas extensas.
   - Descarte automático de fragmentos vacíos o triviales (`---`, encabezados aislados sin cuerpo).
   - Generación de tarjetas individuales (`###`) para cada futbolista y miembro del staff, eliminando la dilución de similitud vectorial.
5. **Consola Web Moderna con Drag-and-Drop y Eliminación en Cascada:**
   - Nueva pestaña **"📤 Subir Archivos"** con zona de arrastre interactiva, vista previa de cola y estado de indexación en vivo.
   - Soporte para eliminación permanente de documentos y sus chunks en cascada (`DELETE /api/documents/{id}`).
6. **Soporte CLI para Carga Masiva (`src/cli.py`):**
   - Argumento `--upload <archivos...>` para indexar documentos directamente desde la terminal con asignación departamental y etiquetas.
7. **Suite de 20 Pruebas Unitarias Automatizadas con Base de Datos Aislada (`tests/test_rag.py`):**
   - Pruebas 100% aisladas en base de datos temporal efímera (cero polución de la base de datos operativa).
   - Cobertura exhaustiva: extractores PDF/DOCX/DOC, seguridad, deduplicación SHA-256, caché matricial y endpoints FastAPI.

---

## 🏗️ Arquitectura del Sistema

```
                       ┌──────────────────────────────────────────────┐
                       │             CONSULTA DEL USUARIO             │
                       └──────────────────────┬───────────────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
     ┌─────────────────────────────┐                    ┌───────────────────────────────┐
     │  Búsqueda Léxica Exacta     │                    │  Búsqueda Semántica Densa     │
     │  SQLite FTS5 + BM25         │                    │  Caché Matricial en Memoria   │
     │  (Stopwords limpiadas)      │                    │  (Producto Punto BLAS / C)    │
     └──────────────┬──────────────┘                    └───────────────┬───────────────┘
                    │                                                   │
                    └─────────────────────────┬─────────────────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │ Reciprocal Rank Fusion (RRF)│
                               │ RRF = α/(K+r_lex) + (1-α)... │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │  Top-K Chunks Re-rankeados  │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │  Generador de Síntesis      │
                               │  (Local o LLM Universal)    │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │  Auditoría en SQLite (Logs) │
                               └─────────────────────────────┘
```

---

## 🏛️ El Club: Club Atlético Puerto Marina

- **Fundación:** 1928 | **Apodo:** *El Tiburón de la Costa* | **Colores:** Azul Marino y Dorado.
- **Estadio:** Costa Brava (32.000 espectadores).
- **Competencia:** Torneo Clausura (Semifinales) y Copa Nacional.

### Departamentos y Servicios Reales Modelados
1. **Departamento Médico y Fisioterapia:** Dr. Federico Arismendi (Jefe Médico) y Lic. Valentina Russo (Kinesióloga). Partes médicos diarios, evolución de lesiones (**Lautaro Carrizo #6**, **Franco Valenzuela #7**), protocolo de 5 fases de Retorno al Juego (RTP).
2. **Preparación Física y Monitoreo GPS:** Prof. Germán Silveira. Cargas Catapult Vector S7, distancias totales, High Speed Running, sprints, ratio ACWR y test de fatiga CMJ.
3. **Nutrición y Rendimiento Metabólico:** Lic. Sofía Carrizo (ISAK 1). Cineantropometría, planes de glucógeno y **protocolo estricto Sin TACC para Agustín Ríos (#14)**.
4. **Cuerpo Técnico y Análisis Táctico:** Javier Méndez (DT) y Pablo D'Angelo (AT). Microciclos, scouting de *Deportivo Los Andes*, balón parado *"Alfa Marina"*, informe arbitral y alineaciones.
5. **Scouting e Inteligencia Deportiva:** Lucas Benítez. Informes de fichajes (Rodrigo "Chelo" Vargas) y métricas ofensivas avanzadas (xG/xA).
6. **Operaciones, Logística y Viajes:** Carlos "Charly" Pereyra. Vuelo chárter AndesJet AR-4912, Hotel Grand Bahía (habitaciones y gabinete de crioterapia).
7. **Prensa, Comunicación y Reglamento Interno:** Mariana Gómez Solís. Zona mixta, régimen disciplinario y tabla oficial de multas en USD.

---

## 🚀 Instalación y Puesta en Marcha

### 1. Dependencias
```bash
pip install -r requirements.txt
```

### 2. Configuración Opcional (`.env`)
Copiar plantilla de entorno:
```bash
cp .env.example .env
```
Por defecto, `USE_EXTERNAL_LLM=false` utiliza el sintetizador local extractivo de alto rendimiento (0 costo, 0 dependencias externas). Si deseas activar un LLM (Ollama, Groq, Gemini u OpenAI), edita `.env`.

### 3. Iniciar el Servidor Web & API
```bash
./run_server.sh
```
- **Consola Web:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Documentación Swagger / OpenAPI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4. Consultas y Carga por Terminal (CLI)
```bash
# Consulta puntual
PYTHONPATH=. python src/cli.py --query "¿Cuál es la lesión de Lautaro Carrizo y cuándo vuelve a jugar?"

# Consulta con filtro por departamento
PYTHONPATH=. python src/cli.py --dept preparacion_fisica --query "¿Qué jugador tiene alerta de fatiga en el GPS?"

# Consulta de datos relacionales indexados
PYTHONPATH=. python src/cli.py --query "¿Cuál es el correo y teléfono de la nutricionista Sofía Carrizo?"

# Subir e indexar archivos multiformato (PDF, DOCX, DOC, MD, TXT)
PYTHONPATH=. python src/cli.py --upload informe_rival.pdf ficha_fichaje.docx --dept scouting --author "Lucas Benítez"

# Estadísticas del sistema
PYTHONPATH=. python src/cli.py --stats

# Modo interactivo en vivo
PYTHONPATH=. python src/cli.py --interactive
```

### 5. Suite de Pruebas Automatizadas
```bash
PYTHONPATH=. python -m unittest tests.test_rag -v
```
*(20 pruebas unitarias con base de datos efímera aislada que validan extractores, seguridad, caché vectorial, deduplicación, FTS5 y RAG end-to-end).*

---

## 📡 Endpoints de la API REST

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/status` | Estado del sistema, recuentos de documentos, chunks, estado del LLM y latencias |
| `POST` | `/api/query` | Ejecución de consulta RAG híbrida con citas, fragmentos y confianza dinámica |
| `POST` | `/api/upload` | **(Nuevo v3.0)** Carga múltiple de archivos (PDF, DOCX, DOC, MD, TXT) con deduplicación e indexación |
| `GET` | `/api/documents` | Listado de documentos y desglose por departamento |
| `GET` | `/api/documents/{id}` | Contenido completo de un documento y sus chunks asociados |
| `DELETE` | `/api/documents/{id}` | **(Nuevo v3.0)** Eliminación en cascada de documento, chunks y registros FTS |
| `POST` | `/api/documents` | Ingesta de documento Markdown con protección contra Path Traversal |
| `GET` | `/api/players` | Fichas biomédicas y técnicas del plantel profesional |
| `GET` | `/api/staff` | Directorio de cuerpo técnico y especialistas |
| `GET` | `/api/departments` | Catálogo de departamentos operativos |
| `GET` | `/api/llm/health` | Diagnóstico de conectividad y modelos activos del LLM configurado |
| `GET` | `/api/logs` | Auditoría de consultas ejecutadas con telemetría de milisegundos |
