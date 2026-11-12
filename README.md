# ⚽ Club Atlético Puerto Marina - Sistema RAG Híbrido (v2.2)

Sistema integral de **Generación Aumentada por Recuperación (RAG)** de grado productivo para la gestión operativa, biomédica, táctica y logística de un club de fútbol profesional.

Arquitectura optimizada basada en **SQLite Híbrido** (**FTS5** con ranking **BM25** y filtrado de *stopwords* en español + almacenamiento binario en **BLOB** de vectores densos multilingües con **FastEmbed ONNX** + búsqueda vectorial en dos fases + fusión de rankings mediante **Reciprocal Rank Fusion (RRF)**).

---

## 🛠️ Novedades y Optimizaciones (v2.2)

1. **Filtrado de *Stopwords*, Normalización de Diacríticos y Términos Técnicos en FTS5 (`src/retriever.py`):**
   Elimina conectores vacíos (*"como"*, *"cual"*, *"para"*, *"los"*, *"del"*, etc.) y normaliza tildes antes de armar la consulta `MATCH` con BM25 ponderado por columna, preservando siglas tácticas/métricas de 2 letras (`xG`, `xA`, `DT`, `AT`, `PF`, `ID`).
2. **Búsqueda Vectorial en Dos Fases y Modos Puros (`src/retriever.py`):**
   - **Fase 1:** Carga únicamente los pares `(id, embedding)` y ejecuta el producto punto matricial en BLAS/C.
   - **Fase 2:** Hidrata texto y metadatos **únicamente de los Top-K ganadores**, respetando `alpha=1.0` (léxico puro) y `alpha=0.0` (semántico puro).
3. **Chunking Consciente de Tablas y Limpieza de Huérfanos (`src/ingestion.py`):**
   Fragmentación ajustada a la ventana de MiniLM (`900` chars / `180` overlap), preservando encabezados (`| col |` + `|---|`) al dividir tablas largas, evitando chunks huérfanos de solo títulos o separadores `---`, y limpiando documentos obsoletos al reindexar.
4. **Puente Relacional SQL-a-RAG (`src/ingestion.py`):**
   Las tablas estructuradas `players` y `staff` se sincronizan automáticamente como documentos virtuales fragmentados y vectorizados.
5. **Sintetizador Extractivo Mejorado, Guardia Out-of-Domain y Telemetría (`src/generator.py`):**
   Reconstruye múltiples tablas Markdown por fragmento, detecta consultas fuera de dominio (`insufficient_evidence`), protege prompts con delimitadores XML y reporta `fallback_used` / `fallback_reason`.
6. **Adaptador Universal para LLMs (`src/config.py` y `src/generator.py`):**
   Funciona 100% local/offline (`USE_EXTERNAL_LLM=false`) o conectado a Ollama (sin API key requerida), Groq, Gemini u OpenAI, con recarga segura de `.env`, backoff ante HTTP 429, auto-descubrimiento ante HTTP 404 y limpieza de bloques `<think>`.
7. **Suite de 13 Pruebas Automatizadas Aisladas (`tests/test_rag.py`):**
   Ejecución sobre base de datos y directorio temporal (`tempfile`) para no mutar la base productiva, cubriendo motor híbrido, chunking, LLM mockeado y seguridad/validación en FastAPI (`TestClient`).

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
     │  SQLite FTS5 + BM25         │                    │  Fase 1: BLOBs IDs + Coseno   │
     │  (Stopwords limpiadas)      │                    │  Fase 2: Hidratación Top-K    │
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
1. **Departamento Médico y Fisioterapia:**
   - *Responsables:* Dr. Federico Arismendi (Jefe Médico) y Lic. Valentina Russo (Kinesióloga).
   - Partes médicos, evolución de lesiones (**Lautaro Carrizo #6**, **Franco Valenzuela #7**), protocolo de 5 fases de Retorno al Juego (Return To Play - RTP).
2. **Preparación Física y Monitoreo GPS:**
   - *Responsable:* Prof. Germán Silveira.
   - Datos de dispositivos Catapult Vector S7, distancias totales, High Speed Running (> 19.8 km/h), sprints (> 25.2 km/h), ratio ACWR (alerta amarilla 1.48) y test de fatiga neuromuscular CMJ.
3. **Nutrición y Rendimiento Metabólico:**
   - *Responsable:* Lic. Sofía Carrizo (ISAK 1).
   - Antropometría mensual de 6 pliegues, pautas de glucógeno y suplementación, y **protocolo estricto Sin TACC para Agustín Ríos (#14)**.
4. **Cuerpo Técnico y Análisis Táctico:**
   - *Responsables:* Javier Méndez (DT) y Pablo D'Angelo (AT).
   - Microciclos semanales (MD-4 a MD+1), scouting del rival *Club Deportivo Los Andes*, balón parado *"Alfa Marina"*, informe del árbitro Pitres y alineación oficial para semifinales.
5. **Scouting e Inteligencia Deportiva:**
   - *Responsable:* Lucas Benítez.
   - Fichas de seguimiento para fichajes (Rodrigo "Chelo" Vargas) y comparativa de efectividad ofensiva Santos vs Cáceres (xG/xA).
6. **Operaciones, Logística y Viajes:**
   - *Responsable:* Carlos "Charly" Pereyra.
   - Plan operativo de semifinal: vuelo chárter AndesJet AR-4912, Hotel Grand Bahía (pisos 4 y 5 exclusivos), y acondicionamiento de Hab. 402 como gabinete médico y crioterapia.
7. **Prensa, Comunicación y Reglamento Interno:**
   - *Responsable:* Mariana Gómez Solís.
   - Protocolos de zona mixta, temas restringidos y **Reglamento Interno de Disciplina** con multas en USD (retrasos, pesaje de los viernes y celulares en vestuario).

---

## 🚀 Instalación y Puesta en Marcha

### 1. Entorno Virtual y Dependencias
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Reindexación Inicial (opcional)
La base `data/database/club_knowledge.db` ya viene pre-indexada y lista para usarse; si modificas o agregas documentos en `data/raw_documents/`, podés regenerarla ejecutando:
```bash
.venv/bin/python -m src.ingestion
```

### 3. Configuración Opcional (`.env`)
Copiar plantilla de entorno:
```bash
cp .env.example .env
```
Por defecto, `USE_EXTERNAL_LLM=false` utiliza el sintetizador local extractivo de alto rendimiento (0 costo, 0 dependencias). Si deseas activar un LLM (Ollama, Groq, Gemini u OpenAI), edita `.env`.

### 4. Iniciar el Servidor Web & API
```bash
./run_server.sh
```
- **Consola Web:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Documentación Swagger:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 5. Consultas por Terminal (CLI)
```bash
# Consulta puntual
.venv/bin/python -m src.cli --query "¿Cuál es la lesión de Lautaro Carrizo y cuándo vuelve a jugar?"

# Consulta con filtro por departamento
.venv/bin/python -m src.cli --dept preparacion_fisica --query "¿Qué jugador tiene alerta de fatiga en el GPS?"

# Consulta de datos relacionales indexados
.venv/bin/python -m src.cli --query "¿Cuál es el correo y teléfono de la nutricionista Sofía Carrizo?"

# Estadísticas del sistema
.venv/bin/python -m src.cli --stats

# Modo interactivo
.venv/bin/python -m src.cli --interactive
```

### 6. Suite de Pruebas Automatizadas
```bash
.venv/bin/pytest -v tests/test_rag.py
```
*(13 pruebas unitarias e integración aisladas que validan esquema, FTS5 BM25, términos técnicos cortos, embeddings, 2-phase vector search, RRF, modos puros alpha 0/1, chunking con preservación de tablas, puente SQL-RAG, LLM mockeado y endpoints/seguridad en FastAPI).*

---

## 📡 Endpoints de la API REST

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/health` | Healthcheck liviano para monitoreo / readiness |
| `GET` | `/api/status` | Estado del sistema, recuentos de documentos, chunks, modo de síntesis y latencias |
| `GET` | `/api/llm/health` | Verificación en vivo de conectividad y modelos disponibles en el proveedor LLM |
| `POST` | `/api/query` | Ejecución de consulta RAG híbrida con citas, confianza dinámica y telemetría de fallback |
| `POST` | `/api/reindex` | Reindexación completa de documentos Markdown y tablas SQL con limpieza de huérfanos |
| `GET` | `/api/documents` | Listado de documentos y desglose por departamento |
| `GET` | `/api/documents/{id}` | Contenido completo de un documento y sus chunks |
| `POST` | `/api/documents` | Ingesta dinámica segura de nuevos partes o reportes en tiempo real |
| `DELETE` | `/api/documents/{id}` | Eliminación de documento del índice y disco (protege documentos virtuales SQL) |
| `GET` | `/api/players` | Fichas biomédicas y técnicas del plantel profesional |
| `GET` | `/api/staff` | Directorio de cuerpo técnico y especialistas |
| `GET` | `/api/departments` | Catálogo de departamentos operativos |
| `GET` | `/api/logs` | Auditoría de consultas ejecutadas con telemetría de milisegundos (`limit` 1..200) |
