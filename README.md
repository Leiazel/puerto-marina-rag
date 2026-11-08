# ⚽ Club Atlético Puerto Marina - Sistema RAG Híbrido (v2.0 Optimizada)

Sistema integral de **Generación Aumentada por Recuperación (RAG)** de grado productivo para la gestión operativa, biomédica, táctica y logística de un club de fútbol profesional.

Arquitectura optimizada basada en **SQLite Híbrido** (**FTS5** con ranking **BM25** y filtrado de *stopwords* en español + almacenamiento binario en **BLOB** de vectores densos multilingües con **FastEmbed ONNX** + búsqueda vectorial en dos fases + fusión de rankings mediante **Reciprocal Rank Fusion (RRF)**).

---

## 🛠️ Novedades y Optimizaciones de la Versión 2.0

1. **Filtrado de *Stopwords* y Normalización de Diacríticos en Español para FTS5 (`src/retriever.py`):**
   Elimina conectores vacíos (*"como"*, *"cual"*, *"para"*, *"los"*, *"del"*, etc.) y normaliza tildes antes de armar la consulta `MATCH` en SQLite FTS5. Esto previene falsos positivos y maximiza la precisión de BM25.
2. **Búsqueda Vectorial en Dos Fases con Reducción Drástica de I/O (`src/retriever.py`):**
   - **Fase 1:** Carga únicamente los pares `(id, embedding)` (blobs binarios compactos de 1.536 bytes) y ejecuta el producto punto matricial en BLAS/C.
   - **Fase 2:** Recupera el texto completo y metadatos **únicamente de los fragmentos Top-K ganadores**.
3. **Puente Relacional SQL-a-RAG (`src/ingestion.py`):**
   Las tablas estructuradas `players` y `staff` se sincronizan automáticamente como documentos virtuales vectorizados. Permite consultar directamente datos de contacto, teléfonos, pesos, estaturas y dorsales.
4. **Sintetizador Extractivo con Preservación de Tablas y Confianza Dinámica (`src/generator.py`):**
   Reconstruye encabezados de tablas Markdown y calcula un índice de confianza dinámico según el acuerdo inter-modal (BM25 + Coseno).
5. **Adaptador Universal para LLMs (`src/config.py` y `src/generator.py`):**
   Funciona 100% de forma local/offline o se conecta a cualquier LLM compatible con OpenAI (Ollama, Groq, Gemini, OpenAI) configurando variables de entorno (`USE_EXTERNAL_LLM`, `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL`).
6. **Suite Ampliada a 10 Pruebas Unitarias Automatizadas (`tests/test_rag.py`):**
   Cobertura completa del motor híbrido, normalización, puente relacional y auditoría.

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

### 1. Dependencias
```bash
pip install -r requirements.txt
```

### 2. Configuración Opcional (`.env`)
Copiar plantilla de entorno:
```bash
cp .env.example .env
```
Por defecto, `USE_EXTERNAL_LLM=false` utiliza el sintetizador local extractivo de alto rendimiento (0 costo, 0 dependencias). Si deseas activar un LLM (Ollama, Groq, Gemini u OpenAI), edita `.env`.

### 3. Iniciar el Servidor Web & API
```bash
./run_server.sh
```
- **Consola Web:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Documentación Swagger:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4. Consultas por Terminal (CLI)
```bash
# Consulta puntual
PYTHONPATH=. python src/cli.py --query "¿Cuál es la lesión de Lautaro Carrizo y cuándo vuelve a jugar?"

# Consulta con filtro por departamento
PYTHONPATH=. python src/cli.py --dept preparacion_fisica --query "¿Qué jugador tiene alerta de fatiga en el GPS?"

# Consulta de datos relacionales indexados
PYTHONPATH=. python src/cli.py --query "¿Cuál es el correo y teléfono de la nutricionista Sofía Carrizo?"

# Estadísticas del sistema
PYTHONPATH=. python src/cli.py --stats

# Modo interactivo
PYTHONPATH=. python src/cli.py --interactive
```

### 5. Suite de Pruebas Automatizadas
```bash
PYTHONPATH=. python tests/test_rag.py
```
*(10 pruebas unitarias que validan esquema, FTS5 BM25, embeddings, 2-phase vector search, RRF, filtros departamentales, puente SQL-RAG y telemetría).*

---

## 📡 Endpoints de la API REST

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/status` | Estado del sistema, recuentos de documentos, chunks, estado del LLM y latencias |
| `POST` | `/api/query` | Ejecución de consulta RAG híbrida con citas, fragmentos y confianza dinámica |
| `GET` | `/api/documents` | Listado de documentos y desglose por departamento |
| `GET` | `/api/documents/{id}` | Contenido completo de un documento y sus chunks |
| `POST` | `/api/documents` | Ingesta dinámica de nuevos partes o reportes en tiempo real |
| `GET` | `/api/players` | Fichas biomédicas y técnicas del plantel profesional |
| `GET` | `/api/staff` | Directorio de cuerpo técnico y especialistas |
| `GET` | `/api/departments` | Catálogo de departamentos operativos |
| `GET` | `/api/logs` | Auditoría de consultas ejecutadas con telemetría de milisegundos |
