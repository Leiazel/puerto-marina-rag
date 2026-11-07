# ⚽ Club Atlético Puerto Marina - Sistema RAG de Alto Rendimiento

Sistema integral de **Generación Aumentada por Recuperación (RAG)** de punta a punta, diseñado para la gestión operativa, biomédica, táctica y logística de un club de fútbol profesional.

Desarrollado con una arquitectura optimizada para consultas basada en **SQLite Híbrido** (**FTS5** con ranking **BM25** para búsqueda léxica exacta + almacenamiento binario en **BLOB** de vectores densos multilingües con **FastEmbed ONNX** + fusión de rankings mediante **Reciprocal Rank Fusion (RRF)**).

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
     │  SQLite FTS5 + BM25         │                    │  FastEmbed ONNX (384 dims)    │
     │  (Nombres, códigos, reglas) │                    │  NumPy Matrix Cosine Dot      │
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
                               │  con Citas de Evidencia     │
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
   - Partes médicos semanales, resonancias en Centro Rossi, protocolos de 5 fases de Retorno al Juego (Return To Play - RTP), crioterapia con GameReady y diatermia Indiba.
2. **Preparación Física y Monitoreo GPS:**
   - *Responsable:* Prof. Germán Silveira.
   - Datos de dispositivos Catapult Vector S7, distancias totales, High Speed Running (> 19.8 km/h), sprints (> 25.2 km/h), PlayerLoad, ratio de carga aguda:crónica (ACWR) con detección de banderas rojas (> 1.45), y test de fatiga neuromuscular de salto CMJ en plataforma Chronojump.
3. **Nutrición y Rendimiento Metabólico:**
   - *Responsable:* Lic. Sofía Carrizo (ISAK 1).
   - Antropometría de 6 pliegues, sumatoria milimétrica, pautas de carga de glucógeno (8-10 g/kg), suplementación con creatina Creapure y cafeína anhidra, y **protocolo estricto Sin TACC para el futbolista celíaco Agustín Ríos (#14)**.
4. **Cuerpo Técnico y Análisis Táctico:**
   - *Responsables:* Javier Méndez (Director Técnico) y Pablo D'Angelo (Asistente / Analista).
   - Microciclos estructurados semanales (MD-4 a MD+1), informes tácticos y de video del próximo rival (*Club Deportivo Los Andes* y neutralización de su extremo Benegas), diseño de balón parado ofensivo ("Alfa Marina") y defensivo, informes del árbitro Pitres y alineación oficial para semifinales.
5. **Scouting e Inteligencia Deportiva:**
   - *Responsable:* Lucas Benítez.
   - Informes de seguimiento con percentiles avanzados (Wyscout / StatsBomb) para fichajes (Rodrigo "Chelo" Vargas) y análisis comparativo de goles esperados (xG/xA) entre los delanteros Santos (#11) y Cáceres (#9).
6. **Operaciones, Logística y Viajes:**
   - *Responsable:* Carlos "Charly" Pereyra.
   - Plan operativo de semifinal: vuelo chárter AndesJet AR-4912, alojamiento en Hotel Grand Bahía (pisos 4 y 5 exclusivos), distribución de habitaciones y acondicionamiento de la Hab. 402 como Gabinete Médico con crioterapia.
7. **Prensa, Comunicación y Reglamento Interno:**
   - *Responsable:* Mariana Gómez Solís.
   - Protocolos de zona mixta, temas prohibidos (arbitraje y rumores de fichajes), y código interno de disciplina con régimen de multas económicas en dólares (retrasos, sobrepeso en el pesaje matutino de los viernes y prohibición de celulares en vestuarios y comedores).

---

## ⚡ ¿Por qué SQLite es la Opción Óptima para este Sistema RAG?

1. **Cero Daemons Externos y Resiliencia Total:** Sin necesidad de procesos de fondo frágiles o servicios pesados en memoria; SQLite opera dentro del proceso Python con latencias de acceso a disco en microsegundos vía WAL (*Write-Ahead Logging*).
2. **FTS5 Integrado de Clase Mundial:** Soporta tokenización `unicode61 remove_diacritics 2`, permitiendo que términos en español con y sin tilde (*"médico"* vs *"medico"*, *"nutrición"* vs *"nutricion"*) coincidan con exactitud matemática mediante el algoritmo **BM25**.
3. **Almacenamiento Vectorial Compacto y BLAS Vectorizado:** Los embeddings de 384 dimensiones de FastEmbed se almacenan como `BLOB` (1.536 bytes por chunk). La similitud de coseno sobre cientos de fragmentos se resuelve en una sola multiplicación matricial C/BLAS de NumPy en **menos de 1 milisegundo**.
4. **Fusión de Rangos Recíprocos (RRF):**
   $$RRF(d) = \frac{\alpha}{60 + rank_{lexical}(d)} + \frac{1 - \alpha}{60 + rank_{vector}(d)}$$
   Permite encontrar tanto códigos y números exactos (ej: *#6*, *ACWR 1.48*, *Sin TACC*, *AR-4912*) como conceptos semánticos abstractos (ej: *"jugador cansado tras el partido"*).

---

## 🚀 Puesta en Marcha Rápida

### 1. Entorno Virtual y Dependencias
El proyecto utiliza un entorno virtual Python con dependencias ultra-rápidas instaladas (`fastembed`, `fastapi`, `uvicorn`, `numpy`, `scikit-learn`):
```bash
# Activar entorno
source /home/antigravity/rag_env/bin/activate
```

### 2. Iniciar el Servidor Web & API
```bash
/home/antigravity/club_rag/run_server.sh
```
El servidor quedará disponible en:
- **Consola Web:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Documentación Swagger:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Consultas Directas por Terminal (CLI)
```bash
# Consulta puntual
PYTHONPATH=/home/antigravity/club_rag /home/antigravity/rag_env/bin/python src/cli.py --query "¿Cuál es la lesión de Lautaro Carrizo y cuándo vuelve?"

# Consulta con filtro de departamento
PYTHONPATH=/home/antigravity/club_rag /home/antigravity/rag_env/bin/python src/cli.py --dept preparacion_fisica --query "¿Qué jugador tiene alerta de fatiga en el GPS?"

# Ver estadísticas de la base de conocimiento
PYTHONPATH=/home/antigravity/club_rag /home/antigravity/rag_env/bin/python src/cli.py --stats

# Modo interactivo tipo chat
PYTHONPATH=/home/antigravity/club_rag /home/antigravity/rag_env/bin/python src/cli.py --interactive
```

### 4. Ejecución del Test Suite Automatizado
```bash
PYTHONPATH=/home/antigravity/club_rag /home/antigravity/rag_env/bin/python tests/test_rag.py
```
*(9 pruebas unitarias que validan esquema, FTS5 BM25, embeddings, coseno, RRF, filtros y citas).*

---

## 📡 Endpoints de la API REST

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/status` | Estado del sistema, recuentos de documentos, chunks y latencias |
| `POST` | `/api/query` | Ejecución de consulta RAG híbrida con citas y fragmentos |
| `GET` | `/api/documents` | Listado de documentos y desglose por departamento |
| `GET` | `/api/documents/{id}` | Contenido completo de un documento y sus chunks |
| `POST` | `/api/documents` | Ingesta dinámica de nuevos partes o reportes en tiempo real |
| `GET` | `/api/players` | Fichas biomédicas y técnicas del plantel profesional |
| `GET` | `/api/staff` | Directorio de cuerpo técnico y especialistas |
| `GET` | `/api/departments` | Catálogo de departamentos operativos |
| `GET` | `/api/logs` | Auditoría de consultas ejecutadas con telemetría de milisegundos |
