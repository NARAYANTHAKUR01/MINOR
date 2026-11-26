# YOLO11m Local Object Detection & Backend API (macOS)

A modular Python backend running the **Ultralytics YOLO11m** object detection model locally on macOS with Apple Silicon GPU acceleration (**Metal Performance Shaders / MPS**).

- **100% Local & Offline**: No cloud APIs, no API keys, and zero telemetry after downloading the model weights once.
- **Hardware Acceleration**: Auto-detects Apple Silicon Metal GPU (`mps`) with standard CPU fallback.
- **Modular Backend**: Clean separation into `core/`, `schemas/`, `services/`, and `api/` layers for easy extensibility and frontend integration.
- **FastAPI Web Server**: Pre-configured with CORS, Swagger docs, REST endpoints for images/videos, and an MJPEG real-time webcam stream.
- **Frontend-Ready**: Standardized Pydantic JSON responses + built-in web dashboard at `http://localhost:8000/`.
- **Dual-Interface**: Use the interactive Web API or the standalone CLI script (`main.py`).

---

## Project Structure

```text
yolo_local/
├── backend/
│   ├── config.py                 # Central configurations & paths
│   ├── core/
│   │   ├── device.py             # Apple Silicon MPS / CPU auto-detection
│   │   └── model_loader.py       # Weights validation, one-time download, model caching
│   ├── schemas/
│   │   └── detection.py          # Pydantic models (BoundingBox, DetectionItem, API responses)
│   ├── services/
│   │   ├── detector.py           # Core YOLO11m inference wrapper
│   │   ├── visualizer.py         # Box drawing, label contrast badges, FPS counter
│   │   ├── image_service.py      # Image processing (files & HTTP byte uploads)
│   │   ├── video_service.py      # Frame-by-frame video processing & export
│   │   └── webcam_service.py     # Live camera preview & MJPEG HTTP stream generator
│   └── api/
│       ├── app.py                # FastAPI application with CORS & static mounting
│       └── routes/
│           ├── detection.py      # REST endpoints (image/video uploads)
│           ├── stream.py         # Live MJPEG webcam stream endpoint
│           └── system.py         # Health, device status, and model metadata
├── static/
│   └── index.html                # Built-in test web UI dashboard
├── models/
│   └── yolo11m.pt                # Local weights file (downloaded once)
├── outputs/                      # Output directory for annotated images & videos
├── bus.jpg                       # Sample test image
├── test_video.mp4                # Sample test video
├── server.py                     # API server launcher
├── main.py                       # CLI runner (image, video, webcam)
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

---

## 1. Setup & Environment

Ensure you are in the `yolo_local` directory and activate the virtual environment:

```bash
cd yolo_local
source .venv/bin/activate
```

Dependencies installed in `requirements.txt`:
```text
ultralytics
opencv-python
torch
torchvision
numpy
fastapi
uvicorn[standard]
python-multipart
pydantic
```

---

## 2. Running the Backend API Server

Start the FastAPI server:
```bash
python server.py
```
*(Options: `--port 8000`, `--host 0.0.0.0`, `--reload`)*

Once started, open your browser to:
- **Web UI Dashboard**: [`http://localhost:8000/`](http://localhost:8000/)
- **Interactive Swagger Docs**: [`http://localhost:8000/docs`](http://localhost:8000/docs)
- **API Base URL**: `http://localhost:8000/api`

---

## 3. Frontend Integration Guide

The backend includes `CORSMiddleware` with `allow_origins=["*"]`, allowing any frontend (React, Vite, Next.js, Vue, Svelte, or mobile apps) to connect directly.

### A. Detect Objects in an Image (JavaScript / React)
```javascript
const formData = new FormData();
formData.append('file', imageFileInput.files[0]);

const response = await fetch('http://localhost:8000/api/detection/image?conf=0.4', {
  method: 'POST',
  body: formData
});

const data = await response.json();
console.log('Inference Time (ms):', data.inference_time_ms);
console.log('Detected Count:', data.num_detections);
console.log('Detections:', data.detections);
// [
//   {
//     "class_id": 0,
//     "class_name": "person",
//     "confidence": 0.92,
//     "bbox": { "x1": 48.48, "y1": 399.53, "x2": 249.97, "y2": 905.41, "width": 201.49, "height": 505.88 }
//   }
// ]
console.log('Annotated Image URL:', `http://localhost:8000${data.annotated_image_url}`);
```

### B. Live Real-Time Webcam Stream (HTML / React)
The backend streams MJPEG video with live bounding boxes and FPS directly from the camera. You can display it in your frontend using a standard `<img>` tag:

```html
<!-- HTML -->
<img src="http://localhost:8000/api/stream/webcam?conf=0.4" alt="YOLO11m Live Stream" />
```

```jsx
// React Component
function LiveCameraView({ conf = 0.4 }) {
  return (
    <img
      src={`http://localhost:8000/api/stream/webcam?conf=${conf}`}
      alt="Real-time Detection"
      className="rounded-xl shadow-lg"
    />
  );
}
```

### C. Direct Image Response (Returns Annotated JPEG File)
```html
<img src="http://localhost:8000/api/detection/image/direct?conf=0.4" />
```

### D. Device & Hardware Status
```javascript
const res = await fetch('http://localhost:8000/api/system/device');
const status = await res.json();
console.log(status);
// {
//   "device": "mps",
//   "device_name": "Apple Silicon GPU (Metal Performance Shaders)",
//   "mps_available": true,
//   "mps_built": true,
//   "torch_version": "2.14.1"
// }
```

---

## 4. CLI Execution (`main.py`)

You can also run all modes from the command line:

### Mode 1: Still Image
```bash
python main.py --source image --path bus.jpg --conf 0.4
```
*(Saves to `outputs/annotated_bus.jpg` and opens a preview window.)*

### Mode 2: Video File
```bash
python main.py --source video --path test_video.mp4 --conf 0.4
```
*(Processes frames, displays live preview, and writes `outputs/annotated_test_video.mp4`.)*

### Mode 3: Real-Time Webcam
```bash
python main.py --source webcam
```
*(Streams your Mac's camera with live FPS. Press **`q`** to quit.)*

---

## 5. macOS Camera Permissions

If webcam mode displays an access error:
1. Open **System Settings** -> **Privacy & Security** -> **Camera**.
2. Toggle **ON** for your terminal or IDE (**Terminal**, **iTerm2**, **VS Code**, **Cursor**).
3. Ensure FaceTime or Photo Booth is closed.

---

## 6. Local Open-Source Speech-to-Text & NLP Object Extraction Engine

The backend integrates an open-source **OpenAI Whisper** Speech-to-Text model (`base`) running **100% locally and offline** on macOS, paired with a dedicated NLP extraction pipeline to map spoken speech to the **80 COCO object classes**:

- **100% Local & Offline Whisper STT**: Pre-downloaded weights in `models/whisper/base.pt`. No external cloud APIs, no billing, no network latency.
- **Microphone & Audio File Support**: Transcribes browser mic recordings (`MediaRecorder` WebM), `.wav`, `.mp3`, `.ogg`, `.m4a`, and `.flac`.
- **All 80 COCO Classes Supported**: Exact ID mapping (0 to 79) from `person` to `toothbrush`.
- **Longest-Phrase Precedence**: Multi-word objects (`hot dog`, `teddy bear`, `dining table`, `cell phone`, `traffic light`, `sports ball`, `wine glass`, etc.) are matched before individual sub-words.
- **Spoken Synonyms & Aliases**: Colloquial speech mapping (`phone` -> `cell phone`, `bike` -> `bicycle`, `sofa` -> `couch`, `fridge` -> `refrigerator`, `macbook` -> `laptop`, `puppy` -> `dog`, `mug` -> `cup`, etc.).
- **Lemmatization**: Automatically reduces regular and irregular plurals (`knives` -> `knife`, `mice` -> `mouse`, `people` -> `person`, `bicycles` -> `bicycle`, `sandwiches` -> `sandwich`).
- **Negation Filtering**: Identifies excluded items (`"find the person, not the car"` -> targets: `[person]`, excluded: `[car]`).
- **Intent & Modifier Extraction**: Detects conversational intent (`locate`, `count`, `detect`, `filter`) and captures quantities/colors (`"two red apples"`).
- **Fuzzy Matching**: High tolerance for phonetic or spelling noise from speech audio (`aeroplan` -> `airplane`, `cellfone` -> `cell phone`, `telivision` -> `tv`).

### Standalone CLI Runner (`nlp_cli.py`)
Transcribe speech audio or test spoken sentences directly from your terminal:

```bash
# 1. Transcribe any audio file locally with Whisper and extract COCO objects
python nlp_cli.py --audio sample.wav

# 2. Record 4 seconds from your MacBook microphone and process
python nlp_cli.py --record 4

# 3. Test a text sentence directly
python nlp_cli.py --sentence "Where did I put my cell phone and two red apples? Don't search for car."

# 4. Interactive REPL mode (supports ':record', ':audio <file>', ':status', and typed queries)
python nlp_cli.py --interactive
```

### REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/nlp/stt-status` | Get active STT provider, local model weights path, device & status |
| `POST` | `/api/nlp/transcribe` | Transcribe speech audio file (.wav, .mp3, .webm) into text |
| `POST` | `/api/nlp/transcribe-and-extract` | Transcribe audio file and extract 80 COCO objects via NLP |
| `POST` | `/api/nlp/extract` | Extract COCO objects, IDs, intent, attributes from sentence JSON |
| `POST` | `/api/stream/query-audio` | Transcribe mic audio and match targets in real-time camera stream |
| `POST` | `/api/nlp/query-image-audio` | Visual search: spoken audio query + uploaded image |
| `GET` | `/api/nlp/classes` | Returns all 80 COCO classes, IDs, aliases, and sample queries |
| `GET` | `/api/nlp/stt-providers` | View available and active Speech-to-Text providers |
| `POST` | `/api/nlp/set-stt-provider` | Switch active STT provider (`whisper`, `mock`) |

#### Example 1: Local Audio Transcription & Object Extraction
```bash
curl -X POST http://localhost:8000/api/nlp/transcribe-and-extract \
  -F "audio_file=@/path/to/speech.wav"
```
Response:
```json
{
  "success": true,
  "transcribed_sentence": "Where is my cell phone and water bottle?",
  "stt_provider": "whisper_base",
  "nlp_result": {
    "intent": "locate",
    "target_classes": ["cell phone", "bottle"],
    "target_class_ids": [67, 39],
    "extracted_objects": [
      {
        "class_id": 67,
        "class_name": "cell phone",
        "matched_phrase": "cell phone",
        "match_type": "exact",
        "confidence": 1.0
      },
      {
        "class_id": 39,
        "class_name": "bottle",
        "matched_phrase": "water bottle",
        "match_type": "alias",
        "confidence": 0.95
      }
    ],
    "processing_time_ms": 14.2
  }
}
```

#### Example 2: Check Local STT Status
```bash
curl http://localhost:8000/api/nlp/stt-status
```
Response:
```json
{
  "active_provider": "whisper",
  "available_providers": { "whisper": true, "mock": true },
  "model_name": "base",
  "model_path": "/Users/.../yolo_local/models/whisper",
  "device": "cpu",
  "is_local": true
}
```

### Running Automated Tests
```bash
python -m unittest discover tests -v
# Or individual test modules:
python -m unittest tests/test_interactive_mode.py -v
python -m unittest tests/test_target_localization.py -v
python -m unittest tests/test_coco_accuracy.py -v
python -m unittest tests/test_stt_whisper.py -v
python -m unittest tests/test_nlp_extractor.py -v
```

---

## 7. Exclusive Target-Only Localization Mode

When asking to find an object (e.g. *"Where is my cell phone?"*, *"Locate water bottle"*), the pipeline can lock exclusively on the target object extracted by NLP, filtering out all background clutter (chairs, tables, background people):

- **Target Lock Reticle**: Draws neon emerald `[ ]` reticle corner brackets, center `+` crosshairs, bearing angle, and distance tags on ONLY the target object.
- **Top HUD Status Banner**: Displays `TARGET LOCKED: [OBJECT] | AZIMUTH: ... | DISTANCE: ...` when in frame, or `SCANNING FOR: [OBJECT] (NOT IN VIEW)` when searching.
- **Background Clutter Suppression**: Suppresses non-target YOLO detections at the model inference/NMS stage, speeding up inference.
- **Web UI Controls**: Floating `TARGET LOCK: [CLASS] · ONLY TARGET LOCATED` badge with one-click `[✕ Clear]` button, and quick-lock buttons on the calibration sidebar.

### CLI Usage (`main.py`)
```bash
# Locate ONLY the phone in live camera:
python main.py --source webcam --target "where is my phone"

# Locate ONLY the bus in an image:
python main.py --source image --path bus.jpg --target "bus"

# Locate ONLY the person in a video:
python main.py --source video --path test_video.mp4 --target "locate person"
```

### Stream Target Lock REST Endpoints
| Method | Endpoint | Payload | Description |
|---|---|---|---|
| `POST` | `/api/stream/set-target` | `{"target": "cell phone"}` or `{"query": "where is my bottle"}` | Activates target lock to locate ONLY requested object(s) |
| `POST` | `/api/stream/clear-target` | *(empty)* | Disables target lock, returning to full-scene detection |
| `GET` | `/api/stream/target` | *(none)* | Returns current target lock status and active target classes |

---

## 8. Interactive Voice-Based Target Object Detection Mode

OmniSight includes a fully hands-free, conversational interactive voice assistant:

1. **State Machine (10 States)**:
   - `DISABLED`: Zero mic and GPU overhead when toggled OFF.
   - `WAITING_FOR_WAKE_WORD`: Continuously listens for the wake phrase **"Hi Omii"** (or sound-alikes: *"Hi Omi"*, *"Hey Omii"*, *"Hey Omi"*, *"Hi Omee"*).
   - `ACKNOWLEDGING_WAKE_WORD`: Speaks acknowledgment (*"Yes, I'm listening. Which object should I find?"*) and mutes the microphone to prevent self-triggering.
   - `LISTENING_FOR_COMMAND`: Records user's voice question with Voice Activity Detection (VAD) silence endpointing (stops after 1.2s silence or max 10s).
   - `TRANSCRIBING`: Transcribes speech locally via the open-source Whisper model.
   - `EXTRACTING_TARGET`: Identifies primary target object, COCO ID, and requested spatial direction (`left`, `right`, `top`, `bottom`, `center`, `top-left`, etc.).
   - `SEARCHING_FOR_OBJECT`: Runs YOLO on camera frames for up to **5.0 seconds** (monotonic clock):
     - If the target is found in the requested region, exits immediately!
     - If the target is found in the wrong region, keeps searching the remainder of 5s before reporting the mismatch.
     - Times out at 5.0s if absent.
   - `GENERATING_RESPONSE`: Formulates natural English spatial feedback (*"I found your bottle on the left side of the frame."* or *"I found your bottle on the right side of the frame, but I couldn't find one on the left."*).
   - `SPEAKING_RESPONSE`: Delivers spoken feedback via TTS and returns seamlessly to wake-word listening.
   - `ERROR_RECOVERY`: Handles empty speech, background noise, or unknown targets gracefully.

### Interactive Mode REST Endpoints
| Method | Endpoint | Payload | Description |
|---|---|---|---|
| `POST` | `/api/interactive/toggle` | `{"enable": true/false}` | Enables/disables interactive voice assistant |
| `GET` | `/api/interactive/status` | *(none)* | Returns live state, active target, and timestamp |
| `POST` | `/api/interactive/set-state` | `{"state": "..."}` | Updates state machine state |
| `POST` | `/api/interactive/process-audio` | `multipart/form-data` (audio file) | Transcribes speech, runs 5s search, and returns TTS response |
| `POST` | `/api/interactive/text-command` | `{"text": "..."}` | Text runner for direct or simulated queries |

### Running Interactive Mode in Terminal
```bash
python main.py --source interactive
```


