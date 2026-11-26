from backend.api.routes.detection import router as detection_router
from backend.api.routes.stream import router as stream_router
from backend.api.routes.system import router as system_router
from backend.api.routes.nlp import router as nlp_router
from backend.api.routes.interactive import router as interactive_router

__all__ = [
    "detection_router",
    "stream_router",
    "system_router",
    "nlp_router",
    "interactive_router",
]

