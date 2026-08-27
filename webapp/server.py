"""M1: the CLI script becomes a service.

    python -m uvicorn webapp.server:app --reload --port 8000

This file is deliberately thin. All it does is translate between HTTP and a
Reconstructor: take bytes off a request, hand them to the model, put bytes back
on a response. Every hard decision lives in webapp/pipeline.py.

Interactive API docs come free at http://127.0.0.1:8000/docs -- FastAPI builds
them from the type hints below, and you can upload a photo straight from the
browser there without writing any frontend at all.
"""

import contextlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import UnidentifiedImageError

from webapp.pipeline import NoFaceFound, Reconstructor

# 10 MB. A phone photo is 2-5 MB; anything bigger is a mistake or an attack,
# and reading it into memory unchecked is how a service gets OOM-killed.
MAX_UPLOAD = 10 * 1024 * 1024
ALLOWED = {"image/jpeg", "image/png", "image/webp"}

# Where the loaded model lives for the process lifetime. A module-level dict is
# crude but honest; FastAPI also offers app.state if you prefer.
state = {}


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs ONCE at startup, and again at shutdown.

    Everything before `yield` happens before the first request is served;
    everything after happens as the process exits.

    This is the whole point of M1. Constructing the Reconstructor costs ~1.5 s
    and one reconstruction costs ~0.4 s. Build it here and every request pays
    only the 0.4. Build it inside the handler and every request pays both --
    the single most common mistake in ML web services.
    """
    print("loading model...", flush=True)
    state["rec"] = Reconstructor()
    print("ready", flush=True)
    yield
    # Shutdown. Nothing to release explicitly -- Python's GC handles it -- but
    # this is where you would close database pools or flush caches.
    state.clear()


app = FastAPI(title="face3d", lifespan=lifespan)

# The browser refuses cross-origin requests unless the server opts in. Your
# frontend will be served from somewhere else during development (a file server
# on :5173 or :8080, say) while the API is on :8000, and the browser treats a
# different PORT as a different origin. Without this you get a console error
# about Access-Control-Allow-Origin and a request that works fine in curl --
# which is the confusing part, because curl does not enforce CORS at all.
#
# allow_origins=["*"] is fine for local development. Narrow it to your actual
# frontend origin before this goes anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Liveness probe. Must stay cheap.

    Deployment platforms poll this every few seconds to decide whether to
    restart you, so it must not run the model. Reporting model_loaded lets you
    distinguish "process is up" from "process is up AND ready to serve", which
    matters during the startup window.
    """
    return {"status": "ok", "model_loaded": "rec" in state}


@app.post("/reconstruct")
def reconstruct(image: UploadFile = File(...)):
    """Photo in, GLB out.

    NOTE THE `def`. Not `async def`.

    FastAPI inspects the handler: a plain `def` is run in a worker threadpool,
    so a slow CPU-bound call blocks only its own thread. An `async def` runs ON
    the event loop, and torch never awaits anything -- so a single request would
    freeze every other connection for its whole 0.4 s. "async is faster" is
    exactly backwards for CPU-bound work.

    That choice is also why Reconstructor keeps a per-thread FaceDetector:
    threadpool means genuine concurrency, and MediaPipe is not thread-safe.
    """
    # Validate BEFORE doing expensive work. Cheap checks first is not just
    # tidiness -- it is what stops a malformed request from costing you a
    # model inference.
    if image.content_type not in ALLOWED:
        raise HTTPException(415, f"unsupported type {image.content_type}; "
                                 f"send {', '.join(sorted(ALLOWED))}")

    # Read at most MAX_UPLOAD + 1 bytes. The extra byte is the trick: if we get
    # it, the body is over the limit and we can reject without ever holding the
    # whole thing. Never trust a client-supplied Content-Length -- it is a claim
    # about the body, not a fact.
    data = image.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"image larger than {MAX_UPLOAD // 1024 // 1024} MB")
    if not data:
        raise HTTPException(400, "empty upload")

    try:
        glb = state["rec"].reconstruct(data)

    except NoFaceFound as e:
        # 422 Unprocessable Content: the request was perfectly well-formed, we
        # simply cannot do anything with the CONTENT. A blurry photo is not a
        # server fault, and returning 500 for it sends you hunting a bug that
        # does not exist.
        raise HTTPException(422, str(e))

    except UnidentifiedImageError:
        # 400 Bad Request: the bytes are not a decodable image at all. The
        # client sent something wrong, and can fix it by sending something else.
        raise HTTPException(400, "could not decode the image")

    except Exception as e:
        # Genuine server fault. Two separate audiences here, and they want
        # different things: you need the exception type and message in the log
        # to debug it; the client gets something generic, because internal
        # details in an HTTP response are an information leak.
        print(f"reconstruct failed: {type(e).__name__}: {e}", flush=True)
        raise HTTPException(500, "reconstruction failed")

    # media_type is the glTF-binary MIME type -- three.js does not care, but it
    # is correct and some tools do. Content-Disposition: attachment tells a
    # browser to download rather than try to display; drop that header if your
    # frontend fetches this and feeds it to GLTFLoader directly, which is what
    # you will want in M3.
    return Response(
        content=glb,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": 'attachment; filename="head.glb"'},
    )
