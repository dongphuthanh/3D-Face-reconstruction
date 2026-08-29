// Upload a photo to the face3d API and show the returned 3D head.
//
// The whole lesson is in sendToApi(). Everything else is plumbing.
// Bare specifiers, resolved by the <script type="importmap"> in index.html.
// The addons import from 'three' internally, so loading them by full URL is
// not enough -- the browser still cannot resolve what is INSIDE them.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const API = "http://127.0.0.1:8000/reconstruct";

const fileInput = document.getElementById("file");
const goButton = document.getElementById("go");
const downloadButton = document.getElementById("download")
const statusEl = document.getElementById("status");


// Blob URLs hold memory until you revoke them. Keep the current one so we can
// release the previous head before showing a new one.
let currentUrl = null;


function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.className = isError ? "err" : "";
}

// ---------------------------------------------------------------------------
// THE FORMDATA PART
// ---------------------------------------------------------------------------
async function sendToApi(file) {
  // A FormData is a builder for a multipart/form-data request body -- the same
  // format a plain <form enctype="multipart/form-data"> produces. It exists
  // because JSON cannot carry raw binary: you would have to base64 it and grow
  // the payload by a third. Multipart sends the bytes as they are.
  const formData = new FormData();

  // append(fieldName, value). THE FIELD NAME IS A CONTRACT.
  //
  // "image" here must match the parameter name in server.py:
  //     def reconstruct(image: UploadFile = File(...))
  //                     ^^^^^
  // Call it "file" or "photo" and FastAPI answers 422 with a validation error
  // about a missing field -- which reads like your PHOTO was rejected, when
  // really your FORM FIELD was. Easy 20 minutes lost.
  formData.append("image", file);

  // A File (from <input type="file">) is already a Blob, so it goes in as-is.
  // You do NOT read it, decode it, or convert it to base64. You also cannot
  // send a filesystem path: the browser deliberately never gives you one.
  //
  // Peek at what you built -- FormData is opaque to console.log, so iterate:
  for (const [key, value] of formData.entries()) {
    console.log("field:", key, "->", value.name, value.type, value.size, "bytes");
  }

  const response = await fetch(API, {
    method: "POST",     // omit this and you send GET -> 405 Method Not Allowed
    body: formData,

    // NOTE WHAT IS *NOT* HERE: no headers.
    //
    // The instinct is to add {"Content-Type": "multipart/form-data"}. Doing
    // that BREAKS the request. Multipart needs a randomly generated boundary
    // string that separates the fields, and it lives inside the Content-Type
    // header, like:
    //     multipart/form-data; boundary=----WebKitFormBoundaryAbc123
    // The browser generates that boundary and writes the header for you. Set
    // the header yourself and you overwrite it WITHOUT a boundary, so the
    // server cannot split the body and rejects it. Say nothing, get it right.
  });

  // Success and failure return DIFFERENT BODY TYPES, so branch before reading.
  //   200      -> binary GLB
  //   4xx/5xx  -> JSON, {"detail": "..."}
  // Call .arrayBuffer() on an error response and you get the JSON's bytes,
  // which then fail to parse as a GLB somewhere far from the real cause.
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const err = await response.json();
      if (err.detail) detail = err.detail;   // e.g. "no face detected in the image"
    } catch {
      // Some failures (a proxy error, a crash before FastAPI) are not JSON.
    }
    throw new Error(detail);
  }

  // Binary, not .json() and not .text(). .text() would corrupt it: the bytes
  // are not valid UTF-8 and invalid sequences get replaced.
  return await response.arrayBuffer();
}

// ---------------------------------------------------------------------------
// SHOWING THE RESULT
//
// This is the only function that changes when you move to three.js.
// <model-viewer> wants a URL, so we wrap the bytes in a Blob and mint one.
// GLTFLoader instead takes the ArrayBuffer directly:
//     new GLTFLoader().parse(buffer, "", (gltf) => scene.add(gltf.scene));
// -- no Blob, no URL, and at that point you should also delete the
// Content-Disposition header from server.py so the browser stops treating the
// response as a download.
// ---------------------------------------------------------------------------
let scene;
let camera;
let renderer;
let controls;
let currentModel = null;

function initThree() {
    const container = document.getElementById("viewer");

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(
        45,
        container.clientWidth / container.clientHeight,
        0.1,
        100
    );

    // Placeholder. frameObject() overwrites this once a head is loaded --
    // the head is only 0.31 m tall, so a camera 3 m away shows a speck.
    camera.position.set(0, 0, 1);

    // alpha: true keeps the canvas transparent so the CSS gradient on #viewer
    // shows through. A gradient backdrop costs nothing on the GPU and beats a
    // flat colour: pure black gives a face no silhouette, makes skin tones look
    // harsh, and draws the eye straight to the hard edge of the skin mask.
    renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true
    });

    renderer.setSize(
        container.clientWidth,
        container.clientHeight
    );

    // Match the display's pixel density or the render looks soft on a laptop
    // screen. Capped at 2: beyond that you are paying 3-4x the fragment cost
    // for a difference nobody can see.
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Tone mapping compresses bright values into displayable range, the way a
    // camera does. Without it, lit skin clips to flat white patches.
    renderer.toneMapping = THREE.ACESFilmicToneMapping;

    // Master brightness dial, and the first knob to reach for.
    //
    // There is a principled target here rather than a taste call. The albedo
    // was fit under the model's own estimated light, whose ambient term works
    // out to roughly a 0.67 multiplier -- so the pipeline's own view is that
    // this face should render around 0.44 luminance, and the SOURCE PHOTOGRAPH
    // measures 0.597. At exposure 0.95 three.js was rendering it at 0.87,
    // because an environment map plus a key light is far more illumination
    // than the albedo was fit under.
    //
    // 0.65 brings the rendered face back to roughly the photograph's own
    // brightness. Raise it if you prefer a brighter look; the number to beat
    // is the photo, not personal preference.
    renderer.toneMappingExposure = 0.65;

    // Soft shadows, so the key light below can ground the head instead of
    // leaving it floating. PCFSoft is the good-looking option; the cost is
    // negligible for one 10k-triangle mesh.
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    container.appendChild(renderer.domElement);

    // THE BIGGEST VISUAL LEVER, and the reason a hand-rolled viewer usually
    // looks worse than <model-viewer> at first.
    //
    // MeshStandardMaterial is physically based: it REFLECTS its surroundings.
    // With only lights and no environment it has nothing to reflect, so skin
    // renders flat and dead. RoomEnvironment is a small procedural room;
    // PMREMGenerator pre-filters it into the roughness-blurred cube map the
    // material actually samples. Three lines, and it does more for the look
    // than every CSS rule in style.css combined.
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

    controls = new OrbitControls(
        camera,
        renderer.domElement
    );

    // Damping gives the drag momentum instead of stopping dead. It only works
    // if controls.update() runs every frame, which animate() does.
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Panning slides the subject off-screen with no way back except reload.
    // For a single-subject viewer it is pure downside.
    controls.enablePan = false;

    // Zoom limits are set in frameObject(), where the model's real size is
    // known. Without them you can zoom straight through the face and end up
    // inside the skull looking at backfaces.

    // A slow spin reads as a showcase rather than a still image, and it shows
    // the geometry from angles the user might not think to drag to.
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.8;

    // Stop the spin as soon as the user takes control -- fighting an
    // auto-rotating model is infuriating. It does not resume, deliberately.
    controls.addEventListener("start", () => { controls.autoRotate = false; });

    // --- lighting -------------------------------------------------------
    // The environment lights evenly from all directions, which is flattering
    // but flat: no single direction means no strong shading, so cheekbones,
    // nose and brow lose definition. A key light restores that.
    // 0.9, not 1.6. The environment is the PRIMARY light here and already
    // exposes the model correctly; the key exists to add DIRECTION, not
    // brightness. Environment and lights are additive, and treating them as
    // independent is how a scene ends up washed out.
    const key = new THREE.DirectionalLight(0xffffff, 0.7);
    key.position.set(0.6, 0.9, 1.2);          // front, above, slightly right
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.bias = -0.0005;                // kills shadow acne on curves
    scene.add(key);

    // Fill from the opposite side at much lower intensity, so the shadowed
    // half does not go black. Standard three-point-lighting logic: the key
    // shapes, the fill rescues detail from shadow.
    const fill = new THREE.DirectionalLight(0xbcd4ff, 0.25);
    fill.position.set(-1.0, 0.2, 0.6);
    scene.add(fill);

    // The environment ALREADY provides ambient from every direction, so a
    // hemisphere light on top is mostly redundant. Kept very low purely to lift
    // the shadow side a little; set it to 0 and you will barely notice.
    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 0.15));

    // Keep the canvas matched to its container when the window resizes.
    // Miss this and the head stretches or squashes, because the drawing buffer
    // no longer matches the element it is displayed in.
    window.addEventListener("resize", () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();   // aspect is only read from here
        renderer.setSize(container.clientWidth, container.clientHeight);
    });

    animate();
}


function frameObject(object) {
    // Point the camera at whatever was loaded, whatever size it is.
    //
    // Hard-coding a distance means re-tuning it every time the asset changes.
    // Box3 measures the actual bounds, so this works for a 0.31 m head or a
    // 3 m statue with no edit.
    const box = new THREE.Box3().setFromObject(object);
    const size = box.getSize(new THREE.Vector3());
    const centre = box.getCenter(new THREE.Vector3());

    // Distance at which the object's largest dimension fills the vertical
    // field of view. fov is in DEGREES; Math.tan wants RADIANS -- forgetting
    // that conversion is the classic way to end up 50x too far away.
    const maxDim = Math.max(size.x, size.y, size.z);
    const fovRad = THREE.MathUtils.degToRad(camera.fov);
    const distance = (maxDim / 2) / Math.tan(fovRad / 2);

    camera.position.set(centre.x, centre.y, centre.z + distance * 1.6);

    // Near/far define the depth range. The default near of 0.1 would clip
    // INTO a 0.31 m head as you zoom; scale both to the object instead.
    camera.near = distance / 100;
    camera.far = distance * 100;
    camera.updateProjectionMatrix();

    // OrbitControls rotates around its target, not the origin. The head sits
    // slightly off-centre, so without this it swings around a point beside it.
    controls.target.copy(centre);

    // Zoom limits, now that the real size is known. Without a minimum you can
    // zoom straight through the face and end up inside the skull looking at
    // backfaces; without a maximum the head shrinks to a dot.
    controls.minDistance = distance * 0.6;
    controls.maxDistance = distance * 4.0;

    // Keep the camera above the floor plane. Orbiting underneath shows the
    // open underside of the neck, which is the least flattering angle a FLAME
    // head has.
    controls.maxPolarAngle = Math.PI * 0.85;

    controls.update();

    // Let every mesh in the model cast and receive shadows. GLTFLoader leaves
    // these false by default, so the key light's shadow map would otherwise be
    // set up and never used.
    object.traverse((child) => {
        if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
        }
    });
}

function animate() {
    requestAnimationFrame(animate);

    controls.update();
    renderer.render(scene, camera);
}

// ---------------------------------------------------------------------------
// THE EXPRESSION RIG
//
// Every GLB carries 21 morph targets -- 20 FLAME expression components plus a
// baked jaw rotation. three.js exposes them on the mesh as:
//
//   mesh.morphTargetDictionary   { "jaw_open": 20, "expr_00": 0, ... }
//   mesh.morphTargetInfluences   [0, 0, 0, ...]   one float per target
//
// Writing into that array deforms the mesh on the next frame. There is no
// update() call and no dirty flag -- assign and it happens.
//
// The dictionary comes from `extras.targetNames` in the glTF, which
// export_head.py writes. Without it three.js would still animate the targets
// but you would only be able to address them by number.
// ---------------------------------------------------------------------------
// PLURAL. Splitting the eyes into their own glTF primitive means GLTFLoader
// builds TWO three.js meshes -- skin and eyes -- and each carries its own copy
// of the 21 morph targets. An earlier version kept a single mesh from
// traverse(), which silently held whichever came last (the eyeballs), so the
// sliders moved the eyes and nothing else. Every mesh has to be driven.
let morphMeshes = [];

// influence 1.0 == expression coefficient 1.0 == one standard deviation of that
// component. FLAME expressions run to roughly +/-3 sigma, so +/-2 is expressive
// without tearing the face apart. Negative is meaningful for a PCA basis: it is
// the opposite deformation, not an error.
const EXPR_RANGE = 2.0;

function prettyName(name) {
    // expr_07 is a PCA component, not a nameable expression like "smile" --
    // FLAME's basis has no semantic labels, so pretending otherwise would be a
    // lie. Number them and let the user explore.
    if (name === "jaw_open") return "Jaw open";
    const m = name.match(/^expr_(\d+)$/);
    return m ? `Expression ${parseInt(m[1], 10) + 1}` : name;
}

function buildSliders(model) {
    const panel = document.getElementById("sliders");
    if (!panel) return;
    panel.innerHTML = "";

    // Collect EVERY mesh carrying morph targets, not just one.
    morphMeshes = [];
    model.traverse((o) => {
        if (o.isMesh && o.morphTargetInfluences && o.morphTargetInfluences.length) {
            morphMeshes.push(o);
        }
    });
    if (!morphMeshes.length || !morphMeshes[0].morphTargetDictionary) {
        console.log("no morph targets on this model");
        return;
    }
    console.log(`rig: ${morphMeshes.length} mesh(es), ` +
                `${morphMeshes[0].morphTargetInfluences.length} targets each`);

    // Jaw first: it is the only target with an obvious real-world meaning, so
    // it is the one that demonstrates the rig at a glance.
    // Both primitives were exported from one mesh, so their dictionaries match.
    const names = Object.keys(morphMeshes[0].morphTargetDictionary);
    names.sort((a, b) => (a === "jaw_open" ? -1 : b === "jaw_open" ? 1 : a.localeCompare(b)));

    for (const name of names) {
        const idx = morphMeshes[0].morphTargetDictionary[name];
        const jaw = name === "jaw_open";

        const row = document.createElement("label");
        row.className = "rig-row";

        const label = document.createElement("span");
        label.className = "rig-name";
        label.textContent = prettyName(name);

        const slider = document.createElement("input");
        slider.type = "range";
        // Jaw only opens. A negative jaw rotation closes past shut, which
        // pushes the lower teeth through the upper lip.
        slider.min = jaw ? 0 : -EXPR_RANGE;
        slider.max = EXPR_RANGE;
        slider.step = 0.01;
        slider.value = 0;

        const readout = document.createElement("span");
        readout.className = "rig-value";
        readout.textContent = "0.00";

        // "input" fires continuously while dragging; "change" only fires on
        // release, which would make the slider feel dead.
        slider.addEventListener("input", () => {
            const v = parseFloat(slider.value);
            // Write into EVERY mesh, or skin and eyes drift apart -- the eyes
            // would hold an expression the face is not making.
            for (const m of morphMeshes) m.morphTargetInfluences[idx] = v;
            readout.textContent = v.toFixed(2);
        });

        row.append(label, slider, readout);
        panel.appendChild(row);
    }
}

function resetSliders() {
    if (!morphMeshes.length) return;
    for (const m of morphMeshes) m.morphTargetInfluences.fill(0);
    document.querySelectorAll("#sliders input[type=range]").forEach((s) => {
        s.value = 0;
        s.parentElement.querySelector(".rig-value").textContent = "0.00";
    });
}

document.getElementById("rig-reset")
    ?.addEventListener("click", resetSliders);


function showGlb(arrayBuffer) {
    console.log("showGlb called");
    console.log("Buffer size:", arrayBuffer.byteLength);

    const blob = new Blob(
        [arrayBuffer],
        { type: "model/gltf-binary" }
    );

    const url = URL.createObjectURL(blob);

    const loader = new GLTFLoader();

    loader.load(
        url,

        (gltf) => {
            console.log("GLB loaded!");

            if (currentModel) {
                scene.remove(currentModel);
            }

            currentModel = gltf.scene;
            scene.add(currentModel);
            frameObject(currentModel);   // aim the camera at what just arrived
            buildSliders(currentModel);  // rebuild the rig for THIS head

            URL.revokeObjectURL(url);
        },

        (progress) => {
            console.log(
                "Loading:",
                (progress.loaded / progress.total * 100) + "%"
            );
        },

        (error) => {
            console.error("GLB loading failed:", error);
        }
    );
}
function downloadGlb(arrayBuffer, fileName = "head.glb") {
  // 1. Convert the ArrayBuffer to a Blob
  const blob = new Blob([arrayBuffer], { type: "model/gltf-binary" });
  
  // 2. Create a temporary object URL pointing to the Blob
  const blobUrl = URL.createObjectURL(blob);
  
  // 3. Create a hidden anchor element
  const link = document.createElement('a');
  link.href = blobUrl;
  link.download = fileName;
  
  // 4. Append to DOM, trigger click, and cleanup
  document.body.appendChild(link);
  link.click();
  
  // 5. Remove element from DOM and revoke URL to release memory
  document.body.removeChild(link);
  URL.revokeObjectURL(blobUrl);
}

let currentGlb = null;
goButton.addEventListener("click", async () => {
  console.log("GO BUTTON CLICKED");
  const file = fileInput.files[0];
  if (!file) {
    setStatus("choose a photo first", true);
    return;
  }

  // Disable while in flight. Without this, an impatient double-click fires two
  // reconstructions and the slower response wins -- a race that shows the
  // wrong head with no error anywhere.
  goButton.disabled = true;
  setStatus(`uploading ${file.name} (${Math.round(file.size / 1024)} KB)...`);

  try {
    const started = performance.now();
    const glb = await sendToApi(file);
    currentGlb = glb;
    showGlb(glb);
    const ms = Math.round(performance.now() - started);
    setStatus(`done: ${(glb.byteLength / 1e6).toFixed(2)} MB in ${ms} ms`);
  } catch (e) {
    // A CORS failure or a server that is not running surfaces here as the
    // famously unhelpful "Failed to fetch" -- the browser withholds detail on
    // purpose. If you see it, check the Network and Console tabs: the real
    // reason is there, and it is usually that uvicorn is not running.
    setStatus(e.message, true);
    console.error(e);
  } finally {
    goButton.disabled = false;
  }
});

downloadButton.addEventListener("click", () => {
  if (!currentGlb) {
    setStatus("generate a model first", true);
    return;
  }

  downloadGlb(currentGlb, "head.glb");
});

initThree();

// Show the sample head immediately, so the page is not an empty box before the
// first upload. Same code path as a reconstruction: fetch the bytes, hand the
// ArrayBuffer to showGlb. If it 404s the viewer just stays empty -- a missing
// sample is not worth an error message to the user.
fetch("head.glb")
    .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(r.status)))
    .then(showGlb)
    .catch(() => console.log("no sample head.glb to preload"));
