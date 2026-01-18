import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const CROP_NODE_CLASS = "SnJakeInteractiveCropLoader";

const MODAL_ID_BASE = "snjake-crop-modal";
const HANDLE_SIZE = 14;

const clampRect = (rect, maxW, maxH) => {
    const res = { ...rect };
    res.width = Math.max(1, res.width);
    res.height = Math.max(1, res.height);

    if (res.x < 0) res.x = 0;
    if (res.y < 0) res.y = 0;

    if (res.x + res.width > maxW) {
        if (res.width > maxW) {
            res.width = maxW;
            res.x = 0;
        } else {
            res.x = maxW - res.width;
        }
    }
    if (res.y + res.height > maxH) {
        if (res.height > maxH) {
            res.height = maxH;
            res.y = 0;
        } else {
            res.y = maxH - res.height;
        }
    }

    return {
        x: Math.round(res.x),
        y: Math.round(res.y),
        width: Math.max(1, Math.round(res.width)),
        height: Math.max(1, Math.round(res.height)),
    };
};

const defaultRect = (w, h) => {
    if (!w || !h) return { x: 0, y: 0, width: 1, height: 1 };
    const side = Math.min(w, h);
    return {
        x: Math.round((w - side) / 2),
        y: Math.round((h - side) / 2),
        width: Math.max(1, side),
        height: Math.max(1, side),
    };
};

const parseCropJson = (value) => {
    if (!value) return null;
    try {
        const data = JSON.parse(value);
        if (data && typeof data === "object") {
            return {
                x: Number.isFinite(data.x) ? data.x : 0,
                y: Number.isFinite(data.y) ? data.y : 0,
                width: Number.isFinite(data.width) ? data.width : 0,
                height: Number.isFinite(data.height) ? data.height : 0,
                orig_width: Number.isFinite(data.orig_width) ? data.orig_width : 0,
                orig_height: Number.isFinite(data.orig_height) ? data.orig_height : 0,
            };
        }
    } catch (err) {
        console.warn("[SnJakeInteractiveCropLoader] Invalid crop_json:", err);
    }
    return null;
};

const updateCropJson = (node) => {
    const crop = node.__sjCrop;
    const widget = node.__sjCropJsonWidget;
    if (!crop || !widget) return;

    const payload = {
        x: Math.round(crop.rect.x),
        y: Math.round(crop.rect.y),
        width: Math.round(crop.rect.width),
        height: Math.round(crop.rect.height),
        orig_width: Math.round(crop.imageSize.w),
        orig_height: Math.round(crop.imageSize.h),
        image: node.__sjCropImageWidget?.value ?? "",
    };

    const serialized = JSON.stringify(payload);
    if (serialized !== widget.value) {
        widget.value = serialized;
        node.setDirtyCanvas(true, true);
    }
};

const ensureRectMatchesImage = (node, scaleFromPrevious = true) => {
    const crop = node.__sjCrop;
    if (!crop) return;

    const imgW = Math.max(1, crop.imageSize.w || 1);
    const imgH = Math.max(1, crop.imageSize.h || 1);
    let rect = { ...crop.rect };

    if (!Number.isFinite(rect.width) || rect.width <= 0 || !Number.isFinite(rect.height) || rect.height <= 0) {
        rect = defaultRect(imgW, imgH);
    }

    if (scaleFromPrevious && crop.prevImageSize) {
        const prev = crop.prevImageSize;
        if (prev.w && prev.w !== imgW) {
            const sx = imgW / prev.w;
            rect.x *= sx;
            rect.width *= sx;
        }
        if (prev.h && prev.h !== imgH) {
            const sy = imgH / prev.h;
            rect.y *= sy;
            rect.height *= sy;
        }
    }

    crop.rect = clampRect(rect, imgW, imgH);
    crop.prevImageSize = { w: imgW, h: imgH };
};

const loadImagePreview = async (node, imageName) => {
    const crop = node.__sjCrop;
    if (!crop) return false;

    crop.loadToken = (crop.loadToken || 0) + 1;
    const token = crop.loadToken;

    const revoke = () => {
        if (crop.imageUrl) {
            URL.revokeObjectURL(crop.imageUrl);
            crop.imageUrl = null;
        }
        crop.image = null;
    };

    if (!imageName) {
        revoke();
        crop.imageSize = { w: 1, h: 1 };
        ensureRectMatchesImage(node, false);
        updateCropJson(node);
        return false;
    }

    const params = new URLSearchParams();
    let filename = imageName;
    let subfolder = "";
    const idx = imageName.lastIndexOf("/");
    if (idx >= 0) {
        filename = imageName.slice(idx + 1);
        subfolder = imageName.slice(0, idx);
    }
    params.set("filename", filename);
    params.set("type", "input");
    params.set("channel", "rgb");
    params.set("preview", "webp;85");
    if (subfolder) params.set("subfolder", subfolder);

    try {
        const response = await api.fetchApi(`/view?${params.toString()}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const blob = await response.blob();
        if (token !== crop.loadToken) return false;

        const url = URL.createObjectURL(blob);
        const img = new Image();

        const loaded = await new Promise((resolve) => {
            img.onload = () => resolve(true);
            img.onerror = () => resolve(false);
            img.src = url;
        });

        if (token !== crop.loadToken) {
            URL.revokeObjectURL(url);
            return false;
        }

        if (!loaded) {
            URL.revokeObjectURL(url);
            return false;
        }

        revoke();
        crop.imageUrl = url;
        crop.image = img;
        crop.imageSize = {
            w: img.naturalWidth || img.width || 1,
            h: img.naturalHeight || img.height || 1,
        };

        ensureRectMatchesImage(node, true);
        updateCropJson(node);
        return true;
    } catch (err) {
        console.error("[SnJakeInteractiveCropLoader] Failed to load preview", err);
        if (token === crop.loadToken) {
            revoke();
            crop.imageSize = { w: 1, h: 1 };
            ensureRectMatchesImage(node, false);
            updateCropJson(node);
        }
        return false;
    }
};

const setModalStylesOnce = (() => {
    let applied = false;
    return () => {
        if (applied) return;
        applied = true;
        const style = document.createElement("style");
        style.id = `${MODAL_ID_BASE}-styles`;
        style.textContent = `
        .snjake-crop-overlay {
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.65);
            z-index: 10000;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 32px;
        }
        .snjake-crop-panel {
            background: #10141c;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 8px;
            box-shadow: 0 18px 48px rgba(0,0,0,0.45);
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            max-width: 90vw;
            max-height: 90vh;
        }
        .snjake-crop-toolbar {
            display: grid;
            grid-template-columns: auto 1fr auto;
            align-items: center;
            gap: 16px;
        }
        .snjake-crop-toolbar h2 {
            margin: 0;
            font-size: 18px;
            font-weight: 600;
            color: #f0f5ff;
        }
        .snjake-crop-zoom {
            display: flex;
            align-items: center;
            gap: 10px;
            justify-content: center;
            color: #d6def5;
            font-size: 13px;
        }
        .snjake-crop-zoom input[type="range"] {
            flex: 1;
            accent-color: #3b82f6;
        }
        .snjake-crop-toolbar button {
            padding: 6px 14px;
            font-size: 14px;
            border-radius: 4px;
            border: 1px solid rgba(255,255,255,0.15);
            background: rgba(255,255,255,0.08);
            color: #f0f6ff;
            cursor: pointer;
        }
        .snjake-crop-toolbar button.primary {
            background: #1e6ff9;
            border-color: rgba(255,255,255,0.2);
        }
        .snjake-crop-toolbar button:hover {
            filter: brightness(1.1);
        }
        .snjake-crop-canvas {
            position: relative;
            overflow: auto;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 6px;
            background: #080b12;
            max-width: 80vw;
            max-height: 70vh;
        }
        .snjake-crop-canvas img {
            display: block;
            max-width: 100%;
            max-height: 100%;
            user-select: none;
        }
        .snjake-crop-rect {
            position: absolute;
            border: 2px dashed rgba(255,0,0,0.9);
            box-shadow: inset 0 0 0 1px rgba(255,0,0,0.35);
            cursor: move;
        }
        .snjake-crop-rect::after {
            content: attr(data-size);
            position: absolute;
            top: -24px;
            left: 0;
            background: rgba(0,0,0,0.65);
            color: #fff;
            font-size: 12px;
            padding: 2px 6px;
            border-radius: 4px;
            white-space: nowrap;
        }
        .snjake-crop-handle {
            position: absolute;
            width: ${HANDLE_SIZE}px;
            height: ${HANDLE_SIZE}px;
            background: rgba(255,0,0,0.95);
            border: 1px solid rgba(255,255,255,0.9);
            border-radius: 2px;
        }
        .snjake-crop-handle[data-dir="nw"] { top: -${HANDLE_SIZE/2}px; left: -${HANDLE_SIZE/2}px; cursor: nwse-resize; }
        .snjake-crop-handle[data-dir="ne"] { top: -${HANDLE_SIZE/2}px; right: -${HANDLE_SIZE/2}px; cursor: nesw-resize; }
        .snjake-crop-handle[data-dir="se"] { bottom: -${HANDLE_SIZE/2}px; right: -${HANDLE_SIZE/2}px; cursor: nwse-resize; }
        .snjake-crop-handle[data-dir="sw"] { bottom: -${HANDLE_SIZE/2}px; left: -${HANDLE_SIZE/2}px; cursor: nesw-resize; }
        .snjake-crop-size {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
            color: #d6def5;
            font-size: 13px;
        }
        .snjake-crop-size input[type="number"] {
            width: 96px;
            padding: 6px 8px;
            border-radius: 4px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.06);
            color: #f0f6ff;
        }
        .snjake-crop-size .divider {
            opacity: 0.6;
        }
        .snjake-crop-size button {
            padding: 6px 12px;
            border-radius: 4px;
            border: 1px solid rgba(255,255,255,0.15);
            background: rgba(255,255,255,0.1);
            color: #f0f6ff;
            cursor: pointer;
        }
        .snjake-crop-size button:hover {
            filter: brightness(1.05);
        }
        `;
        document.head.appendChild(style);
    };
})();

const openCropModal = async (node) => {
    const crop = node.__sjCrop;
    if (!crop) return;

    const imageValue = node.__sjCropImageWidget?.value;
    if (!crop.image) {
        await loadImagePreview(node, imageValue);
    }

    if (!crop.image || !crop.imageUrl) {
        window?.alert?.("Could not load image preview.");
        return;
    }

    setModalStylesOnce();

    let widthInput;
    let heightInput;

    const overlay = document.createElement("div");
    overlay.className = "snjake-crop-overlay";
    overlay.dataset.modal = MODAL_ID_BASE;

    const panel = document.createElement("div");
    panel.className = "snjake-crop-panel";

    const toolbar = document.createElement("div");
    toolbar.className = "snjake-crop-toolbar";

    const title = document.createElement("h2");
    title.textContent = "Crop preview";

    const zoomWrap = document.createElement("div");
    zoomWrap.className = "snjake-crop-zoom";
    const zoomLabel = document.createElement("span");
    zoomLabel.textContent = "Zoom: 100%";
    const zoomInput = document.createElement("input");
    zoomInput.type = "range";
    zoomInput.min = "10";
    zoomInput.max = "300";
    zoomInput.value = "100";
    zoomWrap.append(zoomLabel, zoomInput);

    const actions = document.createElement("div");
    actions.style.display = "flex";
    actions.style.gap = "8px";

    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancel";

    const resetBtn = document.createElement("button");
    resetBtn.textContent = "Reset";

    const applyBtn = document.createElement("button");
    applyBtn.textContent = "Apply";
    applyBtn.classList.add("primary");

    actions.append(cancelBtn, resetBtn, applyBtn);
    toolbar.append(title, zoomWrap, actions);

    const sizeRow = document.createElement("div");
    sizeRow.className = "snjake-crop-size";
    const sizeLabel = document.createElement("span");
    sizeLabel.textContent = "Manual size";

    widthInput = document.createElement("input");
    widthInput.type = "number";
    widthInput.min = "1";
    widthInput.step = "1";
    widthInput.value = String(Math.max(1, Math.round(crop.rect?.width || crop.imageSize?.w || 1)));

    const sizeDivider = document.createElement("span");
    sizeDivider.textContent = "×";
    sizeDivider.className = "divider";

    heightInput = document.createElement("input");
    heightInput.type = "number";
    heightInput.min = "1";
    heightInput.step = "1";
    heightInput.value = String(Math.max(1, Math.round(crop.rect?.height || crop.imageSize?.h || 1)));

    const sizeApplyBtn = document.createElement("button");
    sizeApplyBtn.type = "button";
    sizeApplyBtn.textContent = "Apply";

    const sizeCenterBtn = document.createElement("button");
    sizeCenterBtn.type = "button";
    sizeCenterBtn.textContent = "Center";

    sizeRow.append(sizeLabel, widthInput, sizeDivider, heightInput, sizeApplyBtn, sizeCenterBtn);

    const canvasHolder = document.createElement("div");
    canvasHolder.className = "snjake-crop-canvas";

    const imgWrapper = document.createElement("div");
    imgWrapper.style.position = "relative";
    imgWrapper.style.display = "inline-block";

    const img = document.createElement("img");
    img.src = crop.imageUrl;
    img.alt = "crop preview";
    img.draggable = false;
    img.style.touchAction = "none";

    const rectEl = document.createElement("div");
    rectEl.className = "snjake-crop-rect";
    rectEl.style.touchAction = "none";

    const handles = ["nw", "ne", "se", "sw"].map((dir) => {
        const h = document.createElement("div");
        h.className = "snjake-crop-handle";
        h.dataset.dir = dir;
        h.style.touchAction = "none";
        return h;
    });
    handles.forEach((h) => rectEl.appendChild(h));

    imgWrapper.append(img, rectEl);
    canvasHolder.appendChild(imgWrapper);

    panel.append(toolbar, sizeRow, canvasHolder);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    const cleanup = () => {
        window.removeEventListener("keydown", onKey);
        overlay.remove();
    };

    const cropSize = { ...crop.imageSize };
    const workingRect = { ...crop.rect };
    let scale = 1;
    let baseScale = 1;
    let currentZoom = 1;

    const syncSizeInputs = () => {
        if (!widthInput || !heightInput) return;
        widthInput.value = `${Math.round(Math.max(1, workingRect.width))}`;
        heightInput.value = `${Math.round(Math.max(1, workingRect.height))}`;
    };

    const updateZoomLabel = () => {
        zoomLabel.textContent = `Zoom: ${Math.round(currentZoom * 100)}%`;
        const sliderValue = Math.round(currentZoom * 100);
        if (zoomInput.value !== String(sliderValue)) {
            zoomInput.value = String(sliderValue);
        }
    };

    const updateRectDisplay = () => {
        rectEl.style.left = `${workingRect.x * scale}px`;
        rectEl.style.top = `${workingRect.y * scale}px`;
        rectEl.style.width = `${Math.max(1, workingRect.width * scale)}px`;
        rectEl.style.height = `${Math.max(1, workingRect.height * scale)}px`;
        rectEl.dataset.size = `${Math.round(workingRect.width)} x ${Math.round(workingRect.height)}`;
        syncSizeInputs();
    };

    const recalcScale = () => {
        const w = cropSize.w || img.naturalWidth || 1;
        scale = img.clientWidth / w;
        if (!Number.isFinite(scale) || scale <= 0) {
            scale = img.width / w;
        }
        if (!Number.isFinite(scale) || scale <= 0) scale = 1;
    };

    const computeBaseScale = () => {
        const bounds = canvasHolder.getBoundingClientRect();
        const naturalW = cropSize.w || img.naturalWidth || 1;
        const naturalH = cropSize.h || img.naturalHeight || 1;
        const availW = Math.max(100, bounds.width - 32);
        const availH = Math.max(100, bounds.height - 32);
        const scaleW = availW / naturalW;
        const scaleH = availH / naturalH;
        const minScale = Math.min(scaleW, scaleH);
        baseScale = Number.isFinite(minScale) && minScale > 0 ? Math.min(1, minScale) : 1;
    };

    const clampWorkingRect = () => {
        const clamped = clampRect(workingRect, cropSize.w, cropSize.h);
        Object.assign(workingRect, clamped);
    };

    const applyManualSize = (centerOnImage = false) => {
        if (!widthInput || !heightInput) return;
        const desiredW = Math.max(1, parseInt(widthInput.value, 10) || workingRect.width);
        const desiredH = Math.max(1, parseInt(heightInput.value, 10) || workingRect.height);

        const next = {
            ...workingRect,
            width: desiredW,
            height: desiredH,
        };

        if (centerOnImage) {
            next.x = (cropSize.w - desiredW) / 2;
            next.y = (cropSize.h - desiredH) / 2;
        }

        Object.assign(workingRect, clampRect(next, cropSize.w, cropSize.h));
        updateRectDisplay();
    };

    const applyZoom = () => {
        const naturalW = cropSize.w || img.naturalWidth || 1;
        const targetWidth = Math.max(32, naturalW * baseScale * currentZoom);
        img.style.width = `${targetWidth}px`;
        img.style.height = "auto";
        recalcScale();
        clampWorkingRect();
        updateRectDisplay();
        updateZoomLabel();
    };

    await new Promise((resolve) => {
        if (img.complete && img.naturalWidth > 0) resolve();
        else {
            const once = () => {
                img.removeEventListener("load", once);
                img.removeEventListener("error", once);
                resolve();
            };
            img.addEventListener("load", once);
            img.addEventListener("error", once);
        }
    });

    computeBaseScale();
    applyZoom();

    const state = {
        mode: null,
        handle: null,
        startX: 0,
        startY: 0,
        startRect: null,
    };

    const pointerDown = (event, mode, handle) => {
        event.preventDefault();
        event.stopPropagation();

        state.mode = mode;
        state.handle = handle || null;
        state.startX = event.clientX;
        state.startY = event.clientY;
        state.startRect = { ...workingRect };

        const onMove = (e) => {
            const dx = (e.clientX - state.startX) / scale;
            const dy = (e.clientY - state.startY) / scale;
            let rect = { ...state.startRect };

            if (state.mode === "move") {
                rect.x += dx;
                rect.y += dy;
            } else if (state.mode === "resize") {
                switch (state.handle) {
                    case "nw":
                        rect.x += dx;
                        rect.y += dy;
                        rect.width -= dx;
                        rect.height -= dy;
                        break;
                    case "ne":
                        rect.y += dy;
                        rect.width += dx;
                        rect.height -= dy;
                        break;
                    case "se":
                        rect.width += dx;
                        rect.height += dy;
                        break;
                    case "sw":
                        rect.x += dx;
                        rect.width -= dx;
                        rect.height += dy;
                        break;
                }
            }

            workingRect.x = rect.x;
            workingRect.y = rect.y;
            workingRect.width = rect.width;
            workingRect.height = rect.height;
            clampWorkingRect();
            updateRectDisplay();
        };

        const onUp = () => {
            document.removeEventListener("pointermove", onMove);
            document.removeEventListener("pointerup", onUp);
        };

        document.addEventListener("pointermove", onMove);
        document.addEventListener("pointerup", onUp);
    };

    rectEl.addEventListener("pointerdown", (e) => {
        if (e.target.classList.contains("snjake-crop-handle")) return;
        pointerDown(e, "move");
    });

    handles.forEach((handle) => {
        handle.addEventListener("pointerdown", (e) => pointerDown(e, "resize", handle.dataset.dir));
    });

    img.addEventListener("pointerdown", (e) => {
        if (e.target !== img) return;
        pointerDown(e, "move");
    });

    zoomInput.addEventListener("input", (e) => {
        const next = Math.max(10, Math.min(300, parseInt(e.target.value, 10) || 100));
        currentZoom = next / 100;
        applyZoom();
    });

    const onSizeChange = () => applyManualSize(false);
    [widthInput, heightInput].forEach((input) => {
        input.addEventListener("change", onSizeChange);
        input.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                applyManualSize(e.shiftKey);
            }
        });
    });
    sizeApplyBtn.addEventListener("click", () => applyManualSize(false));
    sizeCenterBtn.addEventListener("click", () => applyManualSize(true));

    const onKey = (e) => {
        if (e.key === "Escape") cleanup();
    };
    window.addEventListener("keydown", onKey);

    cancelBtn.onclick = cleanup;

    resetBtn.onclick = () => {
        const rect = defaultRect(cropSize.w, cropSize.h);
        Object.assign(workingRect, rect);
        clampWorkingRect();
        updateRectDisplay();
    };

    applyBtn.onclick = () => {
        const clamped = clampRect(workingRect, cropSize.w, cropSize.h);
        crop.rect = { ...clamped };
        updateCropJson(node);
        if (node.__sjCropImageWidget?.value) {
            ensureRectMatchesImage(node, false);
        }
        cleanup();
    };
};

const hideWidget = (widget) => {
    if (!widget) return;
    widget.hidden = true;
    widget.visible = false;
    widget.computeSize = () => [0, -4];
    widget.draw = () => 0;
};

const setupCropNode = (node) => {
    if (!node.widgets || !Array.isArray(node.widgets)) return;

    const imageWidget = node.widgets.find((w) => w?.name === "image");
    const cropWidget = node.widgets.find((w) => w?.name === "crop_json");
    if (!imageWidget || !cropWidget) return;

    if (!node.__sjCropInit) {
        node.__sjCropInit = true;
        node.__sjCropImageWidget = imageWidget;
        node.__sjCropJsonWidget = cropWidget;
        hideWidget(cropWidget);

        node.__sjCrop = {
            rect: defaultRect(1, 1),
            imageSize: { w: 1, h: 1 },
            prevImageSize: null,
            image: null,
            imageUrl: null,
            loadToken: 0,
        };

        const originalCallback = imageWidget.callback?.bind(imageWidget);
        imageWidget.callback = async function (value) {
            if (originalCallback) originalCallback(value);
            await loadImagePreview(node, value);
        };

        const editWidget = node.addWidget("button", "Edit Crop", "Open", async () => {
            await openCropModal(node);
        });
        if (editWidget) {
            editWidget.serialize = false;
        }

        const prevRemoved = node.onRemoved?.bind(node);
        node.onRemoved = function () {
            const crop = this.__sjCrop;
            if (crop?.imageUrl) URL.revokeObjectURL(crop.imageUrl);
            if (prevRemoved) prevRemoved();
        };
    }

    const crop = node.__sjCrop;
    const parsed = parseCropJson(cropWidget.value);
    if (parsed) {
        crop.rect = clampRect(
            { x: parsed.x, y: parsed.y, width: parsed.width, height: parsed.height },
            parsed.orig_width || parsed.width || crop.imageSize.w || 1,
            parsed.orig_height || parsed.height || crop.imageSize.h || 1
        );
        crop.imageSize = {
            w: parsed.orig_width || parsed.width || crop.imageSize.w || 1,
            h: parsed.orig_height || parsed.height || crop.imageSize.h || 1,
        };
        crop.prevImageSize = { ...crop.imageSize };
    } else {
        crop.rect = defaultRect(crop.imageSize.w, crop.imageSize.h);
    }

    ensureRectMatchesImage(node, false);
    updateCropJson(node);

    // lazy load preview when node created/configured
    setTimeout(() => {
        loadImagePreview(node, imageWidget.value);
    }, 0);
};

app.registerExtension({
    name: "SnJake.ImageCropNodes",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== CROP_NODE_CLASS) return;
        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origCreated ? origCreated.apply(this, arguments) : undefined;
            setupCropNode(this);
            return r;
        };
        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            setupCropNode(this);
            return r;
        };
    },
    nodeCreated(node) {
        if (node?.type === CROP_NODE_CLASS || node?.comfyClass === CROP_NODE_CLASS) {
            setupCropNode(node);
        }
    },
});
