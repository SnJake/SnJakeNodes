import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { $el } from "../../../scripts/ui.js";

// --- Constants ---
const WIDGET_MIN_HEIGHT = 150;
const ITEM_WIDTH = 100;
const ITEM_MARGIN = 5;

// --- LoraPreviewWidget ---
function LoraPreviewWidget(node, widgetName) {
    this.widgetName = widgetName;
    this.node = node;
    this.node.loraPreviewWidget = this;

    this.selectedLoraWidget = node.widgets.find(w => w.name === "selected_lora");
    if (!this.selectedLoraWidget) {
        console.error(`LoraLoaderWithPreview (Node ${node.id}): Could not find 'selected_lora' widget.`);
    }

    this.element = $el("div.lora-preview-widget", {
        style: {
            // УБИРАЕМ !important отсюда, будем ставить в updateSize
            display: "grid", // Начальное значение
            gridTemplateColumns: `repeat(auto-fill, minmax(${ITEM_WIDTH}px, 1fr))`,
            // Остальные стили оставляем как в ПРОШЛОМ ответе (без position, top, left, width)
            width: "100%", // Попробуем вернуть 100%
            minHeight: `${WIDGET_MIN_HEIGHT}px`,
            overflowY: "auto",
            overflowX: "hidden",
            border: "1px solid var(--border-color, #444)",
            marginTop: "5px",
            marginBottom: "5px",
            gridAutoFlow: "row",
            gap: `${ITEM_MARGIN * 2}px`,
            padding: `${ITEM_MARGIN}px`,
            boxSizing: "border-box"
        }
    });

    this.loras = [];
    // Removed this.images cache - relying on browser cache now
    this.currentFilter = "/";
    this.isLoading = false;

    node.addDOMWidget(widgetName, "LORA_PREVIEW", this.element, {
        getValue: () => undefined,
        setValue: (v) => {},
    });

    this.fetchLoras = this.fetchLoras.bind(this);
    this.renderPreviews = this.renderPreviews.bind(this);
    this.updateSize = this.updateSize.bind(this);
    this.selectLora = this.selectLora.bind(this);
    this.loadImageForItem = this.loadImageForItem.bind(this); // Renamed for clarity

    this.fetchLoras();
}

LoraPreviewWidget.prototype.fetchLoras = function(filter = "/") {
    if (this.isLoading) return;
    this.isLoading = true;
    this.element.innerHTML = '<span style="color: var(--descrip-text, #888); padding: 5px; grid-column: 1 / -1;">Loading...</span>';

    this.currentFilter = filter || "/";
    const url = `/lora_loader_preview/list_loras?directory_filter=${encodeURIComponent(this.currentFilter)}`;

    api.fetchApi(url)
        .then(response => response.ok ? response.json() : Promise.reject(`HTTP error! status: ${response.status}`))
        .then(data => {
            this.loras = data;
            this.renderPreviews();
        })
        .catch(error => {
            console.error("Error fetching LoRAs:", error);
            this.element.innerHTML = `<span style="color: var(--error-text, red); padding: 5px; grid-column: 1 / -1;">Error loading LoRAs. Check console.</span>`;
        })
        .finally(() => {
            this.isLoading = false;
        });
};

// New function to load image specifically for a given DOM item and lora data
LoraPreviewWidget.prototype.loadImageForItem = function(lora, imgContainer, itemElement) {
    const url = lora.preview_url;
    imgContainer.innerHTML = ''; // Clear previous content (placeholder/error)

    if (!url) {
        imgContainer.innerHTML = `<span style="font-size: 10px; color: var(--descrip-text, #666); text-align: center;">No Preview</span>`;
        return;
    }

    // Check if the item element is still in the main widget container
    if (!this.element.contains(itemElement)) {
        //console.log(`[LoraLoaderPreview] Item for ${lora.name} no longer in DOM, skipping image load.`);
        return; // Don't load if the item was removed (e.g., filter changed quickly)
    }

    imgContainer.innerHTML = `<span style="font-size: 9px; color: var(--descrip-text, #555); text-align: center;">Loading...</span>`;

    const img = new Image();
    img.src = url;
    img.style.maxWidth = "100%";
    img.style.maxHeight = "100%";
    img.style.objectFit = "contain";
    img.loading = "lazy";

    img.onload = () => {
        // Double-check if the item element is *still* in the DOM before appending
        if (this.element.contains(itemElement)) {
            imgContainer.innerHTML = ''; // Clear loading text
            imgContainer.appendChild(img); // Append the loaded image
        } else {
             //console.log(`[LoraLoaderPreview] Item for ${lora.name} removed before image onload finished.`);
        }
    };

    img.onerror = (err) => {
        console.error(`[LoraLoaderPreview] Image failed to load: ${url}`, err);
         // Double-check if the item element is *still* in the DOM
         if (this.element.contains(itemElement)) {
             imgContainer.innerHTML = `<span style="font-size: 10px; color: var(--error-text, #666); text-align: center;">Load Error</span>`;
         }
    };
};


LoraPreviewWidget.prototype.selectLora = function(loraPath) {
    if (this.selectedLoraWidget) {
        const currentVal = this.selectedLoraWidget.value;
        if (currentVal !== loraPath) {
            this.selectedLoraWidget.value = loraPath;
            this.renderPreviews(); // Re-render needed to update highlights
            if (this.selectedLoraWidget.callback) {
                 this.selectedLoraWidget.callback(loraPath);
            }
            this.node.setDirtyCanvas(true, false);
        }
    } else {
        console.error("LoraLoaderWithPreview: Cannot set selected LoRA, hidden widget not found.");
    }
};

LoraPreviewWidget.prototype.renderPreviews = function() {
    this.element.innerHTML = ""; // Clear previous content

    if (this.loras.length === 0) {
        this.element.innerHTML = '<span style="color: var(--descrip-text, #888); padding: 5px; grid-column: 1 / -1;">No LoRAs found in this directory.</span>';
        return;
    }

    const selectedPath = this.selectedLoraWidget ? this.selectedLoraWidget.value : "None";

    this.loras.forEach(lora => {
        const item = $el("div.lora-preview-item", {
            title: lora.path, // Show full path on hover
            style: { /* Styles from previous version */
                width: `${ITEM_WIDTH}px`,
                border: "1px solid var(--border-color, #555)",
                borderRadius: "4px",
                overflow: "hidden",
                cursor: "pointer",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                backgroundColor: "var(--comfy-input-bg, #282828)",
                transition: "border-color 0.2s, box-shadow 0.2s",
                paddingBottom: "3px",
                boxSizing: "border-box",
            }
        });

        if (lora.path === selectedPath) {
            item.style.borderColor = "var(--accent-color, #00aaff)";
            item.style.boxShadow = "0 0 5px var(--accent-color, #00aaff)";
        }

        const imgContainer = $el("div.lora-img-container", {
            style: { /* Styles from previous version */
                width: `${ITEM_WIDTH - 10}px`,
                height: `${ITEM_WIDTH - 10}px`,
                marginTop: "5px",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                backgroundColor: "var(--bg-color, #1c1c1c)",
                overflow: "hidden",
            }
        });

        const nameLabel = $el("div.lora-preview-name", {
            textContent: lora.name.replace(/\.(safetensors|pt|ckpt|bin|pth|pkl|sft)$/i, ""),
             style: { /* Styles from previous version */
                fontSize: "10px",
                color: "var(--input-text, #ccc)",
                textAlign: "center",
                padding: "3px 5px",
                wordBreak: "break-word",
                width: "100%",
                boxSizing: "border-box",
                marginTop: "auto",
            }
        });

        item.appendChild(imgContainer);
        item.appendChild(nameLabel);

        item.onclick = () => this.selectLora(lora.path);

        this.element.appendChild(item);

        // IMPORTANT: Load image *after* the item element is added to the DOM
        // This ensures the container exists when onload/onerror fires.
        this.loadImageForItem(lora, imgContainer, item);
    });
};

LoraPreviewWidget.prototype.updateSize = function() {
    // Проверка существования элемента и ноды
    if (!this.element || !this.node || !this.node.widgets) {
        return;
    }
    const graph = this.node.graph; // Сохраняем ссылку для удобства

    // --- Расчет доступной высоты (аналогично прошлому разу) ---
    let lastWidgetY = 0;
    for(const w of this.node.widgets) {
        if (!w || w.type === "LORA_PREVIEW" || w.type === "HIDDEN" || w.name === this.widgetName || w.last_y == null) continue;
        // Используем LiteGraph.NODE_WIDGET_HEIGHT, если он есть, иначе дефолтное значение
        const widgetHeight = LiteGraph.NODE_WIDGET_HEIGHT ? LiteGraph.NODE_WIDGET_HEIGHT : 20;
        const computedHeight = w.computeSize ? w.computeSize()[1] : widgetHeight;
        lastWidgetY = Math.max(lastWidgetY, w.last_y + computedHeight);
    }
    lastWidgetY += 4;
    const titleHeight = this.node.constructor.title_mode === LiteGraph.NO_TITLE ? 0 : (LiteGraph.NODE_TITLE_HEIGHT || 30);
    lastWidgetY = Math.max(lastWidgetY, titleHeight);


    // --- ИСПРАВЛЕНИЕ ОШИБКИ isLive ---
    // Проверяем, существует ли graph и метод isLive перед вызовом
    const isLiveMode = graph && typeof graph.isLive === 'function' ? graph.isLive() : false;
    const bottomMargin = isLiveMode ? 0 : 15; // Нижний отступ
    // --- КОНЕЦ ИСПРАВЛЕНИЯ ---

    let availableHeight = this.node.size[1] - lastWidgetY - bottomMargin;
    availableHeight = Math.max(0, availableHeight);
    let targetHeight = Math.max(WIDGET_MIN_HEIGHT, availableHeight);

   // --- Применение стилей ---
   const elementStyle = this.element.style;

   // Управляем видимостью и ВСЕГДА ставим display grid, если не свернуто
   if (this.node.flags.collapsed) {
        if (elementStyle.display !== "none") {
           elementStyle.display = "none";
        }
   } else {
       // --- ПРИНУДИТЕЛЬНО ставим grid ---
       if (elementStyle.display !== "grid") {
           elementStyle.display = "grid";
       }
       // --- КОНЕЦ ---

       // Обновляем высоту
       const currentHeight = parseInt(elementStyle.height) || 0;
       if (Math.abs(currentHeight - targetHeight) > 2) {
            elementStyle.height = `${targetHeight}px`;
       }
   }
   // Ширину и позицию не трогаем
};

// --- App Extension --- (No changes needed here, keep the previous version)
// --- App Extension ---
app.registerExtension({
    name: "Comfy.LoraLoaderWithPreview",
    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name === "LoraLoaderWithPreview") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                // ... (код onNodeCreated без изменений из прошлого ответа) ...
                 const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                 const previewWidget = new LoraPreviewWidget(this, "lora_preview_widget");
                 this.loraPreviewWidget = previewWidget;

                 const directoryFilterWidget = this.widgets.find(w => w.name === "directory_filter");
                 if (directoryFilterWidget) {
                     const originalCallback = directoryFilterWidget.callback;
                     directoryFilterWidget.callback = (value) => {
                          if(originalCallback) {
                             originalCallback.call(directoryFilterWidget, value);
                          }
                          if (this.loraPreviewWidget) {
                              this.loraPreviewWidget.fetchLoras(value);
                          } else {
                              console.error(`LoraLoaderWithPreview (Node ${this.id}): loraPreviewWidget not found during filter callback.`);
                          }
                     };
                 } else {
                     console.error(`LoraLoaderWithPreview (Node ${this.id}): Could not find 'directory_filter' widget.`);
                 }
                 return r;
            };

             const onResize = nodeType.prototype.onResize;
             nodeType.prototype.onResize = function(size) {
                const r = onResize ? onResize.apply(this, arguments) : undefined;
                 if (this.loraPreviewWidget) {
                    requestAnimationFrame(() => {
                        if (this.loraPreviewWidget) {
                            this.loraPreviewWidget.updateSize();
                        }
                    });
                 }
                 return r;
             }

             // Убедимся, что onDrawForeground не вызывает updateSize
             if (nodeType.prototype.onDrawForeground) {
                 const onDrawForegroundOriginal = nodeType.prototype.onDrawForeground;
                 nodeType.prototype.onDrawForeground = function (ctx, canvas) {
                     return onDrawForegroundOriginal ? onDrawForegroundOriginal.apply(this, arguments) : undefined;
                 }
             }

              const onCollapse = nodeType.prototype.collapse;
               nodeType.prototype.collapse = function() {
                    const wasCollapsed = this.flags.collapsed;
                    const r = onCollapse ? onCollapse.apply(this, arguments) : undefined;
                    if (wasCollapsed !== this.flags.collapsed && this.loraPreviewWidget) {
                        requestAnimationFrame(() => {
                           if (this.loraPreviewWidget) {
                               this.loraPreviewWidget.updateSize();
                           }
                       });
                    }
                    return r;
                }
        }
    },
});