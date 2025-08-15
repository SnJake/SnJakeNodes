// /ComfyUI/custom_nodes/LoRAManagerWithPreview/preview_lora_loader.js
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { $el } from "../../../scripts/ui.js";

const ITEM_W = 110;
const GAP = 8;

function parseStack(jsonStr) {
  try { const v = JSON.parse(jsonStr || "[]"); return Array.isArray(v) ? v : []; }
  catch { return []; }
}
function stringifyStack(arr) {
  try { return JSON.stringify(arr ?? []); } catch { return "[]"; }
}

function buildModal({ initialSelectedPaths = new Set(), directoryFilter = "/", onSave }) {
  const overlay = $el("div", {
    style: {
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", zIndex: 9999,
      display: "flex", alignItems: "center", justifyContent: "center"
    }
  });
  const panel = $el("div", {
    style: {
      width: "80vw", maxWidth: "1200px", height: "80vh", background: "var(--comfy-menu-bg,#222)",
      border: "1px solid var(--border-color,#444)", borderRadius: "8px", boxShadow: "0 10px 30px rgba(0,0,0,.5)",
      display: "flex", flexDirection: "column", overflow: "hidden"
    }
  });

  const header = $el("div", {
    style: {
      padding: "10px 14px", display: "flex", alignItems: "center", gap: "10px",
      borderBottom: "1px solid var(--border-color,#444)"
    }
  }, [
    $el("div", { innerText: "LoRA Manager", style: { fontWeight: 600, flex: 1 } }),
    $el("button", { innerText: "Отмена", onclick: () => document.body.removeChild(overlay) }),
    $el("button", { innerText: "Сохранить", style: { fontWeight: 600 }, onclick: () => {
      const paths = Array.from(selectedPaths);
      onSave(paths);
      document.body.removeChild(overlay);
    }})
  ]);

  const body = $el("div", {
    style: { flex: 1, overflow: "auto", padding: "12px" }
  });

  const grid = $el("div", {
    style: {
      display: "grid",
      gridTemplateColumns: `repeat(auto-fill,minmax(${ITEM_W}px,1fr))`,
      gap: `${GAP}px`
    }
  });

  body.appendChild(grid);
  panel.appendChild(header);
  panel.appendChild(body);
  overlay.appendChild(panel);

  // текущее множество выбранных
  const selectedPaths = new Set(initialSelectedPaths);

  // загрузить список LoRA
  const url = `/lora_loader_preview/list_loras?directory_filter=${encodeURIComponent(directoryFilter || "/")}`;
  api.fetchApi(url)
    .then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(items => {
      if (!Array.isArray(items)) items = [];
      items.forEach(lora => {
        const card = $el("div", {
          title: lora.path,
          style: {
            border: "1px solid var(--border-color,#555)",
            borderRadius: "6px",
            overflow: "hidden",
            background: "var(--comfy-input-bg,#2a2a2a)",
            cursor: "pointer",
            display: "flex", flexDirection: "column", alignItems: "center"
          },
          onclick: () => {
            const was = selectedPaths.has(lora.path);
            if (was) selectedPaths.delete(lora.path); else selectedPaths.add(lora.path);
            drawSelection(card, selectedPaths.has(lora.path));
          }
        });

        const imgWrap = $el("div", {
          style: {
            width: "100%", aspectRatio: "1 / 1",
            display: "flex", alignItems: "center", justifyContent: "center",
            background: "var(--bg-color,#1c1c1c)"
          }
        });

        if (lora.preview_url) {
          const img = new Image();
          img.src = lora.preview_url;
          img.style.maxWidth = "100%";
          img.style.maxHeight = "100%";
          img.style.objectFit = "contain";
          img.loading = "lazy";
          imgWrap.appendChild(img);
        } else {
          imgWrap.appendChild($el("div", { innerText: "No preview", style: { fontSize: "10px", opacity: .7 }}));
        }

        const name = $el("div", {
          innerText: lora.name.replace(/\.(safetensors|pt|ckpt|bin|pth|pkl|sft)$/i,""),
          style: { fontSize: "11px", padding: "6px", textAlign: "center", width: "100%" }
        });

        card.appendChild(imgWrap);
        card.appendChild(name);
        grid.appendChild(card);

        // первичная подсветка
        drawSelection(card, selectedPaths.has(lora.path));
      });
    })
    .catch(err => {
      grid.appendChild($el("div", { innerText: `Ошибка загрузки (${err})`, style: { color: "tomato" }}));
    });

  document.body.appendChild(overlay);
}

function drawSelection(card, isSelected) {
  card.style.outline = isSelected ? "3px solid #FFD000" : "none"; // жёлтая рамка
  card.style.boxShadow = isSelected ? "0 0 0 1px #FFA800 inset" : "none";
}

// ----- РЕГИСТРАЦИЯ НОДЫ -----
app.registerExtension({
  name: "Comfy.LoRAManagerWithPreview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "LoRAManagerWithPreview") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function() {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

      // виджеты
      const jsonW = this.widgets?.find(w => w.name === "lora_stack_json");
      const dirW  = this.widgets?.find(w => w.name === "directory_filter");
      if (!jsonW) console.error("LoRAManager: hidden lora_stack_json not found");

      // кнопка менеджера
      const btn = this.addWidget("button", "Открыть менеджер", null, () => {
        const stack = parseStack(jsonW?.value);
        const selected = new Set(stack.map(x => x.path));
        const filter = dirW ? dirW.value : "/";
        buildModal({
          initialSelectedPaths: selected,
          directoryFilter: filter,
          onSave: (paths) => {
            // сохранить стек: по возможности сохранить прежние силы
            const prev = parseStack(jsonW?.value);
            const prevMap = Object.fromEntries(prev.map(x => [x.path, x]));
            const next = paths.map(p => ({
              path: p,
              name: (prevMap[p]?.name) ?? p.split("/").pop(),
              strength_model: (prevMap[p]?.strength_model ?? 1.0),
              strength_clip:  (prevMap[p]?.strength_clip  ?? 1.0),
            }));
            jsonW.value = stringifyStack(next);
            rebuildStrengthWidgets(this);       // перерисовать пары полей
            moveWidgetToEnd(this, btn);         // увести кнопку вниз
            this.setDirtyCanvas(true, true);
          }
        });
      }, { serialize: false });

      // первичная отрисовка пар полей из сохранённого стека
      rebuildStrengthWidgets(this);
      moveWidgetToEnd(this, btn);

      // если меняют фильтр директории — это только для списка в модалке,
      // поэтому коллбек можно не трогать.

      return r;
    };

    // на ресайз — ничего особого не требуется
  }
});

// Удаляем старые динамические поля и создаём заново под каждый item
function rebuildStrengthWidgets(node) {
  const jsonW = node.widgets?.find(w => w.name === "lora_stack_json");
  if (!jsonW) return;
  const stack = parseStack(jsonW.value);

  // 1) снести старые «наши» поля
  for (let i = (node.widgets?.length || 0) - 1; i >= 0; i--) {
    const w = node.widgets[i];
    if (w && w._isLoraStrengthWidget) node.widgets.splice(i, 1);
  }

  // 2) создать по паре number-виджетов на каждый item
  stack.forEach((it, idx) => {
    const title = it.name || it.path.split("/").pop();

    const wM = node.addWidget(
      "number",
      `strength_model — ${title}`,
      typeof it.strength_model === "number" ? it.strength_model : 1.0,
      (v) => {
        const s = parseStack(jsonW.value);
        if (!s[idx]) return;
        s[idx].strength_model = clamp(v, -20, 20);
        jsonW.value = stringifyStack(s);
      },
      { min: -20, max: 20, step: 0.01 }
    );
    wM._isLoraStrengthWidget = true;

    const wC = node.addWidget(
      "number",
      `strength_clip — ${title}`,
      typeof it.strength_clip === "number" ? it.strength_clip : 1.0,
      (v) => {
        const s = parseStack(jsonW.value);
        if (!s[idx]) return;
        s[idx].strength_clip = clamp(v, -20, 20);
        jsonW.value = stringifyStack(s);
      },
      { min: -20, max: 20, step: 0.01 }
    );
    wC._isLoraStrengthWidget = true;
  });

  // финал: перерисовать
  node.setSize(node.computeSize());
  node.graph && node.graph.setDirtyCanvas(true, true);
}

function moveWidgetToEnd(node, widget) {
  if (!node.widgets || !widget) return;
  const i = node.widgets.indexOf(widget);
  if (i >= 0 && i !== node.widgets.length - 1) {
    node.widgets.splice(i, 1);
    node.widgets.push(widget);
  }
}

function clamp(v, a, b){ return Math.max(a, Math.min(b, v)); }
