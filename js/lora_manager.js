// /ComfyUI/custom_nodes/LoRAManagerWithPreview/preview_lora_loader.js
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { $el } from "../../../scripts/ui.js";

// ITEM_W теперь инициализируется в buildModal через loadScale()
const GAP = 8;

function joinPath(a, b){
  if (!a || a === "/") return b || "";
  if (!b) return a;
  return `${a.replace(/^\/|\/$/g,"")}/${b.replace(/^\/|\/$/g,"")}`;
}
function parentPath(p){
  const parts = (p || "/").split("/").filter(Boolean);
  parts.pop();
  return parts.length ? `/${parts.join("/")}` : "/";
}

// --- Persist keys
const LS_KEYS = { scale: "loraMgr.scale", lastDir: "loraMgr.lastDir" };

function loadScale() {
  const v = parseInt(localStorage.getItem(LS_KEYS.scale), 10);
  return Number.isFinite(v) ? Math.min(220, Math.max(80, v)) : 110;
}
function saveScale(v) { localStorage.setItem(LS_KEYS.scale, String(v)); }

function loadLastDir() {
  const d = localStorage.getItem(LS_KEYS.lastDir);
  return d && d.startsWith("/") ? d : "/";
}
function saveLastDir(d) { localStorage.setItem(LS_KEYS.lastDir, d || "/"); }

function parseStack(jsonStr) {
  try { const v = JSON.parse(jsonStr || "[]"); return Array.isArray(v) ? v : []; }
  catch { return []; }
}
function stringifyStack(arr) {
  try { return JSON.stringify(arr ?? []); } catch { return "[]"; }
}

function buildModal({ initialSelectedPaths = new Set(), directoryFilter = "/", onSave }) {
  let ITEM_W = loadScale(); // вместо жёсткого 110
  const overlay = $el("div", { style: { position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", zIndex: 9999, display: "flex", alignItems: "center", justifyContent: "center" }});
  const panel = $el("div", { style: { width: "80vw", maxWidth: "1200px", height: "80vh", background: "var(--comfy-menu-bg,#222)", border: "1px solid var(--border-color,#444)", borderRadius: "8px", boxShadow: "0 10px 30px rgba(0,0,0,.5)", display: "flex", flexDirection: "column", overflow: "hidden" }});

  // --- header ---
  const titleEl = $el("div", { innerText: "LoRA Manager", style: { fontWeight: 600 }});
  const breadcrumb = $el("div", { style: { fontSize: "12px", opacity: .85 }});
  const searchInput = $el("input", {
    type: "search",
    placeholder: "Поиск в текущей папке…",
    style: { width: "240px" },
    oninput: () => renderGrid()
  });
  const scaleInput = $el("input", {
    type: "range", min: 80, max: 220, step: 10, value: ITEM_W,
    oninput: (e) => {
      ITEM_W = parseInt(e.target.value, 10) || 110;
      saveScale(ITEM_W);        // <-- сохраняем масштаб
      renderGrid();
    }
  });
  const scaleLabel = $el("span", { innerText: "Scale", style: { fontSize: "12px", opacity: .85 } });
  const counter = $el("div", { style: { fontSize: "12px", opacity: .85, justifySelf: "end" }});

  const header = $el("div", {
    style: { padding: "10px 14px", display: "grid", gridTemplateColumns: "1fr auto auto auto auto auto", gap: "10px", alignItems: "center", borderBottom: "1px solid var(--border-color,#444)" }
  }, [
    $el("div", {}, [titleEl, breadcrumb]),
    scaleLabel,
    scaleInput,
    $el("div", {}, [searchInput]),
    counter, // <-- добавили
    $el("div", { style: { display: "flex", gap: "8px", justifySelf: "end" }}, [
      $el("button", { innerText: "Cancel", onclick: () => document.body.removeChild(overlay) }),
      $el("button", { innerText: "Safe", style: { fontWeight: 600 }, onclick: () => { onSave(Array.from(selectedPaths)); document.body.removeChild(overlay); }})
    ])
  ]);

  // --- body ---
  const body = $el("div", { style: { flex: 1, overflow: "auto", padding: "12px" }});
  const grid = $el("div", { style: { display: "grid", gap: `${GAP}px` }});
  body.appendChild(grid);

  panel.appendChild(header);
  panel.appendChild(body);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);

  // --- state ---
  const selectedPaths = new Set(initialSelectedPaths);
  let currentDir = (directoryFilter && directoryFilter !== "/" ? directoryFilter : loadLastDir()) || "/";
  let currentDirs = [];
  let currentFiles = [];

  function updateCounter() {
    const d = currentDirs?.length || 0;
    const f = currentFiles?.length || 0;
    counter.innerText = `Folders: ${d} · Files: ${f} · Total: ${d + f}`;
  }

  function renderBreadcrumb() {
    const parts = currentDir === "/" ? [] : currentDir.split("/").filter(Boolean);
    breadcrumb.innerHTML = "";
    const frag = [];
    const home = $el("a", { href: "#", onclick: (e)=>{ e.preventDefault(); loadDir("/"); }, innerText: " / " });
    frag.push(home);
    let acc = "";
    parts.forEach((p, i) => {
      acc = acc ? `${acc}/${p}` : p;
      frag.push($el("span", { innerText: " " }));
      frag.push($el("a", {
        href: "#",
        onclick: (e)=>{ e.preventDefault(); loadDir("/" + acc); },
        innerText: "/" + p
      }));
    });
    breadcrumb.append(...frag);
  }

  function cardBase(opts={}) {
    return $el("div", Object.assign({
      style: {
        border: "1px solid var(--border-color,#555)",
        borderRadius: "6px",
        overflow: "hidden",
        background: "var(--comfy-input-bg,#2a2a2a)",
        cursor: "pointer",
        display: "flex", flexDirection: "column", alignItems: "center"
      }
    }, opts));
  }

  function renderGrid() {
    renderBreadcrumb();
    grid.style.gridTemplateColumns = `repeat(auto-fill,minmax(${ITEM_W}px,1fr))`;
    grid.innerHTML = "";

    // "… Назад" — только если НЕ корень
    if (currentDir !== "/") {
      const back = cardBase({ onclick: ()=> loadDir(parentPath(currentDir)) });
      const icon = $el("div", { innerText: "…", style: { fontSize: Math.round(ITEM_W * 0.6) + "px", lineHeight: 1, paddingTop: "8px" }});
      const name = $el("div", { innerText: "Назад", style: { fontSize: "11px", padding: "6px", textAlign: "center", width: "100%" }});
      back.appendChild(icon); back.appendChild(name);
      grid.appendChild(back);
    }

    // Папки — иконка 📁 (масштабируемая)
    currentDirs.forEach(d => {
      const card = cardBase({ onclick: ()=> loadDir("/" + d.path.replace(/^\/?/,"")) });
      const icon = $el("div", { innerText: "📁", style: { fontSize: Math.round(ITEM_W * 0.6) + "px", lineHeight: 1, paddingTop: "8px" }});
      const name = $el("div", { innerText: d.name, style: { fontSize: "11px", padding: "6px", textAlign: "center", width: "100%" }});
      card.appendChild(icon); card.appendChild(name);
      grid.appendChild(card);
    });

    // Файлы — с локальным поиском по имени в ТЕКУЩЕЙ папке
    const q = (searchInput.value || "").trim().toLowerCase();
    const files = q ? currentFiles.filter(f => f.name.toLowerCase().includes(q)) : currentFiles;

    files.forEach(lora => {
      const card = cardBase({
        title: lora.path,
        onclick: () => {
          const was = selectedPaths.has(lora.path);
          if (was) selectedPaths.delete(lora.path); else selectedPaths.add(lora.path);
          drawSelection(card, selectedPaths.has(lora.path));
        }
      });

      const imgWrap = $el("div", {
        style: { width: "100%", aspectRatio: "1 / 1", display: "flex", alignItems: "center", justifyContent: "center", background: "var(--bg-color,#1c1c1c)" }
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
      drawSelection(card, selectedPaths.has(lora.path));
    });

    updateCounter();
  }

  function loadDir(dir) {
    const target = dir || "/";
    fetch(`/lora_loader_preview/list_dir?directory=${encodeURIComponent(target)}`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(({cwd, dirs, files}) => {
        currentDir = cwd || target;
        saveLastDir(currentDir);              // <-- сохраняем путь
        currentDirs = Array.isArray(dirs) ? dirs : [];
        currentFiles = Array.isArray(files) ? files : [];
        renderGrid();
      })
      .catch(err => {
        grid.innerHTML = "";
        grid.appendChild($el("div", { innerText: `Ошибка загрузки (${err})`, style: { color: "tomato" }}));
      });
  }

  // первая загрузка
  loadDir(currentDir);
}

function drawSelection(card, isSelected) {
  card.style.outline = isSelected ? "3px solid #FFD000" : "none"; // жёлтая рамка
  card.style.boxShadow = isSelected ? "0 0 0 1px #FFA800 inset" : "none";
}

function ensureDynamicStrengthsBuilt(node) {
  const jsonW = node.widgets?.find(w => w.name === "lora_stack_json");
  if (!jsonW) return;
  const stack = parseStack(jsonW.value);
  const alreadyBuilt = node.widgets?.some(w => w._isLoraStrengthWidget);
  if (stack.length > 0 && !alreadyBuilt) {
    rebuildStrengthWidgets(node);
    const btn = node.widgets?.find(w => w.type === "button" && w.name === "Open Manager");
    moveWidgetToEnd(node, btn);
    node.setDirtyCanvas(true, true);
  }
}

// ----- РЕГИСТРАЦИЯ НОДЫ -----
app.registerExtension({
  name: "Comfy.LoRAManagerWithPreview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "LoRAManagerWithPreview") return;

    // --- Обработка создания ноды (когда добавляется в workflow) ---
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function() {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

      // Включаем сериализацию, чтобы значения виджетов сохранялись в workflow
      this.serialize_widgets = true;
      const jsonW = this.widgets?.find(w => w.name === "lora_stack_json");
      if (jsonW) jsonW.serialize = true;

      const dirW  = this.widgets?.find(w => w.name === "directory_filter");
      if (!jsonW) console.error("LoRAManager: hidden lora_stack_json not found");

      // Добавляем кнопку менеджера
      const btn = this.addWidget("button", "Open Manager", null, () => {
        const stack = parseStack(jsonW?.value);
        const selected = new Set(stack.map(x => x.path));
        const filter = dirW ? dirW.value : "/";
        buildModal({
          initialSelectedPaths: selected,
          directoryFilter: filter,
          onSave: (paths) => {
            const prev = parseStack(jsonW?.value);
            const prevMap = Object.fromEntries(prev.map(x => [x.path, x]));
            const next = paths.map(p => ({
              path: p,
              name: (prevMap[p]?.name) ?? p.split("/").pop(),
              strength_model: (prevMap[p]?.strength_model ?? 1.0),
              strength_clip:  (prevMap[p]?.strength_clip  ?? 1.0),
            }));
            jsonW.value = stringifyStack(next);
            rebuildStrengthWidgets(this);
            moveWidgetToEnd(this, btn);
            this.setDirtyCanvas(true, true);
          }
        });
      }, { serialize: false });

      // Первая отрисовка (при создании ноды с нуля стек будет пуст)
      rebuildStrengthWidgets(this);
      moveWidgetToEnd(this, btn);

      // Страховка на случай, если значения применяются с задержкой
      setTimeout(() => ensureDynamicStrengthsBuilt(this), 0);

      return r;
    };

    // --- Обработка конфигурации (когда workflow ЗАГРУЖАЕТСЯ) ---
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      // В этот момент lora_stack_json уже должен содержать сохранённое значение
      ensureDynamicStrengthsBuilt(this);
      // Иногда Comfy применяет widgets_values чуть позже — подстрахуемся ещё раз
      setTimeout(() => ensureDynamicStrengthsBuilt(this), 0);
      return r;
    };
  }
});

// Удаляем старые динамические поля и создаём заново под каждый item
function rebuildStrengthWidgets(node) {
  const jsonW = node.widgets?.find(w => w.name === "lora_stack_json");
  if (!jsonW) return;
  const stack = parseStack(jsonW.value);

  // --- запоминаем текущий размер до перестройки
  const prevSize = node.size ? [...node.size] : null;

  // 1) удалить старые динамические поля
  for (let i = (node.widgets?.length || 0) - 1; i >= 0; i--) {
    const w = node.widgets[i];
    if (w && w._isLoraStrengthWidget) node.widgets.splice(i, 1);
  }

  // 2) создать новые поля
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

  // 3) не сжимать размер: оставляем пользовательский, но гарантируем минимальный
  const min = node.computeSize();
  const newW = prevSize ? Math.max(prevSize[0], min[0]) : min[0];
  const newH = prevSize ? Math.max(prevSize[1], min[1]) : min[1];
  node.setSize([newW, newH]);

  node.setDirtyCanvas(true, true);
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

