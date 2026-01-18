import { app } from "../../../scripts/app.js";

// Dynamic inputs for LoraSwitchDynamic: starts with 4 pairs, grows to 12
app.registerExtension({
  name: "SnJake.LoraSwitchDynamic.Handler",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "LoraSwitchDynamic") return;

    const DEFAULT_PAIRS = 4;
    const MAX_PAIRS = 12;

    function clampVisiblePairs(value) {
      const num =
        typeof value === "number"
          ? value
          : value != null
          ? Number(value)
          : NaN;
      if (!Number.isFinite(num)) return DEFAULT_PAIRS;
      return Math.max(DEFAULT_PAIRS, Math.min(MAX_PAIRS, Math.round(num)));
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    const onConfigure = nodeType.prototype.onConfigure;
    const onConnectionsChange = nodeType.prototype.onConnectionsChange;

    function inputByName(node, name) {
      if (!node.inputs) return null;
      return node.inputs.find((inp) => inp && inp.name === name) || null;
    }

    function hasInput(node, name) {
      return !!inputByName(node, name);
    }

    function ensurePairs(node, n) {
      const target = Math.max(0, Math.min(MAX_PAIRS, n | 0));
      let added = false;
      for (let i = 1; i <= target; i++) {
        if (!hasInput(node, `model_${i}`)) {
          node.addInput(`model_${i}`, "MODEL");
          added = true;
        }
        if (!hasInput(node, `clip_${i}`)) {
          node.addInput(`clip_${i}`, "CLIP");
          added = true;
        }
      }
      if (added) {
        node.computeSize?.();
        (node.graph || app.graph)?.setDirtyCanvas?.(true, true);
      }
    }

    function prunePairs(node, keepN) {
      const keep = Math.max(1, Math.min(MAX_PAIRS, keepN | 0));
      let removed = false;
      for (let i = MAX_PAIRS; i > keep; i--) {
        let idx;
        idx = node.inputs ? node.inputs.findIndex((inp) => inp?.name === `clip_${i}`) : -1;
        if (idx !== -1) {
          node.removeInput(idx);
          removed = true;
        }
        idx = node.inputs ? node.inputs.findIndex((inp) => inp?.name === `model_${i}`) : -1;
        if (idx !== -1) {
          node.removeInput(idx);
          removed = true;
        }
      }
      if (removed) {
        node.computeSize?.();
        (node.graph || app.graph)?.setDirtyCanvas?.(true, true);
      }
    }

    function getPairsWidget(node) {
      let w = node.widgets?.find((w) => w.name === "pairs_visible");
      if (!w) {
        w = node.addWidget(
          "number",
          "pairs_visible",
          DEFAULT_PAIRS,
          (v) => {
            /* keep serialized only */
          },
          { min: 1, max: MAX_PAIRS, step: 1 }
        );
        w.hidden = true;
        w.serialize = true;
      }
      return w;
    }

    function pairIndexFromName(name) {
      if (typeof name !== "string") return 0;
      const match = name.match(/^(model|clip)_(\d+)$/);
      return match ? parseInt(match[2], 10) || 0 : 0;
    }

    function countExistingPairs(node) {
      if (!node.inputs || node.inputs.length === 0) return 0;
      let maxIdx = 0;
      for (const inp of node.inputs) {
        const idx = pairIndexFromName(inp?.name);
        if (idx > maxIdx) maxIdx = idx;
      }
      return maxIdx;
    }

    function hasSerializedLink(input) {
      if (!input) return false;
      if (input.link != null) return true;
      if (Array.isArray(input.links)) {
        return input.links.some((link) => link != null);
      }
      return false;
    }

    function inferSerializedConnectionState(info) {
      if (!info || !Array.isArray(info.inputs)) {
        return { lastFull: 0, lastAny: 0 };
      }
      const seen = new Map();
      let lastAny = 0;
      for (const inp of info.inputs) {
        const idx = pairIndexFromName(inp?.name);
        if (!idx) continue;
        if (!hasSerializedLink(inp)) continue;
        if (idx > lastAny) lastAny = idx;
        const entry = seen.get(idx) || { model: false, clip: false };
        if (inp.name.startsWith("model_")) {
          entry.model = true;
        } else if (inp.name.startsWith("clip_")) {
          entry.clip = true;
        }
        seen.set(idx, entry);
      }
      let lastFull = 0;
      for (const [idx, entry] of seen.entries()) {
        if (entry.model && entry.clip) {
          lastFull = Math.max(lastFull, idx);
        }
      }
      return { lastFull, lastAny };
    }

    function readPersistedPairs(info) {
      if (!info || !info.properties) return null;
      const raw = info.properties.pairs_visible;
      const num =
        typeof raw === "number"
          ? raw
          : raw != null
          ? Number(raw)
          : NaN;
      return Number.isFinite(num) ? num : null;
    }

    function getSavedPairCount(info) {
      if (!info) return DEFAULT_PAIRS;
      const persisted = readPersistedPairs(info);
      const { lastFull, lastAny } = inferSerializedConnectionState(info);
      const baseline = lastFull > 0 ? lastFull + 1 : DEFAULT_PAIRS;
      const desired = Math.max(baseline, lastAny);
      const fallback = desired || DEFAULT_PAIRS;
      if (persisted != null) {
        return clampVisiblePairs(Math.max(persisted, fallback));
      }
      return clampVisiblePairs(fallback);
    }

    function isConnected(inputSlot) {
      if (!inputSlot) return false;
      return (
        inputSlot.link != null ||
        (Array.isArray(inputSlot.links) && inputSlot.links.length > 0)
      );
    }

    function bothEndsConnected(node, idx) {
      const m = inputByName(node, `model_${idx}`);
      const c = inputByName(node, `clip_${idx}`);
      return isConnected(m) && isConnected(c);
    }

    function updatePersistedPairs(node, value) {
      const clamped = clampVisiblePairs(value);
      const w = getPairsWidget(node);
      if (w) w.value = clamped;
      node.properties = node.properties || {};
      node.properties.pairs_visible = clamped;
    }

    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      const w = getPairsWidget(this);
      const initial = clampVisiblePairs(w.value ?? DEFAULT_PAIRS);
      prunePairs(this, initial);
      ensurePairs(this, initial);
      updatePersistedPairs(this, initial);
      return r;
    };

    nodeType.prototype.onConfigure = function (info) {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      const w = getPairsWidget(this);
      const savedPairs = getSavedPairCount(info);
      const widgetPairs = clampVisiblePairs(w.value ?? savedPairs);
      w.value = widgetPairs;
      const desired = clampVisiblePairs(Math.max(widgetPairs, savedPairs));
      prunePairs(this, desired);
      ensurePairs(this, desired);
      updatePersistedPairs(this, desired);
      return r;
    };

    nodeType.prototype.onConnectionsChange = function (
      type,
      slotIndex,
      isConnected,
      link_info,
      io_slot
    ) {
      const r = onConnectionsChange
        ? onConnectionsChange.apply(this, arguments)
        : undefined;
      let currentPairs = Math.max(1, countExistingPairs(this));
      if (currentPairs < MAX_PAIRS && bothEndsConnected(this, currentPairs)) {
        ensurePairs(this, currentPairs + 1);
        currentPairs = currentPairs + 1;
      }

      const lastFull = lastFullyConnectedIndex(this);
      const desired = Math.max(DEFAULT_PAIRS, Math.min(MAX_PAIRS, lastFull + 1));
      if (countExistingPairs(this) > desired) {
        prunePairs(this, desired);
      }

      const totalPairs = Math.max(DEFAULT_PAIRS, countExistingPairs(this));
      updatePersistedPairs(this, totalPairs);
      return r;
    };

    function lastFullyConnectedIndex(node) {
      const total = countExistingPairs(node);
      let last = 0;
      for (let i = 1; i <= total; i++) {
        if (bothEndsConnected(node, i)) last = i;
      }
      return last;
    }
  },
});
