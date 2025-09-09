import { app } from "../../../scripts/app.js";

// Dynamic inputs for LoraSwitchDynamic: starts with 4 pairs, grows to 12
app.registerExtension({
  name: "SnJake.LoraSwitchDynamic.Handler",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "LoraSwitchDynamic") return;

    const DEFAULT_PAIRS = 4;
    const MAX_PAIRS = 12;

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
        if (idx !== -1) { node.removeInput(idx); removed = true; }
        idx = node.inputs ? node.inputs.findIndex((inp) => inp?.name === `model_${i}`) : -1;
        if (idx !== -1) { node.removeInput(idx); removed = true; }
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
          (v) => { /* keep serialized only */ },
          { min: 1, max: MAX_PAIRS, step: 1 }
        );
        w.hidden = true;
        w.serialize = true;
      }
      return w;
    }

    function countExistingPairs(node) {
      if (!node.inputs || node.inputs.length === 0) return 0;
      let maxIdx = 0;
      for (const inp of node.inputs) {
        const m = inp?.name && inp.name.match(/^(model|clip)_(\d+)$/);
        if (m) maxIdx = Math.max(maxIdx, parseInt(m[2], 10) || 0);
      }
      return maxIdx;
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

    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      // Ensure we have the hidden state widget and start from DEFAULT_PAIRS
      const w = getPairsWidget(this);
      w.value = DEFAULT_PAIRS;
      // Comfy creates all optional inputs on initial class load; prune to default
      // Do it after a tick to avoid interfering with node construction
      setTimeout(() => {
        prunePairs(this, DEFAULT_PAIRS);
        ensurePairs(this, DEFAULT_PAIRS);
      }, 0);
      return r;
    };

    nodeType.prototype.onConfigure = function (info) {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      // Ensure hidden widget exists; adjust after deserialization finishes
      const w = getPairsWidget(this);
      const initial = Math.max(1, Math.min(MAX_PAIRS, Math.round(w.value || DEFAULT_PAIRS)));
      setTimeout(() => {
        // Determine desired count: respect saved value but also consider current connections
        const lastFull = lastFullyConnectedIndex(this);
        let desired = Math.max(DEFAULT_PAIRS, Math.min(MAX_PAIRS, Math.max(initial, lastFull + 1)));
        prunePairs(this, desired);
        ensurePairs(this, desired);
        w.value = desired;
      }, 0);
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
      // Grow when last visible pair is fully connected
      let currentPairs = Math.max(1, countExistingPairs(this));
      if (currentPairs < MAX_PAIRS && bothEndsConnected(this, currentPairs)) {
        ensurePairs(this, currentPairs + 1);
        currentPairs = currentPairs + 1;
      }

      // Shrink to keep exactly one empty pair after the highest fully-connected pair
      const lastFull = lastFullyConnectedIndex(this);
      const desired = Math.max(DEFAULT_PAIRS, Math.min(MAX_PAIRS, lastFull + 1));
      if (countExistingPairs(this) > desired) {
        prunePairs(this, desired);
      }

      const w = getPairsWidget(this);
      w.value = Math.max(DEFAULT_PAIRS, countExistingPairs(this));
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
