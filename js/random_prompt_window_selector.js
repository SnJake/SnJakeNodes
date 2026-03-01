import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "SnJake.RandomPromptWindowSelector.UI",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SnJakeRandomPromptWindowSelector") return;

    const MAX_WINDOWS = 20;

    function findWidget(node, name) {
      return node.widgets?.find((w) => w?.name === name) || null;
    }

    function clampCount(v) {
      const n = Number(v);
      if (!Number.isFinite(n)) return 1;
      return Math.max(1, Math.min(MAX_WINDOWS, Math.round(n)));
    }

    function moveWidgetToEnd(node, widget) {
      if (!node.widgets || !widget) return;
      const idx = node.widgets.indexOf(widget);
      if (idx >= 0 && idx !== node.widgets.length - 1) {
        node.widgets.splice(idx, 1);
        node.widgets.push(widget);
      }
    }

    function applyVisibility(node) {
      const countWidget = findWidget(node, "windows_visible");
      if (!countWidget) return;

      const count = clampCount(countWidget.value);
      countWidget.value = count;
      countWidget.hidden = true;
      countWidget.serialize = true;

      for (let i = 1; i <= MAX_WINDOWS; i++) {
        const w = findWidget(node, `prompt_${i}`);
        if (!w) continue;
        w.hidden = i > count;
      }

      node.computeSize?.();
      (node.graph || app.graph)?.setDirtyCanvas?.(true, true);
    }

    function ensureAddButton(node) {
      let btn = findWidget(node, "Add prompt window");
      if (btn) {
        moveWidgetToEnd(node, btn);
        return;
      }

      btn = node.addWidget(
        "button",
        "Add prompt window",
        null,
        () => {
          const countWidget = findWidget(node, "windows_visible");
          if (!countWidget) return;
          countWidget.value = clampCount(Number(countWidget.value) + 1);
          applyVisibility(node);
          moveWidgetToEnd(node, btn);
        },
        { serialize: false }
      );

      moveWidgetToEnd(node, btn);
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      this.serialize_widgets = true;
      ensureAddButton(this);
      applyVisibility(this);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      this.serialize_widgets = true;
      ensureAddButton(this);
      applyVisibility(this);
      return r;
    };
  },
});
