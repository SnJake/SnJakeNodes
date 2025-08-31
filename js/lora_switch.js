import { app } from "/scripts/app.js";

app.registerExtension({
  name: "SnJake.LoraSwitchDynamic.Handler",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "LoraSwitchDynamic") return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      origOnNodeCreated?.apply(this, arguments);

      // Визуальные мелочи
      this.color = "#2e2e36";
      this.bgcolor = "#41414a";

      // Сохраняем состояние виджетов в граф
      this.serialize_widgets = true;

      // Автоинициализация входов после создания
      queueMicrotask(() => this.updateInputs());

      // Автосинхронизация при изменении pairs
      const pairsWidget = this.widgets?.find((w) => w.name === "pairs");
      if (pairsWidget && !pairsWidget.__ls_bound) {
        pairsWidget.__ls_bound = true;
        const orig = pairsWidget.callback;
        pairsWidget.callback = (v) => {
          orig?.(v);
          this.updateInputs();
          app.graph.setDirtyCanvas(true, true);
        };
      }
    };

    nodeType.prototype.onConfigure = function (info) {
      const res = origOnConfigure?.apply(this, arguments);
      // Восстановление входов после загрузки/перезагрузки страницы
      queueMicrotask(() => this.updateInputs());
      return res;
    };

    nodeType.prototype.updateInputs = function () {
      const pairsWidget = this.widgets?.find((w) => w.name === "pairs");
      const targetPairs = Math.max(1, Math.min(99, pairsWidget ? Number(pairsWidget.value) : 6));

      const cur = (this.inputs || []).filter(
        (i) => i?.name?.startsWith("model_") || i?.name?.startsWith("clip_")
      );
      const currentPairs = Math.floor(cur.length / 2);

      if (targetPairs === currentPairs) return;

      if (targetPairs > currentPairs) {
        for (let i = currentPairs + 1; i <= targetPairs; i++) {
          this.addInput(`model_${i}`, "MODEL");
          this.addInput(`clip_${i}`, "CLIP");
        }
      } else {
        for (let i = currentPairs; i > targetPairs; i--) {
          const ci = this.inputs.findIndex((inp) => inp.name === `clip_${i}`);
          if (ci !== -1) this.removeInput(ci);
          const mi = this.inputs.findIndex((inp) => inp.name === `model_${i}`);
          if (mi !== -1) this.removeInput(mi);
        }
      }

      this.computeSize();
    };
  },
});
