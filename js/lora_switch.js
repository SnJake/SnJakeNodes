// lora_switch.js
import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "SnJake.LoraSwitchDynamic.Handler",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "LoraSwitchDynamic") return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      origOnNodeCreated?.apply(this, arguments);

      this.color = "#2e2e36";
      this.bgcolor = "#41414a";

      // Гарантируем сохранение значений виджетов
      this.serialize_widgets = true;

      // Автообновление входов при создании
      queueMicrotask(() => this.updateInputs());

      // Подписка на изменение pairs
      const pairsWidget = this.widgets?.find((w) => w.name === "pairs");
      if (pairsWidget && !pairsWidget.__ls_bound) {
        pairsWidget.__ls_bound = true;
        const origCallback = pairsWidget.callback;
        pairsWidget.callback = (v) => {
          origCallback?.(v);
          this.updateInputs();
          app.graph.setDirtyCanvas(true, true);
        };
      }
    };

    nodeType.prototype.onConfigure = function (info) {
      // Восстановление динамических входов после загрузки графа
      const res = origOnConfigure?.apply(this, arguments);
      // Ждём, пока Comfy восстановит виджеты/ссылки
      queueMicrotask(() => this.updateInputs());
      return res;
    };

    nodeType.prototype.updateInputs = function () {
      const pairsWidget = this.widgets?.find((w) => w.name === "pairs");
      const targetPairs = Math.max(1, Math.min(99, pairsWidget ? Number(pairsWidget.value) : 6));

      // Считаем текущие model_/clip_ входы
      const current = (this.inputs || []).filter(
        (i) => i?.name?.startsWith("model_") || i?.name?.startsWith("clip_")
      );
      const currentPairs = Math.floor(current.length / 2);

      if (targetPairs === currentPairs) return;

      // Добавление
      if (targetPairs > currentPairs) {
        for (let i = currentPairs + 1; i <= targetPairs; i++) {
          // Сохраняем порядок: сначала model_i, затем clip_i — стабильные slot-индексы
          this.addInput(`model_${i}`, "MODEL");
          this.addInput(`clip_${i}`, "CLIP");
        }
      } else {
        // Удаление с конца, чтобы не ломать индексацию уже существующих линков
        for (let i = currentPairs; i > targetPairs; i--) {
          const clipIdx = this.inputs.findIndex((inp) => inp.name === `clip_${i}`);
          if (clipIdx !== -1) this.removeInput(clipIdx);
          const modelIdx = this.inputs.findIndex((inp) => inp.name === `model_${i}`);
          if (modelIdx !== -1) this.removeInput(modelIdx);
        }
      }

      this.computeSize();
    };
  },
});
