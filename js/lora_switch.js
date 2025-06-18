import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "LoraSwitchDynamic.DynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LoraSwitchDynamic") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
            // Вызывается при создании ноды
            nodeType.prototype.onNodeCreated = function () {
                // Добавляем кнопку "Update Inputs"
                this.addWidget("button", "Update Inputs", null, () => {
                    this.updateInputs();
                });

                // Первоначальное обновление при создании
                this.updateInputs();
            };

            // Логика обновления входов
            nodeType.prototype.updateInputs = function() {
                const pairsWidget = this.widgets.find(w => w.name === "pairs");
                if (!pairsWidget) return;

                const targetPairs = pairsWidget.value;
                const currentInputs = this.inputs ? this.inputs.length : 0;
                const currentPairs = currentInputs / 2;

                if (targetPairs === currentPairs) {
                    return; // Количество пар не изменилось
                }

                // Добавление новых пар
                if (targetPairs > currentPairs) {
                    for (let i = currentPairs + 1; i <= targetPairs; i++) {
                        this.addInput(`model_${i}`, "MODEL");
                        this.addInput(`clip_${i}`, "CLIP");
                    }
                }
                // Удаление лишних пар
                else if (targetPairs < currentPairs) {
                    // Удаляем входы парами, начиная с конца
                    for (let i = currentPairs; i > targetPairs; i--) {
                        this.removeInput(this.inputs.length - 1); // remove clip
                        this.removeInput(this.inputs.length - 1); // remove model
                    }
                }
                
                // Обновляем размер ноды, чтобы все поместилось
                this.computeSize();
                app.graph.setDirtyCanvas(true, true);
            };
        }
    },
});




app.registerExtension({
    name: "SnJake.LoraBlocker",
    async nodeCreated(node) {
        if (node.comfyClass === "LoraBlocker") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});
