import { app } from "../../../scripts/app.js";

// --- Логика для динамического Lora Switcher ---
app.registerExtension({
    name: "SnJake.LoraSwitchDynamic.Handler",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        // Проверяем, что это наша нода
        if (nodeData.name === "LoraSwitchDynamic") {

            // 1. Получаем оригинальную функцию onNodeCreated, если она есть
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            // 2. Переопределяем onNodeCreated, чтобы добавить нашу логику
            nodeType.prototype.onNodeCreated = function () {
                // Сначала вызываем оригинальную функцию
                onNodeCreated?.apply(this, arguments);

                // Теперь 'this' — это наш экземпляр ноды на холсте.
                // Здесь мы можем безопасно менять его свойства и добавлять виджеты.

                // Добавляем кнопку "Update Inputs"
                this.addWidget("button", "Update Inputs", null, () => {
                    this.updateInputs();
                });

                // Вызываем обновление входов при первом создании ноды
                this.updateInputs();
            };

            // 3. Добавляем метод updateInputs в "чертеж" (прототип) ноды
            nodeType.prototype.updateInputs = function() {
                const pairsWidget = this.widgets.find(w => w.name === "pairs");
                if (!pairsWidget) return;

                const targetPairs = pairsWidget.value;

                // Считаем только входы для model/clip, игнорируя остальные
                const currentModelClipInputs = this.inputs ? this.inputs.filter(inp => inp.name.startsWith("model_") || inp.name.startsWith("clip_")) : [];
                const currentPairs = currentModelClipInputs.length / 2;

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
                    for (let i = currentPairs; i > targetPairs; i--) {
                        // Безопасное удаление по имени, чтобы избежать ошибок
                        const clipIndex = this.inputs.findIndex(inp => inp.name === `clip_${i}`);
                        if (clipIndex !== -1) this.removeInput(clipIndex);
                        
                        const modelIndex = this.inputs.findIndex(inp => inp.name === `model_${i}`);
                        if (modelIndex !== -1) this.removeInput(modelIndex);
                    }
                }
                
                this.computeSize();
                app.graph.setDirtyCanvas(true, true);
            };
        }
    }
});

