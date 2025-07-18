import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.TeleportNodes.UI",
    async nodeCreated(node) {
        // --- Логика для окрашивания узлов ---
        if (node.comfyClass === "SnJake_TeleportSet" || node.comfyClass === "SnJake_TeleportGet") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }

        // --- Логика для динамического обновления списка в Get-узле ---
        if (node.comfyClass === "SnJake_TeleportGet") {
            const constantWidget = node.widgets.find(w => w.name === "constant");
            if (constantWidget) {
                // Переопределяем функцию получения значений для выпадающего списка
                constantWidget.options.values = () => {
                    const constants = app.graph._nodes
                        // Находим все узлы TeleportSet в текущем графе
                        .filter(n => n.type === "SnJake_TeleportSet" && n.widgets_values)
                        // Извлекаем значение из их виджета 'constant'
                        // Индекс [1] предполагает, что 'constant' - второй виджет после 'signal'
                        .map(n => n.widgets_values[1])
                        // Фильтруем пустые или неопределенные значения
                        .filter(Boolean);

                    // Возвращаем уникальный, отсортированный список
                    return [...new Set(constants)].sort();
                };
            }
        }
    },
    // Вызывается при изменении графа для обновления виджетов
    onGraphChanged() {
        for (const node of app.graph._nodes) {
            if (node.comfyClass === "SnJake_TeleportGet") {
                const widget = node.widgets.find(w => w.name === "constant");
                if (widget) {
                    // Принудительно обновляем виджет, чтобы он перерисовал список
                    node.setDirtyCanvas(true, true);
                }
            }
        }
    }
});
