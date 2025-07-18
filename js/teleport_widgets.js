import { app } from "../../../scripts/app.js";

/**
 * Функция для принудительного обновления всех узлов TeleportGet в графе.
 * Она находит каждый узел Get, находит его виджет 'constant' и помечает узел
 * как "грязный", чтобы UI перерисовал его, заново вызвав функцию-генератор списка.
 */
function updateAllGetNodes() {
    for (const node of app.graph._nodes) {
        if (node.comfyClass === "SnJake_TeleportGet") {
            const widget = node.widgets.find(w => w.name === "constant");
            if (widget) {
                // Принудительная перерисовка заставит виджет-комбобокс
                // заново запросить список значений через свою функцию .options.values
                node.setDirtyCanvas(true, true);
            }
        }
    }
}

app.registerExtension({
    name: "SnJake.TeleportNodes.UI.v2",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // --- Логика для узла Set ---
        if (nodeData.name === "SnJake_TeleportSet") {
            // Перехватываем метод onNodeCreated для добавления логики к виджету
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const constantWidget = this.widgets.find(w => w.name === "constant");
                if (constantWidget) {
                    // Сохраняем оригинальный callback, если он есть
                    const originalCallback = constantWidget.callback;
                    // Устанавливаем наш собственный callback
                    constantWidget.callback = (value) => {
                        // Вызываем оригинальный callback
                        originalCallback?.(value);
                        // Запускаем обновление всех Get-узлов
                        updateAllGetNodes();
                    };
                }
            };

            // Убеждаемся, что Get-узлы обновятся при удалении Set-узла
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                onRemoved?.apply(this, arguments);
                updateAllGetNodes();
            };
        }

        // --- Логика для узла Get ---
        if (nodeData.name === "SnJake_TeleportGet") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const constantWidget = this.widgets.find(w => w.name === "constant");
                if (constantWidget) {
                    // Переопределяем функцию, которая поставляет значения для выпадающего списка.
                    // Эта функция будет вызываться каждый раз, когда UI перерисовывает виджет.
                    constantWidget.options.values = () => {
                        const constants = app.graph._nodes
                            // Находим все узлы TeleportSet в графе
                            .filter(n => n.type === "SnJake_TeleportSet")
                            // Находим в каждом узле виджет 'constant'
                            .map(n => n.widgets?.find(w => w.name === "constant")?.value)
                            // Отбрасываем пустые или неопределенные значения
                            .filter(Boolean);

                        // Возвращаем уникальный, отсортированный список
                        return [...new Set(constants)].sort();
                    };
                }
            };
        }
    },

    async nodeCreated(node) {
        // --- Логика для окрашивания узлов (остается без изменений) ---
        if (node.comfyClass === "SnJake_TeleportSet" || node.comfyClass === "SnJake_TeleportGet") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    },
});
