import { app } from "../../../scripts/app.js";

function updateAllGetNodes() {
    for (const node of app.graph._nodes) {
        if (node.comfyClass === "SnJake_TeleportGet") {
            const widget = node.widgets.find(w => w.name === "constant");
            if (widget) {
                node.setDirtyCanvas(true, true);
            }
        }
    }
}

app.registerExtension({
    name: "SnJake.TeleportNodes.UI.v4", // Обновил версию для отладки
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // --- Логика для узла Set ---
        if (nodeData.name === "SnJake_TeleportSet") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                const widget = this.widgets.find(w => w.name === "constant");
                if (widget) {
                    const originalCallback = widget.callback;
                    widget.callback = (value) => {
                        originalCallback?.(value);
                        updateAllGetNodes();
                    };
                }
            };
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
                const widget = this.widgets.find(w => w.name === "constant");
                if (widget) {
                    // Теперь мы не меняем тип виджета, а только поставляем ему значения
                    widget.options = widget.options || {};
                    widget.options.values = () => {
                        const constants = app.graph._nodes
                            .filter(n => n.type === "SnJake_TeleportSet")
                            .map(n => n.widgets?.find(w => w.name === "constant")?.value)
                            .filter(Boolean);

                        if (constants.length === 0) {
                            return ["(no channels found)"];
                        }
                        return [...new Set(constants)].sort();
                    };
                }
            };
        }
    },

    async nodeCreated(node) {
        if (node.comfyClass === "SnJake_TeleportSet" || node.comfyClass === "SnJake_TeleportGet") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    },
});
