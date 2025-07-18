// /ComfyUI/custom_nodes/snjake_nodes/js/snjake_teleport_ui.js

import { app } from "../../../scripts/app.js";

function findAvailableConstants() {
    const constants = app.graph._nodes
        .filter(n => n.type === "SnJake_TeleportSet")
        .map(n => n.widgets?.find(w => w.name === "constant")?.value)
        .filter(Boolean);
    const uniqueConstants = [...new Set(constants)].sort();
    return uniqueConstants.length > 0 ? uniqueConstants : ["(no channels found)"];
}

function updateAllGetNodes() {
    for (const node of app.graph._nodes) {
        if (node.type === "SnJake_TeleportGet") {
            const widget = node.widgets.find(w => w.name === "constant");
            if (widget) {
                // Обновляем список и сообщаем UI, что виджет изменился
                widget.options.values = findAvailableConstants();
                node.setDirtyCanvas(true, true);
            }
        }
    }
}

app.registerExtension({
    name: "SnJake.TeleportNodes.Final",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        // --- Логика для узла GET (Receiver) ---
        if (nodeData.name === "SnJake_TeleportGet") {
            // ФУНКЦИЯ-ПЕРЕХВАТЧИК: Самая важная часть.
            // Когда движок запрашивает данные с этого узла, мы подменяем источник.
            nodeType.prototype.getInputLink = function (slot) {
                const constantName = this.widgets[0].value;
                const setter = app.graph._nodes.find(
                    (otherNode) => otherNode.type === "SnJake_TeleportSet" && otherNode.widgets[0].value === constantName
                );

                if (setter && setter.inputs[0] && setter.inputs[0].link) {
                    const linkId = setter.inputs[0].link;
                    return app.graph.links[linkId]; // Возвращаем соединение от ИСТОЧНИКА Set-узла
                }
                return null;
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const widget = this.widgets.find(w => w.name === "constant");
                if (widget) {
                    // Превращаем текстовый виджет в выпадающий список
                    widget.type = "combo";
                    widget.options = widget.options || {};
                    widget.options.values = findAvailableConstants();
                    
                    // Когда меняется значение, обновляем тип выходного сокета
                    const originalCallback = widget.callback;
                    widget.callback = (value) => {
                        const setter = this.graph._nodes.find(n => n.type === 'SnJake_TeleportSet' && n.widgets[0].value === value);
                        if (setter && setter.inputs[0].type) {
                            this.outputs[0].type = setter.inputs[0].type;
                            this.outputs[0].name = setter.inputs[0].type;
                        }
                        originalCallback?.(value);
                    };
                }
            };
        }

        // --- Логика для узла SET (Sender) ---
        if (nodeData.name === "SnJake_TeleportSet") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                
                const widget = this.widgets.find(w => w.name === "constant");
                if (widget) {
                    const originalCallback = widget.callback;
                    // При изменении имени канала, обновляем все Get-узлы
                    widget.callback = (value) => {
                        originalCallback?.(value);
                        updateAllGetNodes();
                    };
                }
            };
        }
    },

    // Окрашивание узлов
    async nodeCreated(node) {
        if (node.comfyClass === "SnJake_TeleportSet" || node.comfyClass === "SnJake_TeleportGet") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});
