// /ComfyUI/custom_nodes/snjake_nodes/js/snjake_teleport_ui.js

import { app } from "../../../scripts/app.js";

/**
 * Находит все уникальные имена каналов, определенные в Set-узлах.
 * @returns {string[]} Отсортированный массив имен каналов.
 */
function findAvailableConstants() {
    const constants = app.graph._nodes
        .filter(n => n.type === "SnJake_TeleportSet")
        .map(n => n.widgets?.find(w => w.name === "constant")?.value)
        .filter(Boolean);
    const uniqueConstants = [...new Set(constants)].sort();
    return uniqueConstants.length > 0 ? uniqueConstants : ["(канал не найден)"];
}

app.registerExtension({
    name: "SnJake.TeleportNodes.Rewire",
    registerCustomNodes() {
        
        // --- Класс для узла-ПОЛУЧАТЕЛЯ (Get) ---
        class SnJakeTeleportGetNode extends LiteGraph.LGraphNode {
            constructor(title = "Teleport Get (Receiver)") {
                super(title);
                this.properties = {};
                this.addOutput("signal", "*");
                this.addWidget("combo", "constant", "default_pipe", this.onConstantChange.bind(this), {
                    values: findAvailableConstants,
                    serialize: false // Не сохранять этот виджет в JSON графа
                });
            }

            // Находит соответствующий Set-узел
            findSetter() {
                const constantName = this.widgets[0].value;
                return app.graph._nodes.find(
                    (otherNode) => otherNode.type === "SnJake_TeleportSet" && otherNode.widgets[0].value === constantName
                );
            }

            // Когда меняется выбранный канал
            onConstantChange(value, widget, node) {
                const setter = this.findSetter();
                const output = this.outputs[0];
                const newType = (setter && setter.inputs[0].type !== "*") ? setter.inputs[0].type : "*";

                if (output.type !== newType) {
                    // Если тип выхода изменился, отсоединяем все существующие провода
                    if (output.links?.length) {
                        for (const linkId of [...output.links]) {
                            app.graph.removeLink(linkId);
                        }
                    }
                    output.type = newType;
                    output.name = newType === "*" ? "signal" : newType;
                }
            }
            
            // ГЛАВНЫЙ МЕХАНИЗМ: ПЕРЕСТРОЙКА ГРАФА
            onConnectOutput(outputIndex, inputType, inputSlot, targetNode, targetSlotIndex) {
                const setter = this.findSetter();
                // Находим узел-источник, подключенный к нашему Set-узлу
                if (setter && setter.inputs[0]?.link != null) {
                    const sourceLink = app.graph.links[setter.inputs[0].link];
                    if (sourceLink) {
                        const sourceNode = app.graph.getNodeById(sourceLink.origin_id);
                        // Программно создаем ПРЯМОЕ соединение от источника к цели
                        sourceNode.connect(sourceLink.origin_slot, targetNode, targetSlotIndex);
                        // Запрещаем создание "неправильного" соединения от нашего Get-узла
                        return false; 
                    }
                }
                // Если Set-узел не найден, ничего не делаем
                return true; 
            }
        }

        // --- Класс для узла-ОТПРАВИТЕЛЯ (Set) ---
        class SnJakeTeleportSetNode extends LiteGraph.LGraphNode {
            constructor(title = "Teleport Set (Sender)") {
                super(title);
                this.properties = {};
                this.addInput("signal", "*");
                this.addOutput("signal_passthrough", "*");
                this.addWidget("text", "constant", "default_pipe", () => {
                    // При изменении имени канала обновляем списки во всех Get-узлах
                    for (const node of app.graph._nodes) {
                        if (node.type === "SnJake_TeleportGet") {
                            node.widgets[0].options.values = findAvailableConstants();
                        }
                    }
                });
            }
        }

        LiteGraph.registerNodeType("SnJake_TeleportSet", SnJakeTeleportSetNode);
        LiteGraph.registerNodeType("SnJake_TeleportGet", SnJakeTeleportGetNode);
    },
    
    // Окрашивание
    async nodeCreated(node) {
        if (node.constructor.name === "SnJakeTeleportSetNode" || node.constructor.name === "SnJakeTeleportGetNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});
