// /ComfyUI/custom_nodes/snjake_nodes/js/snjake_teleport_ui.js

import { app } from "../../../scripts/app.js";

const CONSOLE_LOG_PREFIX = "[SnJake Teleport]";

// Функция для поиска всех доступных каналов от Set-узлов
function findAvailableConstants() {
    const constants = app.graph._nodes
        .filter(n => n.type === "SnJake_TeleportSet")
        .map(n => n.widgets?.find(w => w.name === "constant")?.value)
        .filter(Boolean);
    const uniqueConstants = [...new Set(constants)].sort();
    return uniqueConstants.length > 0 ? uniqueConstants : ["(no channels found)"];
}

app.registerExtension({
    name: "SnJake.TeleportNodes.Virtual",
    registerCustomNodes() {
        // --- РЕГИСТРАЦИЯ ВИРТУАЛЬНОГО УЗЛА GET ---
        class SnJake_TeleportGetNode extends LiteGraph.LGraphNode {
            constructor(title) {
                super(title);
                this.isVirtualNode = true; // Ключевой флаг! Узел не сериализуется и не исполняется на бэкенде.
                this.addOutput("signal", "*");

                // Находим виджет, созданный Python, и настраиваем его
                this.onConstructed = () => {
                    const widget = this.widgets.find(w => w.name === "constant");
                    if (widget) {
                        widget.options = widget.options || {};
                        widget.options.values = findAvailableConstants;
                        widget.callback = this.onConstantChange.bind(this);
                    }
                    this.onConstantChange(); // Первичная настройка
                };
            }

            // Вызывается при изменении значения в выпадающем списке
            onConstantChange() {
                const setter = this.findSetter();
                if (setter) {
                    // Обновляем тип выхода, чтобы соответствовать типу входа в Set-узле
                    const inputType = setter.inputs[0].type;
                    this.outputs[0].type = inputType;
                    this.outputs[0].name = inputType;
                } else {
                    this.outputs[0].type = "*";
                    this.outputs[0].name = "signal";
                }
            }

            // Найти соответствующий Set-узел в графе
            findSetter() {
                const constantName = this.widgets[0].value;
                return app.graph._nodes.find(
                    (otherNode) => otherNode.type === "SnJake_TeleportSet" && otherNode.widgets[0].value === constantName
                );
            }

            // САМАЯ ГЛАВНАЯ ЧАСТЬ: Перехват запроса на соединение
            // Движок спрашивает: "Откуда брать данные для твоего выхода?"
            // Мы отвечаем: "Бери их напрямую с того узла, который подключен к нашему Set-узелу"
            getInputLink(slot) {
                const setter = this.findSetter();
                if (setter && setter.inputs[0] && setter.inputs[0].link) {
                    const linkId = setter.inputs[0].link;
                    return app.graph.links[linkId];
                }
                return null;
            }
        }

        // --- РЕГИСТРАЦИЯ ВИРТУАЛЬНОГО УЗЛА SET ---
        class SnJake_TeleportSetNode extends LiteGraph.LGraphNode {
            constructor(title) {
                super(title);
                this.isVirtualNode = true;
                this.addInput("signal", "*");
                this.addOutput("signal_passthrough", "*");

                // Настраиваем виджет
                this.onConstructed = () => {
                    const widget = this.widgets.find(w => w.name === "constant");
                    if (widget) {
                        // При изменении имени канала, заставляем все Get-узлы обновиться
                        widget.callback = () => {
                            for (const node of app.graph._nodes) {
                                if (node.type === "SnJake_TeleportGetNode") {
                                    node.widgets[0].options.values = findAvailableConstants;
                                    node.onConstantChange();
                                }
                            }
                        };
                    }
                };
            }
        }
        
        // Регистрируем наши новые классы в LiteGraph
        LiteGraph.registerNodeType("SnJake_TeleportGet", Object.assign(SnJake_TeleportGetNode, { title: "Teleport Get (Receiver)" }));
        LiteGraph.registerNodeType("SnJake_TeleportSet", Object.assign(SnJake_TeleportSetNode, { title: "Teleport Set (Sender)" }));
    },
    
    // Окрашивание узлов
    async nodeCreated(node) {
        if (node.comfyClass === "SnJake_TeleportSet" || node.comfyClass === "SnJake_TeleportGet") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});
