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

/**
 * Обновляет все Get-узлы в графе, чтобы их выпадающие списки были актуальны.
 */
function updateAllGetNodes() {
    for (const node of app.graph._nodes) {
        if (node.type === "SnJake_TeleportGet") {
            node.widgets[0].options.values = findAvailableConstants();
            node.onConstantChange(); // Обновляем состояние узла
        }
    }
}

app.registerExtension({
    name: "SnJake.TeleportNodes.Virtual.Final",
    registerCustomNodes() {
        // --- Класс для узла-ОТПРАВИТЕЛЯ (Set) ---
        class SnJakeTeleportSetNode extends LiteGraph.LGraphNode {
            constructor(title) {
                super(title);
                this.isVirtualNode = true; // Указывает, что узел существует только на фронте
                this.addInput("signal", "*");
                this.addOutput("signal_passthrough", "*");
                this.addWidget("text", "constant", "default_pipe", () => updateAllGetNodes());
            }

            // Когда к нашему входу что-то подключают, обновляем все Get-узлы
            onConnectionsChange(type, index, connected, link_info) {
                if (type === LiteGraph.INPUT && connected) {
                    updateAllGetNodes();
                }
            }
        }

        // --- Класс для узла-ПОЛУЧАТЕЛЯ (Get) ---
        class SnJakeTeleportGetNode extends LiteGraph.LGraphNode {
            constructor(title) {
                super(title);
                this.isVirtualNode = true;
                this.addOutput("signal", "*");
                this.addWidget("combo", "constant", "default_pipe", this.onConstantChange.bind(this), {
                    values: findAvailableConstants
                });
            }

            // Когда меняется выбранный канал, обновляем тип выхода
            onConstantChange() {
                const setter = this.findSetter();
                const output = this.outputs[0];
                if (setter && setter.inputs[0].type !== "*") {
                    const newType = setter.inputs[0].type;
                    if (output.type !== newType) {
                        output.type = newType;
                        output.name = newType;
                    }
                } else {
                    if (output.type !== "*") {
                        output.type = "*";
                        output.name = "signal";
                    }
                }
            }

            // Находит соответствующий Set-узел
            findSetter() {
                const constantName = this.widgets[0].value;
                return app.graph._nodes.find(
                    (otherNode) => otherNode.type === "SnJake_TeleportSet" && otherNode.widgets[0].value === constantName
                );
            }

            // ГЛАВНЫЙ МЕХАНИЗМ: перехватываем запрос на данные
            getInputLink(slot) {
                const setter = this.findSetter();
                // Если найден Set-узел и к его первому входу что-то подключено...
                if (setter && setter.inputs[0]?.link != null) {
                    // ...возвращаем информацию о соединении ИСТОЧНИКА Set-узла.
                    return app.graph.links[setter.inputs[0].link];
                }
                return null;
            }
        }

        // Регистрируем наши JS-классы, связывая их с именами из Python
        LiteGraph.registerNodeType("SnJake_TeleportSet", SnJakeTeleportSetNode);
        LiteGraph.registerNodeType("SnJake_TeleportGet", SnJakeTeleportGetNode);
    },
    
    // Окрашивание (остается для эстетики)
    async nodeCreated(node) {
        if (node.constructor.name === "SnJakeTeleportSetNode" || node.constructor.name === "SnJakeTeleportGetNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});
