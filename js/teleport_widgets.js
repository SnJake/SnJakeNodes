import { app } from "../../../scripts/app.js";

/**
 * Возвращает отсортированный массив уникальных имен каналов из всех Set-нод на графе.
 * @returns {string[]}
 */
const getTeleportConstants = () => {
    const setNodes = app.graph._nodes.filter(n => n.type === "SnJake_TeleportSet");
    // Используем Set для автоматического удаления дубликатов
    const constants = new Set();
    setNodes.forEach(node => {
        const widget = node.widgets.find(w => w.name === "constant");
        if (widget && widget.value.trim()) {
            constants.add(widget.value.trim());
        }
    });
    return Array.from(constants).sort();
};

app.registerExtension({
    name: "SnJake.Teleport",
    
    // Переопределяем поведение LGraphNode, чтобы кастомизировать наши ноды
    registerCustomNodes() {
        
        // --- Кастомизация ноды SET ---
        const TeleportSetNode = LiteGraph.getNodeType("SnJake_TeleportSet");
        if (TeleportSetNode) {
            TeleportSetNode.prototype.onNodeCreated = function() {
                // Устанавливаем цвет при создании ноды
                this.color = "#2e2e36";
                this.bgcolor = "#41414a";
                
                // Находим виджет и добавляем ему callback, чтобы обновлять Get-ноды
                const constantWidget = this.widgets.find(w => w.name === "constant");
                if (constantWidget) {
                    // Сохраняем оригинальный callback, если он есть
                    const originalCallback = constantWidget.callback;
                    constantWidget.callback = (value) => {
                        if (originalCallback) {
                            originalCallback.call(this, value);
                        }
                        // Эта строчка не нужна, так как Get-нода сама обновит свой список при клике
                        // Но если бы нужно было принудительно обновить, мы бы вызывали функцию здесь
                    };
                }
            };
        }

        // --- Кастомизация ноды GET ---
        const TeleportGetNode = LiteGraph.getNodeType("SnJake_TeleportGet");
        if (TeleportGetNode) {
            TeleportGetNode.prototype.onNodeCreated = function() {
                // Устанавливаем цвет при создании ноды
                this.color = "#2e2e36";
                this.bgcolor = "#41414a";
            };

            // Самая важная часть: делаем виджет динамическим
            TeleportGetNode.prototype.onConfigure = function(info) {
                // Эта функция вызывается при создании и конфигурации ноды.
                // Мы находим наш виджет и заменяем его значения функцией.
                const constantWidget = this.widgets.find(w => w.name === "constant");
                if (constantWidget) {
                    // Теперь каждый раз, когда пользователь нажимает на выпадающий список,
                    // будет вызываться функция getTeleportConstants,
                    // которая вернет АКТУАЛЬНЫЙ список каналов.
                    constantWidget.options.values = getTeleportConstants;
                }
            };
        }
    }
});
