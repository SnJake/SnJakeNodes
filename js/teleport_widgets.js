// /ComfyUI/custom_nodes/snjake_teleport_nodes/js/teleport_widgets.js

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Функция для обновления выпадающего списка в Get-нодах
async function updateGetNodeLists() {
    try {
        const response = await api.fetchApi("/snjake/get_teleport_constants");
        const constants = await response.json();
        
        const getNodes = app.graph._nodes.filter(node => node.type === "SnJake_TeleportGet");
        
        getNodes.forEach(node => {
            const widget = node.widgets.find(w => w.name === "constant");
            if (widget) {
                const currentValue = widget.value;
                widget.options.values = constants;
                if (constants.includes(currentValue)) {
                    widget.value = currentValue;
                } else if (constants.length > 0) {
                    // Если текущего значения больше нет, выбираем первое в списке
                    widget.value = constants[0];
                }
            }
        });
    } catch (error) {
        console.error("Failed to update Teleport Get lists:", error);
    }
}


app.registerExtension({
    name: "SnJake.TeleportWidgets",
    async setup() {
        updateGetNodeLists();
    },
    
    nodeCreated(node) {
        // --- Логика для ноды Set ---
        if (node.type === "SnJake_TeleportSet") {
            // -- ДОБАВЛЕН КОД ДЛЯ ЦВЕТА --
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
            // -----------------------------

            const widget = node.widgets.find(w => w.name === "constant");
            if (widget) {
                const originalCallback = widget.callback;
                widget.callback = async (value) => {
                    if (originalCallback) {
                        originalCallback.call(widget, value);
                    }
                    if (value && value.trim()) {
                        try {
                            await api.fetchApi("/snjake/add_teleport_constant", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ constant: value.trim() }),
                            });
                            await updateGetNodeLists();
                        } catch (error) {
                            console.error("Failed to add Teleport constant:", error);
                        }
                    }
                };
            }
        }

        // --- Логика для ноды Get ---
        if (node.type === "SnJake_TeleportGet") {
            // -- ДОБАВЛЕН КОД ДЛЯ ЦВЕТА --
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
            // -----------------------------

            setTimeout(updateGetNodeLists, 100);
        }
    }
});
