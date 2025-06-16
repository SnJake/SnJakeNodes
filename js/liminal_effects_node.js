import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.LiminalEffectsNode",
    async nodeCreated(node) {
        if (node.comfyClass === "LiminalEffectsNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});