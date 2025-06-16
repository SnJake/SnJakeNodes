import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.OpenAICompatibleNode",
    async nodeCreated(node) {
        if (node.comfyClass === "OpenAICompatibleNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});