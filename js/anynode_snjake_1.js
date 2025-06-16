import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.LocalOpenAICompatibleNode",
    async nodeCreated(node) {
        if (node.comfyClass === "LocalOpenAICompatibleNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});