import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.VLMApiNode",
    async nodeCreated(node) {
        if (node.comfyClass === "VLMApiNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});