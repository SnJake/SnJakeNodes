import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.ImageAdjustmentNode",
    async nodeCreated(node) {
        if (node.comfyClass === "ImageAdjustmentNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});