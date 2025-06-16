import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.ImageResizeNode",
    async nodeCreated(node) {
        if (node.comfyClass === "ImageResizeNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});