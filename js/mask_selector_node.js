import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.ImageMaskSelector",
    async nodeCreated(node) {
        if (node.comfyClass === "ImageMaskSelector") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});