import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.PixelArtNode",
    async nodeCreated(node) {
        if (node.comfyClass === "PixelArtNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});