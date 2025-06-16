import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.PixelArtPostProcessNode",
    async nodeCreated(node) {
        if (node.comfyClass === "PixelArtPostProcessNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});