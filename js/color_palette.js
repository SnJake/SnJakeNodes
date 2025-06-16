import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.ColorPaletteImageNode",
    async nodeCreated(node) {
        if (node.comfyClass === "ColorPaletteImageNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});