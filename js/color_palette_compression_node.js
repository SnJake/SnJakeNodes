import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.ColorPaletteCompressionNode",
    async nodeCreated(node) {
        if (node.comfyClass === "ColorPaletteCompressionNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});