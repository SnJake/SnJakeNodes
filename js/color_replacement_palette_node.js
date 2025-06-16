import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.ColorPaletteNode",
    async nodeCreated(node) {
        if (node.comfyClass === "ColorPaletteNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});