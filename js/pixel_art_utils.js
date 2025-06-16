import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.ExtractPaletteNode",
    async nodeCreated(node) {
        if (node.comfyClass === "ExtractPaletteNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});


app.registerExtension({
    name: "SnJake.ApplyPaletteNode",
    async nodeCreated(node) {
        if (node.comfyClass === "ApplyPaletteNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});


app.registerExtension({
    name: "SnJake.ReplacePaletteColorsNode",
    async nodeCreated(node) {
        if (node.comfyClass === "ReplacePaletteColorsNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});