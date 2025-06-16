import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.PixelArtRestorationNode",
    async nodeCreated(node) {
        if (node.comfyClass === "PixelArtRestorationNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});