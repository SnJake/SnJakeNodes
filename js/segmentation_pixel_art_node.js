import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.SegmentationPixelArtNode",
    async nodeCreated(node) {
        if (node.comfyClass === "SegmentationPixelArtNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});