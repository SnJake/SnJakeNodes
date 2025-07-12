import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.ResizeAllMasks",
    async nodeCreated(node) {
        if (node.comfyClass === "ResizeAllMasks") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});

app.registerExtension({
    name: "SnJake.BlurImageByMasks",
    async nodeCreated(node) {
        if (node.comfyClass === "BlurImageByMasks") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});

app.registerExtension({
    name: "SnJake.OverlayImageByMasks",
    async nodeCreated(node) {
        if (node.comfyClass === "OverlayImageByMasks") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});