import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.YoloModelLoader",
    async nodeCreated(node) {
        if (node.comfyClass === "YoloModelLoader") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});

app.registerExtension({
    name: "SnJake.YoloInference",
    async nodeCreated(node) {
        if (node.comfyClass === "YoloInference") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});