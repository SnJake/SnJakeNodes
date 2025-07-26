import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.VAEEncodeWithPrecision",
    async nodeCreated(node) {
        if (node.comfyClass === "VAEEncodeWithPrecision") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});

app.registerExtension({
    name: "SnJake.VAEDecodeWithPrecision",
    async nodeCreated(node) {
        if (node.comfyClass === "VAEDecodeWithPrecision") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});
