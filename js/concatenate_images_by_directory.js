import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.ConcatenateImagesByDirectory",
    async nodeCreated(node) {
        if (node.comfyClass === "ConcatenateImagesByDirectory") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});