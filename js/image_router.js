import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.ImageRouter",
    async nodeCreated(node) {
        if (node.comfyClass === "ImageRouter") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});