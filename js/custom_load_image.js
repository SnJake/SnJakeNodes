import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.LoadImageFromPath",
    async nodeCreated(node) {
        if (node.comfyClass === "LoadImageFromPath") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});