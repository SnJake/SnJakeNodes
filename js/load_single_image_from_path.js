import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.LoadSingleImageFromPath",
    async nodeCreated(node) {
        if (node.comfyClass === "LoadSingleImageFromPath") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});