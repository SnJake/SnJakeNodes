import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.SaveImageToPath",
    async nodeCreated(node) {
        if (node.comfyClass === "SaveImageToPath") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});