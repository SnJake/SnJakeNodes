import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.ColorBalance",
    async nodeCreated(node) {
        if (node.comfyClass === "ColorBalance") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});