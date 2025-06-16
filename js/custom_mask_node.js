import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.CustomMaskNodeStyle",
    async nodeCreated(node) {
        if (node.comfyClass === "CustomMaskNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});