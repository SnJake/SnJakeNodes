import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.StringToNumber",
    async nodeCreated(node) {
        if (node.comfyClass === "StringToNumber") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});