import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.StringReplace",
    async nodeCreated(node) {
        if (node.comfyClass === "StringReplace") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});