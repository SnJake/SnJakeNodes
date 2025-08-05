import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.SnJakeTextConcatenate",
    async nodeCreated(node) {
        if (node.comfyClass === "SnJakeTextConcatenate") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});

app.registerExtension({
    name: "SnJake.SnJakeMultilineText",
    async nodeCreated(node) {
        if (node.comfyClass === "SnJakeMultilineText") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});
