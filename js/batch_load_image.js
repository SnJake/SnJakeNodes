import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.BatchLoadImages",
    async nodeCreated(node) {
        if (node.comfyClass === "BatchLoadImages") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});