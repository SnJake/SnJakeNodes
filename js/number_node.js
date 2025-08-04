import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.SnJakeNumberNode",
    async nodeCreated(node) {
        if (node.comfyClass === "SnJakeNumberNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});
