import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.RandomIntNode",
    async nodeCreated(node) {
        if (node.comfyClass === "RandomIntNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});