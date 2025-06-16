import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.RandomFloatNode",
    async nodeCreated(node) {
        if (node.comfyClass === "RandomFloatNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});