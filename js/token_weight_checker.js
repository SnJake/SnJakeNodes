import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.TokenWeightChecker",
    async nodeCreated(node) {
        if (node.comfyClass === "TokenWeightChecker") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});
