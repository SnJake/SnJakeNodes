import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.ExpandImageRight",
    async nodeCreated(node) {
        if (node.comfyClass === "ExpandImageRight") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});