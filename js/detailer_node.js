import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.DetailerForEachMask",
    async nodeCreated(node) {
        if (node.comfyClass === "DetailerForEachMask") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});