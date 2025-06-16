import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.RegionMergingNode",
    async nodeCreated(node) {
        if (node.comfyClass === "RegionMergingNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});