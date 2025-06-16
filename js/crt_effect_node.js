import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.CRTEffectNode",
    async nodeCreated(node) {
        if (node.comfyClass === "CRTEffectNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});