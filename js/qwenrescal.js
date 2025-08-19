import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.QwenImageResolutionCalc",
    async nodeCreated(node) {
        if (node.comfyClass === "QwenImageResolutionCalc") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});
