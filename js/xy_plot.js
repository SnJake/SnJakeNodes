import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.XYPlotAdvanced",
    async nodeCreated(node) {
        if (node.comfyClass === "XYPlotAdvanced") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});