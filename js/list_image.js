import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.ScanImageFolder",
    async nodeCreated(node) {
        if (node.comfyClass === "ScanImageFolder") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});