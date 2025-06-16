import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.ScanImageFolder2",
    async nodeCreated(node) {
        if (node.comfyClass === "ScanImageFolder2") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});