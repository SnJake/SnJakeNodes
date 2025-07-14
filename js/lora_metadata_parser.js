import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.LoraMetadataParser",
    async nodeCreated(node) {
        if (node.comfyClass === "LoraMetadataParser") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});
