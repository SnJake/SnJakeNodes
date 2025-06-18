import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.LoraSwitch6",
    async nodeCreated(node) {
        if (node.comfyClass === "LoraSwitch6") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});
