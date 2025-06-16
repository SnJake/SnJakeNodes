import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.DateTimeToStringNode",
    async nodeCreated(node) {
        if (node.comfyClass === "DateTimeToStringNode") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});