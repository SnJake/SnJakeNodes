import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.SnJake_TeleportSet",
    async nodeCreated(node) {
        if (node.comfyClass === "SnJake_TeleportSet") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});


app.registerExtension({
    name: "SnJake.SnJake_TeleportGet",
    async nodeCreated(node) {
        if (node.comfyClass === "SnJake_TeleportGet") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});
