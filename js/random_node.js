import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.SnJakeRandomNumberGenerator",
    async nodeCreated(node) {
        if (node.comfyClass === "SnJakeRandomNumberGenerator") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});
