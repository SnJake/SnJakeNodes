import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.SnJakeAnySwitch",
    async nodeCreated(node) {
        if (node.comfyClass === "SnJakeAnySwitch") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});


app.registerExtension({
    name: "SnJake.SnJakeImageSwitch",
    async nodeCreated(node) {
        if (node.comfyClass === "SnJakeImageSwitch") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});


app.registerExtension({
    name: "SnJake.SnJakeMaskSwitch",
    async nodeCreated(node) {
        if (node.comfyClass === "SnJakeMaskSwitch") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});


app.registerExtension({
    name: "SnJake.SnJakeStringSwitch",
    async nodeCreated(node) {
        if (node.comfyClass === "SnJakeStringSwitch") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});


app.registerExtension({
    name: "SnJake.SnJakeLatentSwitch",
    async nodeCreated(node) {
        if (node.comfyClass === "SnJakeLatentSwitch") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});


app.registerExtension({
    name: "SnJake.SnJakeConditioningSwitch",
    async nodeCreated(node) {
        if (node.comfyClass === "SnJakeConditioningSwitch") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});
