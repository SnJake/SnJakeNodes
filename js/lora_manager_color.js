import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "SnJake.LoRAManagerWithPreview.Color",
  nodeCreated(node) {
    if (node.comfyClass === "LoRAManagerWithPreview") {
      node.color   = "#2e2e36";
      node.bgcolor = "#41414a";
      node.boxcolor = "#555A"; // опционально
      node.setDirtyCanvas(true, true);
    }
  }
});
