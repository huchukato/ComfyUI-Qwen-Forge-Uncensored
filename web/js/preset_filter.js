import { app } from "/scripts/app.js";

const VISION_NODES = new Set([
    "QwenUncensoredVision",
    "QwenUncensoredVisionAdvanced",
    "QwenUncensoredVisionGGUF",
    "QwenUncensoredVisionGGUFAdvanced",
]);

function familyForPreset(preset) {
    if (preset.includes("MiniMax H3")) return "MiniMax H3";
    if (preset.includes("Wan 2.2")) return "Wan 2.2";
    if (preset.includes("LTX")) return "LTX";
    return "Generic";
}

function configurePresetFilter(node) {
    const familyWidget = node.widgets?.find((widget) => widget.name === "preset_family");
    const presetWidget = node.widgets?.find((widget) => widget.name === "preset_prompt");
    if (!familyWidget || !presetWidget) return;

    const allPresets = [...presetWidget.options.values];
    const applyFilter = () => {
        const filtered = allPresets.filter((preset) => familyForPreset(preset) === familyWidget.value);
        presetWidget.options.values = filtered;
        if (!filtered.includes(presetWidget.value)) presetWidget.value = filtered[0] ?? "";
        node.setDirtyCanvas(true, true);
    };

    const originalCallback = familyWidget.callback;
    familyWidget.callback = function (value) {
        originalCallback?.call(this, value);
        applyFilter();
    };

    const originalConfigure = node.onConfigure;
    node.onConfigure = function (info) {
        originalConfigure?.call(this, info);
        applyFilter();
    };

    applyFilter();
}

app.registerExtension({
    name: "QwenUncensored.presetFilter",
    nodeCreated(node) {
        if (VISION_NODES.has(node.comfyClass)) configurePresetFilter(node);
    },
});
