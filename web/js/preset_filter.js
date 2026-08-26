import { app } from "/scripts/app.js";

const VISION_NODES = new Set([
    "QwenUncensoredVision",
    "QwenUncensoredVisionAdvanced",
    "QwenUncensoredVisionGGUF",
    "QwenUncensoredVisionGGUFAdvanced",
]);
const filterStates = new WeakMap();

function familyForPreset(preset) {
    if (preset.includes("MiniMax H3")) return "MiniMax H3";
    if (preset.includes("Wan 2.2")) return "Wan 2.2";
    if (preset.includes("LTX")) return "LTX";
    return "Generic";
}

function configurePresetFilter(node) {
    if (!VISION_NODES.has(node.comfyClass)) return;

    const familyWidget = node.widgets?.find((widget) => widget.name === "preset_family");
    const presetWidget = node.widgets?.find((widget) => widget.name === "preset_prompt");
    if (!familyWidget || !presetWidget) return;

    const existing = filterStates.get(node);
    if (existing?.familyWidget === familyWidget && existing?.presetWidget === presetWidget) {
        existing.applyFilter();
        return;
    }

    const allPresets = existing?.allPresets ?? [...presetWidget.options.values];
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

    filterStates.set(node, { familyWidget, presetWidget, allPresets, applyFilter });
    applyFilter();
}

function walkGraph(graph) {
    for (const node of graph?.nodes ?? []) {
        configurePresetFilter(node);
        if (node.subgraph) walkGraph(node.subgraph);
    }
}

function refreshAllGraphs() {
    requestAnimationFrame(() => walkGraph(app.graph));
}

app.registerExtension({
    name: "QwenUncensored.presetFilter",
    nodeCreated(node) {
        configurePresetFilter(node);
    },
    loadedGraphNode(node) {
        configurePresetFilter(node);
    },
    afterConfigureGraph() {
        walkGraph(app.graph);
    },
    setup() {
        app.canvas?.canvas?.addEventListener("subgraph-opened", refreshAllGraphs);
        app.canvas?.canvas?.addEventListener("subgraph-converted", refreshAllGraphs);
    },
});
