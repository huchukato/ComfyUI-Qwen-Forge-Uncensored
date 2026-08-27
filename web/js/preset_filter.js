import { app } from "/scripts/app.js";

const VISION_NODES = new Set([
    "QwenUncensoredVision",
    "QwenUncensoredVisionAdvanced",
    "QwenUncensoredVisionGGUF",
    "QwenUncensoredVisionGGUFAdvanced",
]);
const filterStates = new WeakMap();

// Full preset list from system_prompts.json — used as fallback when the widget
// values are not yet populated (common inside subgraphs).
const ALL_PRESETS = [
    "🎬 MiniMax H3 NSFW (5s)",
    "🎬 MiniMax H3 NSFW (10s)",
    "🎬 MiniMax H3 NSFW (15s)",
    "🎞️ MiniMax H3 NSFW R2VA (5s)",
    "🎞️ MiniMax H3 NSFW R2VA (10s)",
    "🎞️ MiniMax H3 NSFW R2VA (15s)",
    "🔄 MiniMax H3 NSFW FL2VA (5s)",
    "🔄 MiniMax H3 NSFW FL2VA (10s)",
    "🔄 MiniMax H3 NSFW FL2VA (15s)",
    "🍿 Wan 2.2 NSFW I2V Timeline (5s)",
    "🎥 Wan 2.2 NSFW I2V Scene (5s)",
    "🎬 Wan 2.2 NSFW I2V Timeline (20s)",
    "🎥 Wan 2.2 NSFW T2V Scene (5s)",
    "🎬 Wan 2.2 NSFW T2V Timeline (20s)",
    "📖 Wan 2.2 NSFW T2V Scene (20s)",
    "🎥 LTX 2.3 NSFW I2V",
    "🎥 LTX 2.3 NSFW I2V (10s)",
    "🎥 LTX 2.3 NSFW I2V (20s)",
    "🔀 LTX 2.3 NSFW FL2VA",
    "🔀 LTX 2.3 NSFW FL2VA (10s)",
    "🔀 LTX 2.3 NSFW FL2VA (20s)",
    "🖼️ Tags",
    "🖼️ Simple Description",
    "🖼️ Detailed Description",
    "🖼️ Video Tags",
    "🖼️ Image Analysis",
    "📹 Video Summary",
];

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

    // Use the widget's current values if populated, otherwise fall back to the
    // hardcoded list. Inside subgraphs the widget may be empty at creation time.
    const widgetValues = presetWidget.options?.values;
    const allPresets = (Array.isArray(widgetValues) && widgetValues.length > 0)
        ? [...widgetValues]
        : [...ALL_PRESETS];

    const applyFilter = () => {
        // Re-capture allPresets if they were empty before but are now populated
        let currentAll = filterStates.get(node)?.allPresets ?? allPresets;
        const currentValues = presetWidget.options?.values;
        if (Array.isArray(currentValues) && currentValues.length > currentAll.length) {
            currentAll = [...currentValues];
            filterStates.get(node).allPresets = currentAll;
        }

        const filtered = currentAll.filter((preset) => familyForPreset(preset) === familyWidget.value);
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
    name: "QwenForge.presetFilter",
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
