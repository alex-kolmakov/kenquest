/**
 * KnowledgeMap — interactive prerequisite DAG powered by React Flow + Dagre.
 *
 * Layouts : "layer"  — columns by learning depth (default, great for study order)
 *           "graph"  — Dagre LR DAG layout (great for seeing edge topology)
 * Nodes   : colour-coded by criticality tier, size by transitive fanout
 * Edges   : opacity by strength
 * Controls: tier filter, layout toggle, zoom, minimap, concept detail panel on click
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type NodeMouseHandler,
  MarkerType,
  Panel,
  Handle,
  Position,
} from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../../api/client";
import "@xyflow/react/dist/style.css";

// ── Types ─────────────────────────────────────────────────────────────────────

type Tier = "CORE" | "IMPORTANT" | "STANDARD" | "SUPPLEMENTARY";
type LayoutMode = "layer" | "graph";

// Node type discriminators
const CONCEPT_NODE_TYPE    = "conceptNode";
const LAYER_LABEL_TYPE     = "layerLabel";

// Typed node data accessor — eliminates repeated inline casts
interface NodeData { apiNode: ApiNode; w: number; h: number }
const nd = (n: Node): NodeData => n.data as unknown as NodeData;

interface ApiNode {
  id: string;
  name: string;
  description: string;
  difficulty: number;
  status: string;
  transitive_fanout: number;
  criticality_score: number;
  tier: Tier;
  layer: number;
}

interface ApiEdge {
  source: string;
  target: string;
  strength: number;
  rationale: string;
}

interface GraphData {
  topic_id: string;
  nodes: ApiNode[];
  edges: ApiEdge[];
}

// ── Constants ──────────────────────────────────────────────────────────────────

const TIER_COLOR: Record<Tier, string> = {
  CORE:          "#FFD700",
  IMPORTANT:     "#FF8C00",
  STANDARD:      "#4a90d9",
  SUPPLEMENTARY: "#4a5568",
};

const TIER_BORDER: Record<Tier, string> = {
  CORE:          "#FFF176",
  IMPORTANT:     "#FFAB40",
  STANDARD:      "#82b4e8",
  SUPPLEMENTARY: "#6b7280",
};

const TIER_ORDER: Record<Tier, number> = {
  CORE: 0, IMPORTANT: 1, STANDARD: 2, SUPPLEMENTARY: 3,
};

const STATUS_RING: Record<string, string> = {
  unlocked:  "#34d399",
  completed: "#6ee7b7",
  locked:    "transparent",
};

const NODE_W = 160;
const NODE_H = 48;

// ── Dagre layout ──────────────────────────────────────────────────────────────

function applyDagreLayout(
  nodes: Node[],
  edges: Edge[],
): { nodes: Node[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "LR", ranksep: 80, nodesep: 18, edgesep: 10 });
  g.setDefaultEdgeLabel(() => ({}));

  nodes.forEach((n) => g.setNode(n.id, { width: NODE_W, height: NODE_H }));
  edges.forEach((e) => g.setEdge(e.source, e.target));
  dagre.layout(g);

  return {
    nodes: nodes.map((n) => {
      const { x, y } = g.node(n.id);
      return { ...n, position: { x: x - NODE_W / 2, y: y - NODE_H / 2 } };
    }),
    edges,
  };
}

// ── Layer column layout ────────────────────────────────────────────────────────

const COLUMN_W   = 240;  // horizontal stride per layer
const ROW_GAP    = 10;   // vertical gap between nodes in the same column
const LABEL_H    = 28;   // height reserved for the "Layer N" header label

function applyLayerLayout(
  nodes: Node[],
  edges: Edge[],
): { nodes: Node[]; edges: Edge[] } {
  // Group and sort nodes within each layer
  const byLayer = new Map<number, Node[]>();
  nodes.forEach((n) => {
    const layer = nd(n).apiNode.layer;
    if (!byLayer.has(layer)) byLayer.set(layer, []);
    byLayer.get(layer)!.push(n);
  });

  byLayer.forEach((layerNodes) => {
    layerNodes.sort((a, b) => {
      const ta = TIER_ORDER[nd(a).apiNode.tier];
      const tb = TIER_ORDER[nd(b).apiNode.tier];
      if (ta !== tb) return ta - tb;
      return nd(a).apiNode.name.localeCompare(nd(b).apiNode.name);
    });
  });

  // Position each node
  const positioned = nodes.map((n) => {
    const { apiNode } = nd(n);
    const layer = apiNode.layer;
    const layerNodes = byLayer.get(layer)!;
    const idx = layerNodes.findIndex((ln) => ln.id === n.id);

    // Total column height for vertical centering
    const totalH = layerNodes.reduce((sum, ln) => sum + nd(ln).h + ROW_GAP, 0);

    let yOffset = LABEL_H - totalH / 2;
    for (let i = 0; i < idx; i++) {
      yOffset += nd(layerNodes[i]).h + ROW_GAP;
    }

    const x = layer * COLUMN_W;
    return { ...n, position: { x, y: yOffset } };
  });

  // Build layer label nodes (non-interactive)
  const labelNodes: Node[] = Array.from(byLayer.entries()).map(([layer, layerNodes]) => ({
    id:       `__layer_label_${layer}`,
    type:     LAYER_LABEL_TYPE,
    position: { x: layer * COLUMN_W, y: -60 },
    data:     { layer, count: layerNodes.length },
    selectable: false,
    draggable:  false,
    focusable:  false,
    style:    { pointerEvents: "none" as const },
  }));

  return { nodes: [...labelNodes, ...positioned], edges };
}

// ── Transform API data → React Flow nodes/edges ───────────────────────────────

function buildFlowElements(
  data: GraphData,
  visibleTiers: Set<Tier>,
  layoutMode: LayoutMode,
): { nodes: Node[]; edges: Edge[] } {
  const visibleIds = new Set(
    data.nodes.filter((n) => visibleTiers.has(n.tier)).map((n) => n.id),
  );

  const maxFanout = Math.max(...data.nodes.map((n) => n.transitive_fanout), 1);

  const nodes: Node[] = data.nodes
    .filter((n) => visibleIds.has(n.id))
    .map((n) => {
      const scale = 0.6 + (n.transitive_fanout / maxFanout) ** 0.4 * 0.8;
      const w = Math.round(NODE_W * scale);
      const h = Math.round(NODE_H * Math.max(scale * 0.8, 0.7));
      return {
        id: n.id,
        type: CONCEPT_NODE_TYPE,
        position: { x: 0, y: 0 },
        data: { apiNode: n, w, h },
        style: { width: w, height: h },
      };
    });

  const edges: Edge[] = data.edges
    .filter((e) => visibleIds.has(e.source) && visibleIds.has(e.target))
    .map((e) => ({
      id:     `${e.source}--${e.target}`,
      source: e.source,
      target: e.target,
      markerEnd: { type: MarkerType.ArrowClosed, width: 12, height: 12, color: "#7b7bbb" },
      style: {
        stroke: "#7b7bbb",
        strokeWidth: 1.0 + e.strength * 1.5,
        opacity: 0.35 + e.strength * 0.5,
      },
      data: { rationale: e.rationale },
    }));

  return layoutMode === "layer"
    ? applyLayerLayout(nodes, edges)
    : applyDagreLayout(nodes, edges);
}

// ── Custom node: concept ───────────────────────────────────────────────────────

function ConceptNode({ data }: { data: { apiNode: ApiNode; w: number; h: number } }) {
  const { apiNode: n, w, h } = data;
  const color  = TIER_COLOR[n.tier];
  const border = TIER_BORDER[n.tier];
  const ring   = STATUS_RING[n.status] ?? "transparent";
  const handleStyle = { background: "transparent", border: "none", width: 6, height: 6 };

  return (
    <div
      style={{
        width: w,
        height: h,
        background: `${color}18`,
        border: `1.5px solid ${border}55`,
        borderRadius: 6,
        outline: ring !== "transparent" ? `2px solid ${ring}` : undefined,
        outlineOffset: 2,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "2px 6px",
        cursor: "pointer",
        userSelect: "none",
      }}
    >
      <Handle type="target" position={Position.Left} style={handleStyle} />
      <Handle type="source" position={Position.Right} style={handleStyle} />
      <span
        style={{
          color,
          fontSize: n.tier === "CORE" ? 10.5 : 9.5,
          fontWeight: n.tier === "CORE" ? 700 : 500,
          fontFamily: "monospace",
          textAlign: "center",
          lineHeight: 1.25,
          wordBreak: "break-word",
        }}
      >
        {n.name}
      </span>
      {n.tier === "CORE" || n.tier === "IMPORTANT" ? (
        <span style={{ fontSize: 8, color: `${color}99`, fontFamily: "monospace" }}>
          ↓{n.transitive_fanout}
        </span>
      ) : null}
    </div>
  );
}

// ── Custom node: layer label ───────────────────────────────────────────────────

function LayerLabelNode({ data }: { data: { layer: number; count: number } }) {
  return (
    <div
      style={{
        width: NODE_W,
        textAlign: "center",
        fontFamily: "monospace",
        pointerEvents: "none",
        userSelect: "none",
      }}
    >
      <div style={{ fontSize: 9, color: "#555", letterSpacing: 1 }}>
        LAYER {data.layer}
      </div>
      <div style={{ fontSize: 8, color: "#3a3a4a", marginTop: 2 }}>
        {data.count} concepts
      </div>
      <div style={{ marginTop: 6, borderTop: "1px solid #1e2333", width: "60%", margin: "6px auto 0" }} />
    </div>
  );
}

const nodeTypes = {
  [CONCEPT_NODE_TYPE]: ConceptNode,
  [LAYER_LABEL_TYPE]:  LayerLabelNode,
};

// Module-level so it's not recreated on every render
const miniMapNodeColor = (n: Node) =>
  n.type === LAYER_LABEL_TYPE
    ? "transparent"
    : TIER_COLOR[nd(n).apiNode?.tier ?? "SUPPLEMENTARY"];

// ── Detail panel ──────────────────────────────────────────────────────────────

function DetailPanel({ node, onClose }: { node: ApiNode; onClose: () => void }) {
  const color = TIER_COLOR[node.tier];
  return (
    <div
      style={{
        position: "absolute",
        top: 12,
        right: 12,
        width: 280,
        background: "#12151e",
        border: `1px solid ${color}44`,
        borderRadius: 10,
        padding: "14px 16px",
        zIndex: 100,
        fontFamily: "monospace",
        color: "#ddd",
        boxShadow: "0 4px 24px #00000066",
      }}
    >
      <button
        onClick={onClose}
        style={{ position: "absolute", top: 8, right: 10, background: "none", border: "none", color: "#888", cursor: "pointer", fontSize: 16 }}
      >
        ✕
      </button>
      <div style={{ color, fontWeight: 700, fontSize: 13, marginBottom: 6 }}>{node.name}</div>
      <div style={{ fontSize: 10, color: "#aaa", marginBottom: 10 }}>
        <span style={{ background: `${color}22`, border: `1px solid ${color}44`, borderRadius: 4, padding: "1px 6px", marginRight: 6 }}>
          {node.tier}
        </span>
        <span>Difficulty {node.difficulty}/5</span>
        <span style={{ marginLeft: 8 }}>Layer {node.layer}</span>
        <span style={{ marginLeft: 8 }}>↓{node.transitive_fanout}</span>
      </div>
      {node.description && (
        <p style={{ fontSize: 10.5, lineHeight: 1.55, color: "#bbb" }}>
          {node.description.slice(0, 300)}
          {node.description.length > 300 ? "…" : ""}
        </p>
      )}
    </div>
  );
}

// ── Tier filter pill ──────────────────────────────────────────────────────────

function TierPill({ tier, count, active, onClick }: { tier: Tier; count: number; active: boolean; onClick: () => void }) {
  const color = TIER_COLOR[tier];
  return (
    <button
      onClick={onClick}
      style={{
        background:   active ? `${color}22` : "transparent",
        border:       `1px solid ${active ? color : "#444"}`,
        borderRadius: 20,
        padding:      "3px 10px",
        color:        active ? color : "#666",
        cursor:       "pointer",
        fontSize:     11,
        fontFamily:   "monospace",
        whiteSpace:   "nowrap",
      }}
    >
      {tier} {count}
    </button>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export function KnowledgeMap({ topicId }: { topicId: string }) {
  const ALL_TIERS: Tier[] = ["CORE", "IMPORTANT", "STANDARD", "SUPPLEMENTARY"];
  const [visibleTiers, setVisibleTiers] = useState<Set<Tier>>(
    new Set(["CORE", "IMPORTANT", "STANDARD"]),
  );
  const [layoutMode, setLayoutMode] = useState<LayoutMode>("layer");
  const [selectedNode, setSelectedNode] = useState<ApiNode | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  const { data, isLoading, error } = useQuery<GraphData>({
    queryKey: ["graph", topicId],
    queryFn:  () => apiFetch(`/topics/${topicId}/graph`),
    staleTime: 5 * 60 * 1000,
  });

  useEffect(() => {
    if (!data) return;
    const { nodes: n, edges: e } = buildFlowElements(data, visibleTiers, layoutMode);
    setNodes(n);
    setEdges(e);
  }, [data, visibleTiers, layoutMode]);

  const onNodeClick: NodeMouseHandler = useCallback(
    (_evt, node) => {
      if (node.type === LAYER_LABEL_TYPE) return;
      const apiNode = nd(node).apiNode;
      setSelectedNode((prev) => (prev?.id === apiNode.id ? null : apiNode));
    },
    [],
  );

  const tierCounts = useMemo(() => {
    if (!data) return {} as Record<Tier, number>;
    return data.nodes.reduce(
      (acc, n) => ({ ...acc, [n.tier]: (acc[n.tier] ?? 0) + 1 }),
      {} as Record<Tier, number>,
    );
  }, [data]);

  const maxLayer = useMemo(() => {
    if (!data) return 0;
    return Math.max(...data.nodes.map((n) => n.layer));
  }, [data]);

  const toggleTier = (tier: Tier) =>
    setVisibleTiers((prev) => {
      const next = new Set(prev);
      next.has(tier) ? next.delete(tier) : next.add(tier);
      return next;
    });

  if (isLoading)
    return <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", background: "#0d1117", color: "#666", fontFamily: "monospace", fontSize: 13 }}>Loading graph…</div>;
  if (error)
    return <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", background: "#0d1117", color: "#f87171", fontFamily: "monospace", fontSize: 13 }}>{(error as Error).message}</div>;

  return (
    <div style={{ width: "100%", height: "100%", background: "#0d1117", position: "relative" }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.08 }}
        minZoom={0.05}
        maxZoom={3}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={28} size={1} color="#1e2333" />
        <Controls style={{ background: "#12151e", border: "1px solid #333" }} />
        <MiniMap
          style={{ background: "#0d1117", border: "1px solid #333" }}
          nodeColor={miniMapNodeColor}
          maskColor="#0d111788"
          zoomable
          pannable
        />

        {/* Filter + layout bar */}
        <Panel position="top-left">
          <div style={{ display: "flex", gap: 6, padding: "6px 10px", background: "#12151ecc", border: "1px solid #333", borderRadius: 8, backdropFilter: "blur(6px)", alignItems: "center" }}>
            <span style={{ color: "#555", fontSize: 10, fontFamily: "monospace" }}>TIER</span>
            {ALL_TIERS.map((t) => (
              <TierPill key={t} tier={t} count={tierCounts[t] ?? 0} active={visibleTiers.has(t)} onClick={() => toggleTier(t)} />
            ))}
            <div style={{ width: 1, height: 18, background: "#333", margin: "0 4px" }} />
            <button
              onClick={() => setLayoutMode((m) => m === "layer" ? "graph" : "layer")}
              style={{
                background:   "#1e2333",
                border:       "1px solid #444",
                borderRadius: 6,
                padding:      "3px 10px",
                color:        "#aaa",
                cursor:       "pointer",
                fontSize:     10,
                fontFamily:   "monospace",
                whiteSpace:   "nowrap",
              }}
            >
              {layoutMode === "layer" ? "⬜ GRAPH" : "▤ LAYERS"}
            </button>
          </div>
        </Panel>

        {/* Stats bar */}
        <Panel position="top-right">
          <div style={{ padding: "5px 12px", background: "#12151ecc", border: "1px solid #333", borderRadius: 8, backdropFilter: "blur(6px)", color: "#555", fontSize: 10, fontFamily: "monospace" }}>
            {layoutMode === "layer"
              ? `${maxLayer + 1} layers · ${data?.nodes.length ?? 0} concepts · ${data?.edges.length ?? 0} edges`
              : `${data?.nodes.length ?? 0} concepts · ${data?.edges.length ?? 0} edges`
            }
          </div>
        </Panel>
      </ReactFlow>

      {selectedNode && (
        <DetailPanel node={selectedNode} onClose={() => setSelectedNode(null)} />
      )}
    </div>
  );
}
