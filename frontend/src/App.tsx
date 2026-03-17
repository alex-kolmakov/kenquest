import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { KnowledgeMap } from "./components/KnowledgeMap/KnowledgeMap";
import { apiFetch } from "./api/client";

interface Topic {
  id: string;
  name: string;
  status: string;
  concept_count: number;
}

function App() {
  const [selectedTopic, setSelectedTopic] = useState<string>("ocean-conservation");

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: () => apiFetch<{ model: string }>("/health"),
    retry: false,
  });

  const { data: topics } = useQuery<Topic[]>({
    queryKey: ["topics"],
    queryFn: () => apiFetch<Topic[]>("/topics"),
    retry: false,
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "#0d1117", color: "#ccc", fontFamily: "monospace" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 16, padding: "10px 20px", borderBottom: "1px solid #1e2333", flexShrink: 0 }}>
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#eee" }}>
          Ken<span style={{ color: "#818cf8" }}>Quest</span>
        </h1>

        {topics && topics.length > 0 && (
          <select
            value={selectedTopic}
            onChange={(e) => setSelectedTopic(e.target.value)}
            style={{ background: "#12151e", border: "1px solid #333", borderRadius: 6, color: "#ccc", padding: "4px 10px", fontSize: 12, fontFamily: "monospace", cursor: "pointer" }}
          >
            {topics.map((t) => (
              <option key={t.id} value={t.id}>{t.name} ({t.concept_count} concepts)</option>
            ))}
          </select>
        )}

        <div style={{ marginLeft: "auto", fontSize: 10, color: "#444" }}>
          {health
            ? <span style={{ color: "#34d399" }}>● API online · {health.model}</span>
            : <span style={{ color: "#f87171" }}>● API unreachable</span>
          }
        </div>
      </div>

      {/* Graph canvas */}
      <div style={{ flex: 1, overflow: "hidden" }}>
        <KnowledgeMap topicId={selectedTopic} />
      </div>
    </div>
  );
}

export default App;
