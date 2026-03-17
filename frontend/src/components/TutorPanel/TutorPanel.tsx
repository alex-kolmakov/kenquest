/**
 * TutorPanel — Side panel with learn/quiz tabs.
 * Placeholder: full implementation in Phase 5.
 */
export function TutorPanel({ conceptId }: { conceptId: string | null }) {
  if (!conceptId) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        Select a concept to start learning
      </div>
    );
  }

  return (
    <div className="p-4">
      <h2 className="text-lg font-semibold text-gray-200">
        Concept: {conceptId}
      </h2>
      <p className="text-gray-500 mt-2">Tutor panel — coming in Phase 5</p>
    </div>
  );
}
