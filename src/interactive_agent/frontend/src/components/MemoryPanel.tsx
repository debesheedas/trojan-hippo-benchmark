import { useEffect, useState } from "react";

interface MemoryState {
  short_term: string[];
  long_term: string[];
}

export default function MemoryPanel() {
  const [memory, setMemory] = useState<MemoryState>({ short_term: [], long_term: [] });
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Establish SSE connection to memory stream
    const evtSource = new EventSource("/memory/stream");
    
    evtSource.onopen = () => {
      console.log("Memory stream connected");
      setIsConnected(true);
    };
    
    evtSource.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        setMemory(data);
      } catch (error) {
        console.error("Error parsing memory update:", error);
      }
    };
    
    evtSource.onerror = (error) => {
      console.error("Memory stream error:", error);
      setIsConnected(false);
    };
    
    // Cleanup on unmount
    return () => {
      console.log("Closing memory stream");
      evtSource.close();
    };
  }, []);

  async function clearMemory() {
    try {
      const response = await fetch("/memory", { method: "DELETE" });
      if (response.ok) {
        console.log("Memory cleared successfully");
      } else {
        console.error("Failed to clear memory");
      }
    } catch (error) {
      console.error("Error clearing memory:", error);
    }
  }

  async function refreshMemory() {
    try {
      const response = await fetch("/memory");
      const data = await response.json();
      setMemory(data);
    } catch (error) {
      console.error("Error refreshing memory:", error);
    }
  }

  return (
    <div className="p-4 border rounded-2xl shadow-sm bg-white w-96">
      <div className="flex items-center justify-between mb-3">
        <h2 className="font-bold text-lg flex items-center gap-2">
          🧠 Agent Memory
          {isConnected && (
            <span className="text-xs text-green-500">●</span>
          )}
        </h2>
        <button 
          onClick={refreshMemory}
          className="text-xs text-blue-600 hover:underline"
          title="Refresh memory"
        >
          ↻
        </button>
      </div>
      
      <div className="space-y-4">
        {/* Short-Term Memory */}
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">
            Short-Term (Recent Messages)
          </h3>
          {memory.short_term.length > 0 ? (
            <ul className="text-xs list-disc ml-4 space-y-1 max-h-32 overflow-y-auto">
              {memory.short_term.map((m, i) => (
                <li key={i} className="text-gray-600">{m}</li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-gray-400 italic">No recent messages</p>
          )}
        </div>

        {/* Long-Term Memory */}
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">
            Long-Term (Persistent Facts)
          </h3>
          {memory.long_term.length > 0 ? (
            <ul className="text-xs list-disc ml-4 space-y-1 max-h-32 overflow-y-auto">
              {memory.long_term.map((m, i) => (
                <li key={i} className="text-gray-600">{m}</li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-gray-400 italic">No stored facts</p>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="mt-4 pt-3 border-t border-gray-200 flex justify-between items-center">
        <span className="text-xs text-gray-500">
          {memory.long_term.length} fact{memory.long_term.length !== 1 ? 's' : ''} stored
        </span>
        <button 
          onClick={clearMemory} 
          className="text-xs text-red-600 hover:underline"
        >
          Clear Memory
        </button>
      </div>
    </div>
  );
}

