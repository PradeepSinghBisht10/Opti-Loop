import React, { useState, useEffect, useMemo, useRef } from 'react';
import { RefreshCcw, Map, Zap, Route, ListChecks, FileText, Bot } from 'lucide-react';

// --- Global Data Structure: Weighted Graph Representation ---
// The delivery network representing locations and weighted travel costs.
const DELIVERY_NETWORK = {
    'Warehouse': { 'A': 4, 'B': 2 },
    'A': { 'Warehouse': 4, 'C': 5, 'D': 10 },
    'B': { 'Warehouse': 2, 'C': 8, 'E': 3 },
    'C': { 'A': 5, 'B': 8, 'D': 2, 'F': 6 },
    'D': { 'A': 10, 'C': 2, 'F': 2 },
    'E': { 'B': 3, 'G': 7 },
    'F': { 'C': 6, 'D': 2, 'G': 1 },
    'G': { 'E': 7, 'F': 1, 'Delivery_Hub': 9 },
    'Delivery_Hub': { 'G': 9 }
};

// Node layout for visualization (coordinates based on a 100x100 grid)
const NODE_POSITIONS = {
    'Warehouse': { x: 5, y: 50 },
    'A': { x: 25, y: 30 },
    'B': { x: 25, y: 70 },
    'C': { x: 50, y: 50 },
    'D': { x: 75, y: 20 },
    'E': { x: 50, y: 85 },
    'F': { x: 75, y: 65 },
    'G': { x: 90, y: 70 },
    'Delivery_Hub': { x: 95, y: 50 }
};

const ALL_NODES = Object.keys(DELIVERY_NETWORK);
const INTERMEDIATE_NODES = ALL_NODES.filter(n => n !== 'Warehouse' && n !== 'Delivery_Hub');

// --- API Configuration ---
const API_KEY = ""; // Placeholder for Gemini API Key
const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${API_KEY}`;

// --- API Helper Function with Exponential Backoff ---
async function fetchWithRetry(url, options, maxRetries = 3) {
    let lastError = null;
    for (let i = 0; i < maxRetries; i++) {
        try {
            const response = await fetch(url, options);
            if (response.ok) {
                return response.json();
            }
            // If response is not OK, throw error to trigger retry 
            const errorBody = await response.text();
            lastError = new Error(`API returned status ${response.status}: ${errorBody}`);
            
        } catch (error) {
            lastError = error;
        }

        if (i < maxRetries - 1) {
            const delay = Math.pow(2, i) * 1000; // 1s, 2s, 4s
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
    console.error("Failed to fetch from Gemini API after max retries.", lastError);
    // Throwing the error ensures the calling function handles the failure state
    throw lastError; 
}


// --- DSA Helper: Permutation Generator (For Brute-Force TSP) ---
/**
 * Recursively generates all possible permutations of an array.
 */
function getPermutations(array) {
    if (array.length === 0) return [[]];
    
    const results = [];
    for (let i = 0; i < array.length; i++) {
        const current = array[i];
        const remaining = array.slice(0, i).concat(array.slice(i + 1));
        const innerPermutations = getPermutations(remaining);
        
        for (const perm of innerPermutations) {
            results.push([current].concat(perm));
        }
    }
    return results;
}

/**
 * --- DSA Implementation: Dijkstra's Algorithm ---
 * Calculates the shortest path and distance between two nodes.
 */
function dijkstra(graph, startNode, endNode) {
    if (!startNode || !endNode) return { distance: null, path: null };

    const distances = {};
    const predecessors = {};
    let priorityQueue = []; 

    for (const node of Object.keys(graph)) {
        distances[node] = Infinity;
        predecessors[node] = null;
    }
    distances[startNode] = 0;
    priorityQueue.push([0, startNode]);

    while (priorityQueue.length > 0) {
        // Simple array sort to simulate min-heap
        priorityQueue.sort((a, b) => a[0] - b[0]);
        const [currentDistance, currentNode] = priorityQueue.shift();

        if (currentDistance > distances[currentNode]) continue;
        if (currentNode === endNode) break;

        for (const [neighbor, weight] of Object.entries(graph[currentNode])) {
            const newDistance = currentDistance + weight;

            if (newDistance < distances[neighbor]) {
                distances[neighbor] = newDistance;
                predecessors[neighbor] = currentNode;
                priorityQueue.push([newDistance, neighbor]);
            }
        }
    }

    // Path Reconstruction
    const path = [];
    let current = endNode;
    while (current !== null) {
        path.push(current);
        current = predecessors[current];
    }

    if (path[path.length - 1] === startNode) {
        const shortestPath = path.reverse();
        const shortestDistance = distances[endNode];
        return { distance: shortestDistance, path: shortestPath };
    }

    return { distance: null, path: null };
}

/**
 * --- DSA Implementation: Traveling Salesperson Problem (TSP) Solver ---
 * Uses Brute Force (Permutations) combined with Dijkstra's shortest paths
 * to find the most optimal sequence for visiting all mandatory stops.
 */
function tspSolver(graph, mandatoryStops, startNode, endNode) {
    if (mandatoryStops.length === 0) {
        // If no mandatory stops, run standard Dijkstra
        return dijkstra(graph, startNode, endNode);
    }
    
    // 1. Generate all permutations of mandatory stops
    const stopPermutations = getPermutations(mandatoryStops);
    
    let bestDistance = Infinity;
    let bestRoute = null;

    // 2. Iterate through all possible routes
    for (const permutation of stopPermutations) {
        // Route sequence: Start -> Mandatory Stops (in order) -> End
        const fullRouteSequence = [startNode, ...permutation, endNode];
        let currentTotalDistance = 0;
        let possible = true;
        
        // Accumulate the continuous path segments
        const currentPathSegments = [];

        // 3. Calculate total distance for this sequence
        for (let i = 0; i < fullRouteSequence.length - 1; i++) {
            const currentStart = fullRouteSequence[i];
            const currentEnd = fullRouteSequence[i+1];
            
            // Find the shortest distance between the two sequential stops
            const { distance, path } = dijkstra(graph, currentStart, currentEnd);
            
            if (distance === null || distance === Infinity) {
                possible = false; // Route segment is impossible
                break;
            }
            
            currentTotalDistance += distance;
            currentPathSegments.push(path);
        }

        // 4. Check if this is the best route found
        if (possible && currentTotalDistance < bestDistance) {
            bestDistance = currentTotalDistance;
            
            // Flatten the path segments into a single, continuous array
            const finalPath = [];
            currentPathSegments.forEach((pathSegment, index) => {
                if (index === 0) {
                    // First segment: include all nodes
                    finalPath.push(...pathSegment);
                } else {
                    // Subsequent segments: skip the first node (as it duplicates the last node of the previous segment)
                    finalPath.push(...pathSegment.slice(1));
                }
            });
            bestRoute = finalPath;
        }
    }
    
    return { distance: bestDistance !== Infinity ? bestDistance : null, path: bestRoute };
}

// --- Custom Top-Down 2D Delivery Truck SVG Icon (Red) ---
const DeliveryTruckIcon = ({ x, y, rotation }) => (
    <g 
        // Scale adjusted and rotation applied for directional movement on a 2D map
        transform={`translate(${x}, ${y}) rotate(${rotation}) scale(0.25)`} 
        className="transition-transform duration-100 ease-linear"
    >
        {/* Cargo Box (main body) - Red */}
        <rect x="-6" y="-12" width="12" height="15" fill="#DC2626" stroke="#991B1B" strokeWidth="0.5" rx="1" ry="1" /> 
        
        {/* Cabin (front area) - Darker Red */}
        <rect x="-6" y="-18" width="12" height="6" fill="#B91C1C" stroke="#7F1D1D" strokeWidth="0.5" rx="1" ry="1" />
        
        {/* Windshield/Top of Hood Detail (Front) - Gray/Glass. Gives directional focus. */}
        <rect x="-4" y="-17" width="8" height="3" fill="#60A5FA" /> 

        {/* Wheels (Top view: small circles on the side) */}
        <circle cx="-7" cy="-8" r="1.5" fill="#1F2937" /> 
        <circle cx="7" cy="-8" r="1.5" fill="#1F2937" />  
        <circle cx="-7" cy="1" r="1.5" fill="#1F2937" /> 
        <circle cx="7" cy="1" r="1.5" fill="#1F2937" />  
    </g>
);

// --- Component for Drawing the Network Visualization ---
const NetworkVisualizer = ({ path, startNode, endNode, mandatoryStops, truckCoords, isBlinking }) => {
    const nodes = Object.keys(NODE_POSITIONS);
    
    // --- Pre-calculate unique edges for reliable rendering ---
    const uniqueEdges = useMemo(() => {
        const edges = [];
        const drawnKeys = new Set();
        Object.entries(DELIVERY_NETWORK).forEach(([u, connections]) => {
            Object.entries(connections).forEach(([v, weight]) => {
                const canonicalKey = u < v ? `${u}-${v}` : `${v}-${u}`;
                if (!drawnKeys.has(canonicalKey)) {
                    drawnKeys.add(canonicalKey);
                    edges.push({ u, v, weight });
                }
            });
        });
        return edges;
    }, []); 

    // Function to calculate coordinates for SVG lines
    const getCoords = (nodeName) => NODE_POSITIONS[nodeName];

    // Determine if an edge (connection) is part of the shortest path (including the loop)
    const isPathEdge = (n1, n2) => {
        if (!path || path.length < 2) return false;
        
        // Helper to check sequence n1 -> n2
        const checkSequence = (u, v) => {
            const uIndex = path.indexOf(u);
            const vIndex = path.indexOf(v);
            
            // Standard segment check
            if (uIndex !== -1 && vIndex === uIndex + 1) return true;
            
            // Loop-back check (last node back to first node)
            if (uIndex === path.length - 1 && vIndex === 0) return true;
            
            return false;
        };
        
        // Check both directions (u -> v) and (v -> u)
        return checkSequence(n1, n2) || checkSequence(n2, n1);
    };

    // Calculate rotation for the truck SVG to point towards the direction of travel
    const calculateRotation = (truckX, truckY, nextNode) => {
        if (!nextNode) return 0;
        const nextPos = NODE_POSITIONS[nextNode];
        const dx = nextPos.x - truckX;
        const dy = nextPos.y - truckY;
        return Math.atan2(dy, dx) * (180 / Math.PI) + 90; 
    };

    // Determine the next node in the path for rotation calculation
    const nextNodeName = useMemo(() => {
        if (!truckCoords || !path || path.length < 2) return null;
        
        let closestNode = null;
        let minDistanceSq = Infinity;
        
        for (const node of path) {
            const pos = NODE_POSITIONS[node];
            const distSq = (truckCoords.x - pos.x)**2 + (truckCoords.y - pos.y)**2;
            if (distSq < minDistanceSq) {
                minDistanceSq = distSq;
                closestNode = node;
            }
        }
        
        const currentIndex = path.indexOf(closestNode);
        
        // Use modulo operator to correctly handle the loop-back segment
        return currentIndex !== -1 ? path[(currentIndex + 1) % path.length] : null;

    }, [truckCoords, path]);

    const truckRotation = nextNodeName ? calculateRotation(truckCoords.x, truckCoords.y, nextNodeName) : 0;
    
    // Function to select the correct gradient fill for the node
    const getNodeFill = (node) => {
        if (node === startNode) return 'url(#startGradient)';
        if (node === endNode) return 'url(#endGradient)';
        if (mandatoryStops.includes(node)) return 'url(#mandatoryGradient)';
        if (path && path.includes(node)) return 'url(#pathGradient)';
        return 'url(#defaultGradient)';
    };

    return (
        <div className="w-full h-96 md:h-[500px] border-2 border-slate-700 bg-slate-900 rounded-xl shadow-2xl relative overflow-hidden">
            <svg viewBox="0 0 100 100" className="w-full h-full absolute top-0 left-0">
                <defs>
                    {/* Radial Gradients for Reflective Nodes */}
                    <radialGradient id="startGradient" cx="50%" cy="50%" r="50%" fx="60%" fy="40%">
                        <stop offset="0%" style={{stopColor: "rgb(190, 242, 100)"}} /> 
                        <stop offset="100%" style={{stopColor: "rgb(22, 163, 74)"}} /> 
                    </radialGradient>
                    <radialGradient id="endGradient" cx="50%" cy="50%" r="50%" fx="60%" fy="40%">
                        <stop offset="0%" style={{stopColor: "rgb(252, 165, 165)"}} /> 
                        <stop offset="100%" style={{stopColor: "rgb(185, 28, 28)"}} /> 
                    </radialGradient>
                    <radialGradient id="mandatoryGradient" cx="50%" cy="50%" r="50%" fx="60%" fy="40%">
                        <stop offset="0%" style={{stopColor: "rgb(253, 186, 116)"}} /> 
                        <stop offset="100%" style={{stopColor: "rgb(234, 88, 12)"}} /> 
                    </radialGradient>
                    <radialGradient id="pathGradient" cx="50%" cy="50%" r="50%" fx="60%" fy="40%">
                        <stop offset="0%" style={{stopColor: "rgb(147, 197, 253)"}} /> 
                        <stop offset="100%" style={{stopColor: "rgb(37, 99, 235)"}} /> 
                    </radialGradient>
                    <radialGradient id="defaultGradient" cx="50%" cy="50%" r="50%" fx="60%" fy="40%">
                        <stop offset="0%" style={{stopColor: "rgb(156, 163, 175)"}} /> 
                        <stop offset="100%" style={{stopColor: "rgb(55, 65, 81)"}} /> 
                    </radialGradient>
                </defs>

                {/* 1. Draw all edges (Order: Base Line -> Glow Layer -> Primary Line -> Weights) */}
                {uniqueEdges.map(({ u, v, weight }) => {
                    const uPos = getCoords(u);
                    const vPos = getCoords(v);
                    const isPath = isPathEdge(u, v) || isPathEdge(v, u); 
                    
                    // Base line thickness for static network is now 0.8
                    const baseLineClass = 'stroke-slate-600 stroke-[0.8]'; 
                    
                    // Calculate position offset for text label
                    const dx = uPos.x - vPos.x;
                    const dy = uPos.y - vPos.y;
                    const isVertical = Math.abs(dx) < 1.0; 
                    const isHorizontal = Math.abs(dy) < 1.0; 
                    
                    let textOffsetX = 0;
                    let textOffsetY = -2.5; 

                    if (isVertical) {
                        textOffsetX = 3.5; 
                        textOffsetY = 0; 
                    } else if (isHorizontal) {
                        textOffsetX = 0;
                        textOffsetY = 5;
                    }

                    return (
                        <React.Fragment key={`${u}-${v}`}>
                            {/* 1A. Draw Base/Static Line (Always visible, dark gray) */}
                            <line 
                                x1={uPos.x} y1={uPos.y} 
                                x2={vPos.x} y2={vPos.y} 
                                className={baseLineClass}
                            />
                            
                            {/* 1B. Draw GLOW Layer (Thick, translucent yellow for guaranteed visibility) */}
                            {isPath && (
                                <line 
                                    x1={uPos.x} y1={uPos.y} 
                                    x2={vPos.x} y2={vPos.y} 
                                    stroke="rgba(255, 255, 100, 0.4)" 
                                    strokeWidth="4.0" 
                                    className="transition-all duration-500 ease-in-out"
                                />
                            )}

                            {/* 1C. Draw Primary Optimal Path Line (Thin, bright yellow on top of glow) */}
                            {isPath && (
                                <line 
                                    x1={uPos.x} y1={uPos.y} 
                                    x2={vPos.x} y2={vPos.y} 
                                    stroke="rgb(255, 255, 100)" 
                                    strokeWidth="1.2" // Slightly thinner than glow for contrast
                                    className="transition-all duration-500 ease-in-out"
                                />
                            )}
                            
                            {/* Edge Weight (on top) */}
                            <text 
                                x={(uPos.x + vPos.x) / 2 + textOffsetX} 
                                y={(uPos.y + vPos.y) / 2 + textOffsetY} 
                                fontSize="2" 
                                textAnchor="middle" 
                                fill={isPath ? "rgb(255, 255, 200)" : "rgb(100, 116, 139)"}
                                className="font-bold"
                            >
                                {weight}
                            </text>
                        </React.Fragment>
                    );
                })}

                {/* 2. Draw all nodes */}
                {nodes.map(node => {
                    const pos = NODE_POSITIONS[node];
                    const isStart = node === startNode;
                    const isEnd = node === endNode;
                    const isMandatory = mandatoryStops.includes(node);
                    const isKeyStation = isStart || isEnd || isMandatory;
                    const blinkOpacity = isBlinking ? 0.4 : 1.0; 

                    return (
                        <React.Fragment key={node}>
                            {/* Node Circle - Use Gradient Fill for reflective look & Blinking effect */}
                            <circle 
                                cx={pos.x} cy={pos.y} r="1.3" 
                                fill={getNodeFill(node)}
                                stroke={isKeyStation ? 'white' : 'transparent'} 
                                strokeWidth="0.1"
                                className={`transition-all duration-300 transform hover:scale-125`}
                                style={{ opacity: isKeyStation ? blinkOpacity : 1.0, transition: 'opacity 1s linear' }}
                            />
                            {/* Node Label */}
                            <text 
                                x={pos.x} y={pos.y - 2.5} 
                                fontSize="2.5" 
                                textAnchor="middle" 
                                fill="white"
                            >
                                {node.split('_').map(w => w.charAt(0)).join('')}
                            </text>
                        </React.Fragment>
                    );
                })}

                {/* 3. Draw the animated Truck */}
                {truckCoords && (
                    <DeliveryTruckIcon x={truckCoords.x} y={truckCoords.y} rotation={truckRotation} />
                )}

            </svg>
            <div className="absolute top-2 right-2 text-xs font-mono text-white/50">
                Units: km/min
            </div>
            <div className="absolute bottom-2 left-2 text-xs text-white/70 bg-slate-900/50 p-1 rounded">
                Legend: <span className="text-green-400">Start</span>, <span className="text-red-400">End</span>, <span className="text-orange-400">Mandatory Stop</span>, <span className="text-yellow-400">Optimal Path</span>
            </div>
        </div>
    );
};

// --- Gemini Integration Component ---
const GeminiIntegration = ({ result, graph, startNode, endNode }) => {
    const [summary, setSummary] = useState(null);
    const [report, setReport] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const isPathReady = result && result.path && result.path.length > 0;
    
    // Generates a descriptive summary and risk assessment for the optimized route
    const generateSummary = async () => {
        if (!isPathReady) return;
        setIsLoading(true);
        setError(null);
        setSummary(null);

        const routeText = `Optimized Route Sequence: ${result.path.join(' -> ')}. Total Distance: ${result.distance} units.`;
        const graphDescription = JSON.stringify(graph);

        const userQuery = `Analyze the following optimal delivery route and network structure. Provide a concise, two-paragraph summary (max 150 words). The first paragraph should describe the sequence and total distance. The second paragraph should identify the single longest segment and one potential high-traffic or high-cost node to watch for, providing a brief risk assessment based on the provided data.
        Route details: ${routeText}
        Network weights: ${graphDescription}`;

        const payload = {
            contents: [{ parts: [{ text: userQuery }] }],
            systemInstruction: {
                parts: [{ text: "You are a specialized Logistics AI Assistant. Provide helpful, conversational, and actionable insights to a human operator." }]
            },
            config: { maxOutputTokens: 250 }
        };

        try {
            const apiResponse = await fetchWithRetry(GEMINI_API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const text = apiResponse.candidates?.[0]?.content?.parts?.[0]?.text || "Error: AI could not generate a valid summary.";
            setSummary(text);
            setReport(null); // Clear other output
        } catch (err) {
            setError("Failed to fetch summary from AI. Check API key/console for details.");
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };
    
    // Generates a simple, copy-paste ready operational report draft
    const generateReportDraft = async () => {
        if (!isPathReady) return;
        setIsLoading(true);
        setError(null);
        setReport(null);
        
        const routeSequence = result.path.join(' → ');
        const distance = result.distance;

        const userQuery = `Draft a concise, bulleted operational report for a logistics manager based on the following route. Include: 1) Date, 2) Start/End, 3) Optimal Sequence, 4) Total Distance, 5) Note on Looping. Use bullet points and professional language.`;

        const payload = {
            contents: [{ parts: [{ text: userQuery }] }],
            systemInstruction: {
                parts: [{ text: `You are a professional report drafting AI. Use the following data: Start Node: ${startNode}, End Node: ${endNode}, Optimal Path: ${routeSequence}, Distance: ${distance}.` }]
            },
            config: { maxOutputTokens: 150 }
        };

        try {
            const apiResponse = await fetchWithRetry(GEMINI_API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const text = apiResponse.candidates?.[0]?.content?.parts?.[0]?.text || "Error: AI could not generate a valid report.";
            setReport(text);
            setSummary(null); // Clear other output
        } catch (err) {
            setError("Failed to fetch report draft from AI. Check API key/console for details.");
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="p-5 bg-white border border-gray-200 rounded-lg shadow-md mt-6 space-y-4">
            <h2 className="text-xl font-semibold text-gray-800 flex items-center mb-3">
                <Bot className="w-5 h-5 mr-2 text-fuchsia-500" />
                Gemini AI Operational Insights
            </h2>
            
            <div className='flex space-x-3'>
                <button
                    onClick={generateSummary}
                    disabled={!isPathReady || isLoading}
                    className={`flex-1 flex items-center justify-center p-2 rounded-lg text-sm font-semibold transition duration-300 ${isPathReady ? 'bg-fuchsia-500 hover:bg-fuchsia-600 text-white' : 'bg-gray-300 text-gray-600 cursor-not-allowed'}`}
                >
                    {isLoading && summary === null ? 'Generating...' : <>✨ Route Summary & Risk</>}
                </button>

                <button
                    onClick={generateReportDraft}
                    disabled={!isPathReady || isLoading}
                    className={`flex-1 flex items-center justify-center p-2 rounded-lg text-sm font-semibold transition duration-300 ${isPathReady ? 'bg-fuchsia-500 hover:bg-fuchsia-600 text-white' : 'bg-gray-300 text-gray-600 cursor-not-allowed'}`}
                >
                     {isLoading && report === null ? 'Drafting...' : <>✨ Operational Report Draft</>}
                </button>
            </div>
            
            {(summary || report) && (
                <div className="p-4 bg-fuchsia-50 border border-fuchsia-300 rounded-lg whitespace-pre-wrap text-sm text-gray-800 font-mono">
                    {summary || report}
                </div>
            )}
            
            {error && <p className="text-red-500 text-sm">{error}</p>}
        </div>
    );
};

// --- Main Application Component ---
const App = () => {
    const [startNode, setStartNode] = useState('Warehouse');
    const [endNode, setEndNode] = useState('Delivery_Hub');
    const [mandatoryStops, setMandatoryStops] = useState(['C', 'F']); 
    const [result, setResult] = useState({ distance: null, path: null, sequence: null });
    const [isLoading, setIsLoading] = useState(false);
    const [maxNodesError, setMaxNodesError] = useState(false);
    const [isBlinking, setIsBlinking] = useState(false); // New state for blinking effect
    
    // New state for animation coordinates
    const initialTruckCoords = NODE_POSITIONS['Warehouse'];
    const [truckCoords, setTruckCoords] = useState(initialTruckCoords); 

    // Effect for Node Blinking: Toggles state every 1 second
    useEffect(() => {
        const blinkInterval = setInterval(() => {
            setIsBlinking(prev => !prev);
        }, 1000); 

        return () => clearInterval(blinkInterval);
    }, []); 


    // --- Animation Logic (Modified for Looping) ---
    const animationRef = React.useRef(null); // Ref to hold the animation interval
    const animateTruck = (path) => {
        // Clear any existing animation before starting a new one
        if (animationRef.current) {
            clearInterval(animationRef.current);
        }

        const animationSpeedMsPerUnit = 200; 
        
        if (path.length < 2) {
             setTruckCoords(NODE_POSITIONS[path[0]]); 
             return; // Loading is cleared outside, this handles position
        }

        let currentSegmentIndex = 0; // Index for the current segment's start node
        
        // Reset truck to the start node position before starting
        setTruckCoords(NODE_POSITIONS[path[0]]);

        const moveNextSegment = () => {
            let startNodeName = path[currentSegmentIndex];
            let endNodeName;
            
            // Check for loop condition
            const isLastSegment = (currentSegmentIndex === path.length - 1);

            if (isLastSegment) {
                // Loop back to the start node (path[0])
                endNodeName = path[0]; 
                currentSegmentIndex = 0; // Reset index for the next loop start
            } else {
                endNodeName = path[currentSegmentIndex + 1];
                currentSegmentIndex++; // Move to the next segment's start node
            }
            
            const start = NODE_POSITIONS[startNodeName];
            const end = NODE_POSITIONS[endNodeName];
            
            // Determine distance for segment. 
            // We use the direct edge weight from the graph, which covers the loop-back (END to START) if defined.
            const segmentDistance = DELIVERY_NETWORK[startNodeName]?.[endNodeName] || 0;

            if (segmentDistance === 0 && startNodeName !== endNodeName) { 
                console.warn(`No direct weight found between ${startNodeName} and ${endNodeName}. Stopping loop.`);
                if (animationRef.current) clearInterval(animationRef.current);
                return;
            }

            const segmentAnimationDuration = segmentDistance * animationSpeedMsPerUnit;
            const stepsPerSegment = 30;
            const intervalDuration = segmentAnimationDuration / stepsPerSegment;

            let step = 0;
            const interval = setInterval(() => {
                step++;
                const t = Math.min(1, step / stepsPerSegment);

                const newX = start.x * (1 - t) + end.x * t;
                const newY = start.y * (1 - t) + end.y * t;

                setTruckCoords({ x: newX, y: newY });

                if (step >= stepsPerSegment) {
                    clearInterval(interval);
                    // Start the next segment animation
                    moveNextSegment(); 
                }
            }, intervalDuration);
            animationRef.current = interval; 
        };

        moveNextSegment();
    };
    // --- End Animation Logic ---

    // Handler for multi-select dropdown
    const handleMandatoryChange = (e) => {
        const selectedOptions = Array.from(e.target.selectedOptions, option => option.value);
        if (selectedOptions.length > 5) {
            // Brute force is exponential (N!). Limit to keep the app responsive.
            setMaxNodesError(true);
            return;
        }
        setMaxNodesError(false);
        setMandatoryStops(selectedOptions);
    };

    // Processes the TSP result and initiates the truck animation
    const processResultAndAnimate = (distance, path) => {
        if (path && path.length > 0) {
            // Extract the sequence of mandatory stops from the full path for display
            const optimalSequence = path.filter(node => mandatoryStops.includes(node));
            setResult({ distance, path, sequence: optimalSequence });
            animateTruck(path); // Start animation (which now loops)
        } else {
            setResult({ distance: null, path: null, sequence: null });
        }
        // FIX: Always stop loading here, regardless of success or failure
        setIsLoading(false);
    };

    // Main optimization runner
    const runOptimization = () => {
        setIsLoading(true);
        setResult({ distance: null, path: null, sequence: null });
        setTruckCoords(NODE_POSITIONS[startNode]); // Place truck at start immediately

        // Enforce limits and validity
        if (mandatoryStops.length > 5 || startNode === endNode) {
             setIsLoading(false);
             return;
        }

        // Simulate a slight delay for better UX (as TSP can take a moment)
        setTimeout(() => {
            const { distance, path } = tspSolver(DELIVERY_NETWORK, mandatoryStops, startNode, endNode);
            processResultAndAnimate(distance, path);
        }, 500); 
    };

    // Run once on initial load and whenever inputs change
    useEffect(() => {
        runOptimization();
    }, [startNode, endNode, mandatoryStops]); // eslint-disable-line react-hooks/exhaustive-deps

    // Cleanup interval on unmount
    useEffect(() => {
        return () => {
            if (animationRef.current) {
                clearInterval(animationRef.current);
            }
        };
    }, []);


    // Determine the style for the main button based on loading state
    const buttonClass = isLoading || startNode === endNode || maxNodesError
        ? "bg-indigo-600 cursor-not-allowed"
        : "bg-indigo-500 hover:bg-indigo-600 active:bg-indigo-700 shadow-lg hover:shadow-xl";

    const isError = result.path === null && !isLoading && !maxNodesError;
    const isReady = !isLoading && result.path !== null;

    return (
        <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4 font-inter">
            <style>{`
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap');
                .font-inter { font-family: 'Inter', sans-serif; }
                /* Custom Keyframe for subtle shimmer effect on success */
                @keyframes shimmer {
                    0%, 100% { background-position: -200% 0; }
                    50% { background-position: 200% 0; }
                }
                .shimmer-bg {
                    background: linear-gradient(90deg, #10b981 0%, #059669 50%, #10b981 100%);
                    background-size: 200% 100%;
                    animation: shimmer 1.5s infinite linear;
                }
                /* Custom style to make multi-select look better */
                .multi-select-custom {
                    height: 150px;
                    background-image: none; /* Hide default select arrow */
                }
            `}</style>
            
            <div className="w-full max-w-6xl bg-white rounded-2xl shadow-2xl p-6 md:p-10 border border-gray-200">
                <header className="mb-8 text-center">
                    <h1 className="text-4xl font-extrabold text-gray-900 flex items-center justify-center mb-2">
                        <Route className="w-8 h-8 mr-3 text-indigo-500" />
                        OptiLoop: Optimal Delivery Route Planner
                    </h1>
                    <p className="text-gray-500 text-lg">
                        DSA Project: Solving the Traveling Salesperson Problem (TSP) Variant
                    </p>
                </header>

                <div className="grid md:grid-cols-3 gap-8">
                    {/* Control Panel */}
                    <div className="md:col-span-1 space-y-6">
                        <div className="p-5 bg-indigo-50 border-t-4 border-indigo-500 rounded-lg shadow-md transition duration-300 hover:shadow-xl">
                            <h2 className="text-xl font-semibold text-indigo-800 flex items-center mb-3">
                                <Map className="w-5 h-5 mr-2" />
                                Configure Route Parameters
                            </h2>
                            
                            {/* Start Node Dropdown */}
                            <label className="block text-sm font-medium text-gray-700 mb-1">Start Location</label>
                            <select 
                                value={startNode} 
                                onChange={(e) => setStartNode(e.target.value)}
                                className="w-full p-2.5 mb-4 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500 transition duration-150 ease-in-out shadow-sm"
                            >
                                {ALL_NODES.map(node => (
                                    <option key={`start-${node}`} value={node} disabled={node === endNode}>
                                        {node}
                                    </option>
                                ))}
                            </select>

                            {/* End Node Dropdown */}
                            <label className="block text-sm font-medium text-gray-700 mb-1">Final Destination</label>
                            <select 
                                value={endNode} 
                                onChange={(e) => setEndNode(e.target.value)}
                                className="w-full p-2.5 mb-4 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500 transition duration-150 ease-in-out shadow-sm"
                            >
                                {ALL_NODES.map(node => (
                                    <option key={`end-${node}`} value={node} disabled={node === startNode}>
                                        {node}
                                    </option>
                                ))}
                            </select>

                            {/* Mandatory Stops Multi-Select */}
                            <label className="block text-sm font-medium text-gray-700 mb-1 flex items-center">
                                <ListChecks className="w-4 h-4 mr-1 text-gray-500" />
                                Mandatory Stops (Hold Ctrl/Cmd to select multiple)
                            </label>
                            <select 
                                multiple={true}
                                value={mandatoryStops} 
                                onChange={handleMandatoryChange}
                                className="w-full p-2.5 border border-gray-300 rounded-lg focus:ring-orange-500 focus:border-orange-500 transition duration-150 ease-in-out shadow-sm multi-select-custom"
                            >
                                {INTERMEDIATE_NODES.map(node => (
                                    <option key={`mand-${node}`} value={node} disabled={node === startNode || node === endNode}>
                                        {node}
                                    </option>
                                ))}
                            </select>
                            {maxNodesError && (
                                <p className="text-red-500 text-xs mt-1">Limit to 5 stops max (due to exponential complexity).</p>
                            )}

                        </div>
                        
                        {/* Optimize Button */}
                        <button
                            onClick={runOptimization}
                            disabled={isLoading || startNode === endNode || maxNodesError}
                            className={`w-full text-white font-bold py-3 px-4 rounded-lg flex items-center justify-center transition ease-in-out duration-300 transform ${buttonClass}`}
                        >
                            {isLoading ? (
                                <>
                                    <RefreshCcw className="w-5 h-5 mr-2 animate-spin" />
                                    Calculating All {getPermutations(mandatoryStops).length} Routes...
                                </>
                            ) : (
                                <>
                                    <Zap className="w-5 h-5 mr-2" />
                                    Optimize Multi-Stop Route
                                </>
                            )}
                        </button>

                        {/* Results Panel */}
                        <div className="p-5 bg-white border rounded-lg shadow-md transition duration-300 hover:shadow-xl">
                            <h2 className="text-xl font-semibold text-gray-800 mb-3 flex items-center">
                                <Route className="w-5 h-5 mr-2 text-green-600" />
                                Optimization Result
                            </h2>

                            {isError && (
                                <div className="p-3 bg-red-100 border-l-4 border-red-500 text-red-700 rounded-md">
                                    Error: Cannot find a path that connects all required stops.
                                </div>
                            )}

                            {isReady && (
                                <div className="space-y-4">
                                    <div className={`p-4 rounded-lg text-white font-bold transition duration-500 shimmer-bg`}>
                                        <div className="text-sm font-medium">Optimal Total Distance / Cost</div>
                                        <div className="text-3xl mt-1">{result.distance} units</div>
                                    </div>
                                    
                                    <div className="p-4 bg-gray-50 border rounded-lg">
                                        <div className="text-sm font-medium text-gray-600 mb-2">Optimal Stop Sequence</div>
                                        <p className="text-lg text-gray-800 break-words font-mono font-bold">
                                            {startNode} → {result.sequence.join(' → ')} → {endNode}
                                        </p>
                                    </div>

                                    <div className="p-4 bg-gray-50 border rounded-lg">
                                        <div className="text-sm font-medium text-gray-600 mb-2">Full Optimal Path</div>
                                        <p className="text-sm text-gray-700 break-words">
                                            {result.path.join(' → ')}
                                        </p>
                                    </div>
                                </div>
                            )}
                            
                            {!isReady && !isError && !isLoading && (
                                <p className="text-gray-500 italic">
                                    Select your start point, mandatory stops, and destination, then click "Optimize Multi-Stop Route."
                                </p>
                            )}
                        </div>

                    </div>

                    {/* Visualization Area */}
                    <div className="md:col-span-2">
                        <NetworkVisualizer 
                            path={result.path} 
                            startNode={startNode} 
                            endNode={endNode} 
                            mandatoryStops={mandatoryStops}
                            truckCoords={truckCoords}
                            isBlinking={isBlinking} // Pass blinking state
                        />
                        <div className="mt-4 p-4 text-center bg-indigo-100 rounded-lg text-indigo-800">
                            **DSA Concepts in Action:** **TSP Brute Force** (Permutations) combined with **Dijkstra's Algorithm** (for shortest path segments).
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
};

export default App;
