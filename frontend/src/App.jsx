import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import { Shield, ShieldAlert, Activity, Server, Zap, Lock, AlertTriangle, Crosshair, LayoutTemplate } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts';
import AttackConsole from './AttackConsole';

function IDSDashboard() {
    const [status, setStatus] = useState('SECURE');
    const [packets, setPackets] = useState([]);
    const [stats, setStats] = useState({ total: 0, attacks: 0, normal: 0 });
    const [graphData, setGraphData] = useState([]);
    const [wsConnected, setWsConnected] = useState(false);
    const navigate = useNavigate();
    const maxPackets = 20;

    useEffect(() => {
        const ws = new WebSocket('ws://localhost:8000/ws/traffic');

        ws.onopen = () => {
            console.log('Connected to IDS Backend');
            setWsConnected(true);
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const timestamp = new Date().toLocaleTimeString();

            setPackets(prev => {
                const newPackets = [data, ...prev];
                return newPackets.slice(0, maxPackets);
            });

            setStats(prev => ({
                total: prev.total + 1,
                attacks: prev.attacks + (data.prediction === 'Attack' ? 1 : 0),
                normal: prev.normal + (data.prediction === 'Normal' ? 1 : 0)
            }));

            // Update Graph
            setGraphData(prev => {
                const newData = [...prev, {
                    time: timestamp,
                    confidence: parseFloat(data.confidence) * 100,
                    isAttack: data.prediction === 'Attack' ? 100 : 0,
                    type: data.prediction === 'Attack' ? 1 : 0
                }];
                if (newData.length > 20) newData.shift();
                return newData;
            });

            if (data.status_msg === 'BLOCKED') {
                setStatus('THREAT BLOCKED');
                if (window.statusTimeout) clearTimeout(window.statusTimeout);
                window.statusTimeout = setTimeout(() => setStatus('SECURE'), 3000);
            } else if (data.prediction === 'Attack') {
                setStatus('INTRUSION DETECTED');
                if (window.statusTimeout) clearTimeout(window.statusTimeout);
                window.statusTimeout = setTimeout(() => setStatus('SECURE'), 2000);
            }
        };

        ws.onclose = () => setWsConnected(false);

        return () => ws.close();
    }, []);

    return (
        <div className="min-h-screen text-white p-6 relative overflow-hidden flex flex-col">
            <div className="grid-bg"></div>

            {/* Header */}
            <header className="flex justify-between items-center mb-8">
                <div className="flex items-center gap-3">
                    <Shield className="w-10 h-10 text-[var(--primary)]" />
                    <div>
                        <h1 className="text-2xl font-bold font-['JetBrains_Mono'] tracking-tighter">
                            SENTINEL<span className="text-[var(--primary)]">.AI</span>
                        </h1>
                        <p className="text-xs text-[var(--text-secondary)] mono">SAC-Reinforced Intrusion Detection System</p>
                    </div>
                </div>

                <div className="flex items-center gap-4">
                    <button
                        onClick={() => navigate('/split-view')}
                        className="flex items-center gap-2 px-4 py-2 rounded-full border border-blue-500/50 bg-blue-900/10 hover:bg-blue-900/30 transition-all text-blue-400 text-sm font-bold tracking-wider"
                    >
                        <LayoutTemplate size={16} /> SPLIT VIEW
                    </button>

                    <button
                        onClick={() => navigate('/attack')}
                        className="flex items-center gap-2 px-4 py-2 rounded-full border border-red-500/50 bg-red-900/10 hover:bg-red-900/30 transition-all text-red-400 text-sm font-bold tracking-wider"
                    >
                        <Crosshair size={16} /> ATTACKER CONSOLE
                    </button>

                    <div className={`flex items-center gap-2 px-4 py-2 rounded-full border ${wsConnected ? 'border-[var(--success)] bg-[var(--success)]/10' : 'border-[var(--danger)] bg-[var(--danger)]/10'}`}>
                        <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-[var(--success)] animate-pulse' : 'bg-[var(--danger)]'}`}></div>
                        <span className="text-sm font-semibold mono">{wsConnected ? 'SYSTEM ONLINE' : 'DISCONNECTED'}</span>
                    </div>
                </div>
            </header>

            {/* Main Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-6">

                {/* Status Panel */}
                <div className="glass-panel p-6 flex flex-col items-center justify-center min-h-[180px] relative overflow-hidden group lg:col-span-1">
                    <div className={`absolute inset-0 opacity-10 ${status === 'SECURE' ? 'bg-[var(--success)]' : (status === 'THREAT BLOCKED' ? 'bg-[var(--primary)]' : 'bg-[var(--danger)]')} transition-colors duration-500`}></div>

                    {status === 'SECURE' ? (
                        <>
                            <Lock className="w-20 h-20 text-[var(--success)] mb-2 drop-shadow-[0_0_15px_rgba(0,255,136,0.5)]" />
                            <h2 className="text-2xl font-bold text-[var(--success)] tracking-widest">SECURE</h2>
                        </>
                    ) : status === 'THREAT BLOCKED' ? (
                        <>
                            <Shield className="w-20 h-20 text-[var(--primary)] mb-2 drop-shadow-[0_0_15px_rgba(0,242,255,0.5)] animate-pulse" />
                            <h2 className="text-xl font-bold text-[var(--primary)] tracking-widest text-center">MITIGATION ACTIVE<br />IP BLOCKED</h2>
                        </>
                    ) : (
                        <>
                            <ShieldAlert className="w-20 h-20 text-[var(--danger)] mb-2 drop-shadow-[0_0_15px_rgba(255,0,85,0.5)] animate-bounce" />
                            <h2 className="text-2xl font-bold text-[var(--danger)] tracking-widest danger-glow animate-pulse">INTRUSION</h2>
                        </>
                    )}
                </div>

                {/* Real-time Threat Graph */}
                <div className="glass-panel p-4 lg:col-span-2 flex flex-col">
                    <div className="flex items-center justify-between mb-2">
                        <h3 className="font-bold text-sm text-[var(--text-primary)] flex items-center gap-2">
                            <Activity size={16} className="text-[var(--primary)]" />
                            Live Threat Analysis
                        </h3>
                    </div>
                    <div className="flex-1 w-full h-[150px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={graphData}>
                                <defs>
                                    <linearGradient id="colorConf" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#00f2ff" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#00f2ff" stopOpacity={0} />
                                    </linearGradient>
                                    <linearGradient id="colorAttack" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#ff0055" stopOpacity={0.8} />
                                        <stop offset="95%" stopColor="#ff0055" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <XAxis dataKey="time" hide />
                                <YAxis hide domain={[0, 100]} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#0f0f1a', border: '1px solid #333' }}
                                    itemStyle={{ color: '#fff' }}
                                />
                                <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                                <Area type="monotone" dataKey="confidence" stroke="#00f2ff" fillOpacity={1} fill="url(#colorConf)" strokeWidth={2} />
                                <Area type="step" dataKey="isAttack" stroke="#ff0055" fillOpacity={1} fill="url(#colorAttack)" strokeWidth={2} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Stats Panel */}
                <div className="glass-panel p-6 flex flex-col justify-center gap-4 lg:col-span-1">
                    <div className="flex justify-between items-center border-b border-white/10 pb-2">
                        <span className="text-xs text-[var(--text-secondary)]">PACKETS</span>
                        <span className="font-mono font-bold text-xl">{stats.total}</span>
                    </div>
                    <div className="flex justify-between items-center border-b border-white/10 pb-2">
                        <span className="text-xs text-[var(--danger)]">ATTACKS</span>
                        <span className="font-mono font-bold text-xl text-[var(--danger)]">{stats.attacks}</span>
                    </div>
                    <div className="flex justify-between items-center">
                        <span className="text-xs text-[var(--success)]">NORMAL</span>
                        <span className="font-mono font-bold text-xl text-[var(--success)]">{stats.normal}</span>
                    </div>
                </div>

            </div>

            {/* Live Traffic Table */}
            <div className="glass-panel overflow-hidden flex-1 flex flex-col min-h-0">
                <h3 className="p-4 text-sm font-bold flex items-center gap-2 border-b border-white/10">
                    <Server size={16} className="text-[var(--primary)]" />
                    Network Packet Stream
                </h3>
                <div className="overflow-auto flex-1">
                    <table className="w-full text-left border-collapse">
                        <thead className="sticky top-0 bg-[#0f0f1a] z-10">
                            <tr className="text-[var(--text-secondary)] text-xs uppercase border-b border-[var(--text-secondary)]/10">
                                <th className="p-3 font-normal">Time</th>
                                <th className="p-3 font-normal">Source</th>
                                <th className="p-3 font-normal">Proto</th>
                                <th className="p-3 font-normal">Prediction</th>
                                <th className="p-3 font-normal">Confidence</th>
                            </tr>
                        </thead>
                        <tbody className="mono text-xs">
                            {packets.map((pkt) => (
                                <tr key={pkt.id} className="border-b border-[var(--text-secondary)]/5 hover:bg-[var(--primary)]/5 transition-colors">
                                    <td className="p-3 text-[var(--text-secondary)]">
                                        {new Date(pkt.timestamp * 1000).toLocaleTimeString()}
                                    </td>
                                    <td className="p-3">{pkt.src_ip}</td>
                                    <td className="p-3 text-[var(--text-primary)]">{pkt.protocol}</td>
                                    <td className="p-3">
                                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${pkt.prediction === 'Attack'
                                                ? 'bg-[var(--danger)]/20 text-[var(--danger)] border border-[var(--danger)]/50'
                                                : 'bg-[var(--success)]/20 text-[var(--success)] border border-[var(--success)]/50'
                                            }`}>
                                            {pkt.prediction.toUpperCase()}
                                        </span>
                                    </td>
                                    <td className="p-3">
                                        <div className="flex items-center gap-2">
                                            <div className="w-12 h-1 bg-[var(--card-bg)] rounded-full overflow-hidden">
                                                <div
                                                    className={`h-full ${pkt.prediction === 'Attack' ? 'bg-[var(--danger)]' : 'bg-[var(--success)]'}`}
                                                    style={{ width: `${parseFloat(pkt.confidence) * 100}%` }}
                                                ></div>
                                            </div>
                                            <span className="opacity-70">{parseInt(parseFloat(pkt.confidence) * 100)}%</span>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}

function SplitView() {
    return (
        <div className="w-screen h-screen flex overflow-hidden bg-black">
            <div className="w-1/2 h-full border-r border-[#333]">
                <IDSDashboard />
            </div>
            <div className="w-1/2 h-full">
                <AttackConsole />
            </div>
        </div>
    )
}

function App() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<IDSDashboard />} />
                <Route path="/attack" element={<AttackConsole />} />
                <Route path="/split-view" element={<SplitView />} />
            </Routes>
        </Router>
    );
}

export default App;
