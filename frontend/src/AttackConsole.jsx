import React, { useState } from 'react';
import { Skull, Zap, Database, Code, Terminal, Activity, ArrowRight, ShieldAlert, Shield } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

function AttackConsole() {
    const [activeAttack, setActiveAttack] = useState(null);
    const [log, setLog] = useState([]);
    const navigate = useNavigate();

    const addLog = (msg) => {
        const timestamp = new Date().toLocaleTimeString();
        setLog(prev => [`[${timestamp}] ${msg}`, ...prev]);
    };

    const launchAttack = async (type, endpoint, name) => {
        setActiveAttack(type);
        addLog(`INITIALIZING ${name}...`);
        addLog(`TARGET: 127.0.0.1 (IDS SYSTEM)`);

        try {
            const res = await fetch(`http://localhost:8001/${endpoint}`, { method: 'POST' });
            const data = await res.json();
            addLog(`STATUS: ${data.status}`);
            addLog(`EXECUTING PAYLOADS...`);

            setTimeout(() => {
                setActiveAttack(null);
                addLog(`COMPLETE: ${name}`);
            }, 3000);

        } catch (e) {
            addLog(`ERROR: Connection Refused. Is Attacker Backend running?`);
            setActiveAttack(null);
        }
    };

    return (
        <div className="h-full bg-[#0a0a0a] text-red-500 font-mono p-4 relative overflow-y-auto">

            {/* Background Grid - Matrix style */}
            <div className="fixed inset-0 pointer-events-none opacity-20"
                style={{ backgroundImage: 'linear-gradient(0deg, transparent 24%, rgba(255, 0, 0, .3) 25%, rgba(255, 0, 0, .3) 26%, transparent 27%, transparent 74%, rgba(255, 0, 0, .3) 75%, rgba(255, 0, 0, .3) 76%, transparent 77%, transparent), linear-gradient(90deg, transparent 24%, rgba(255, 0, 0, .3) 25%, rgba(255, 0, 0, .3) 26%, transparent 27%, transparent 74%, rgba(255, 0, 0, .3) 75%, rgba(255, 0, 0, .3) 76%, transparent 77%, transparent)', backgroundSize: '50px 50px' }}>
            </div>

            <div className="max-w-6xl mx-auto relative z-10">

                {/* Header */}
                <header className="flex justify-between items-end mb-12 border-b border-red-900/50 pb-6">
                    <div>
                        <h1 className="text-4xl font-bold tracking-tighter flex items-center gap-4">
                            <Skull className="w-12 h-12" />
                            RED TEAM CONSOLE
                        </h1>
                        <p className="text-red-700 mt-2">ADVANCED THREAT EMULATION FRAMEWORK</p>
                    </div>
                    <button
                        onClick={() => navigate('/')}
                        className="flex items-center gap-2 text-sm bg-red-900/20 px-4 py-2 border border-red-900 rounded hover:bg-red-900/40 transition-colors"
                    >
                        SWITCH TO DEFENDER VIEW <ArrowRight size={16} />
                    </button>
                </header>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                    {/* Attack Menu */}
                    <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-4">

                        {/* Normal Traffic Card (Green) */}
                        <button
                            onClick={() => launchAttack('normal', 'traffic/normal', 'NORMAL TRAFFIC SIM')}
                            disabled={activeAttack !== null}
                            className={`group p-6 border ${activeAttack === 'normal' ? 'bg-green-900/30 border-green-500 animate-pulse' : 'bg-neutral-900/50 border-green-900/40 hover:border-green-500 hover:bg-green-950/30'} text-left transition-all rounded-lg col-span-1 md:col-span-2`}
                        >
                            <div className="flex justify-between items-start mb-4">
                                <Shield className="w-8 h-8 opacity-80 text-green-500" />
                                {activeAttack === 'normal' && <span className="text-xs animate-pulse text-green-500">GENERATING...</span>}
                            </div>
                            <h3 className="text-xl font-bold mb-2 text-green-500">NORMAL TRAFFIC</h3>
                            <p className="text-sm opacity-60 text-green-500/80">Generate standard HTTP/User traffic to demonstrate normal system behavior and baseline metrics. No attacks.</p>
                        </button>

                        {/* DDoS Card */}
                        <button
                            onClick={() => launchAttack('ddos', 'attack/ddos', 'VOLUMETRIC DDoS')}
                            disabled={activeAttack !== null}
                            className={`group p-6 border ${activeAttack === 'ddos' ? 'bg-red-900/30 border-red-500 animate-pulse' : 'bg-neutral-900/50 border-red-900/40 hover:border-red-500 hover:bg-red-950/30'} text-left transition-all rounded-lg`}
                        >
                            <div className="flex justify-between items-start mb-4">
                                <Activity className="w-8 h-8 opacity-80" />
                                {activeAttack === 'ddos' && <span className="text-xs animate-pulse">RUNNING...</span>}
                            </div>
                            <h3 className="text-xl font-bold mb-2">UDP FLOOD (DDoS)</h3>
                            <p className="text-sm opacity-60">High-volume packet flooding to overwhelm network bandwidth and induce Denial of Service.</p>
                        </button>

                        {/* Malware Card */}
                        <button
                            onClick={() => launchAttack('malware', 'attack/malware', 'MALWARE INFECTION')}
                            disabled={activeAttack !== null}
                            className={`group p-6 border ${activeAttack === 'malware' ? 'bg-red-900/30 border-red-500 animate-pulse' : 'bg-neutral-900/50 border-red-900/40 hover:border-red-500 hover:bg-red-950/30'} text-left transition-all rounded-lg`}
                        >
                            <div className="flex justify-between items-start mb-4">
                                <ShieldAlert className="w-8 h-8 opacity-80" />
                                {activeAttack === 'malware' && <span className="text-xs animate-pulse">RUNNING...</span>}
                            </div>
                            <h3 className="text-xl font-bold mb-2">MALWARE C2 BEACON</h3>
                            <p className="text-sm opacity-60">Simulates infected endpoint communication with Command & Control servers.</p>
                        </button>

                        {/* SQL Injection Card */}
                        <button
                            onClick={() => launchAttack('sqli', 'attack/sqli', 'SQL INJECTION')}
                            disabled={activeAttack !== null}
                            className={`group p-6 border ${activeAttack === 'sqli' ? 'bg-red-900/30 border-red-500 animate-pulse' : 'bg-neutral-900/50 border-red-900/40 hover:border-red-500 hover:bg-red-950/30'} text-left transition-all rounded-lg`}
                        >
                            <div className="flex justify-between items-start mb-4">
                                <Database className="w-8 h-8 opacity-80" />
                                {activeAttack === 'sqli' && <span className="text-xs animate-pulse">RUNNING...</span>}
                            </div>
                            <h3 className="text-xl font-bold mb-2">SQL INJECTION</h3>
                            <p className="text-sm opacity-60">Payload delivery targeting database vulnerabilities using automated injection patterns.</p>
                        </button>

                        {/* XSS Card */}
                        <button
                            onClick={() => launchAttack('xss', 'attack/xss', 'CROSS-SITE SCRIPTING')}
                            disabled={activeAttack !== null}
                            className={`group p-6 border ${activeAttack === 'xss' ? 'bg-red-900/30 border-red-500 animate-pulse' : 'bg-neutral-900/50 border-red-900/40 hover:border-red-500 hover:bg-red-950/30'} text-left transition-all rounded-lg`}
                        >
                            <div className="flex justify-between items-start mb-4">
                                <Code className="w-8 h-8 opacity-80" />
                                {activeAttack === 'xss' && <span className="text-xs animate-pulse">RUNNING...</span>}
                            </div>
                            <h3 className="text-xl font-bold mb-2">XSS EXPLOIT</h3>
                            <p className="text-sm opacity-60">Injecting client-side scripts to test validation filters and session hijacking defense.</p>
                        </button>

                    </div>

                    {/* Terminal / Logs */}
                    <div className="bg-black border border-red-900/60 rounded-lg p-4 font-mono text-xs overflow-hidden flex flex-col h-[500px]">
                        <div className="flex items-center gap-2 border-b border-red-900/30 pb-2 mb-2 text-red-600">
                            <Terminal size={14} />
                            <span>ATTACK_LOGS</span>
                        </div>
                        <div className="flex-1 overflow-y-auto space-y-1">
                            {log.length === 0 && <span className="opacity-30">Waiting for command...</span>}
                            {log.map((l, i) => (
                                <div key={i} className="break-words">
                                    <span className="text-red-400">&gt;</span> {l}
                                </div>
                            ))}
                        </div>
                    </div>

                </div>
            </div>
        </div>
    );
}

export default AttackConsole;
