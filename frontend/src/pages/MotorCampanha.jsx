import React, { useState } from 'react';
import { Users, LogOut, Download, Cog } from 'lucide-react';
import Card from '../components/Card';
import MetricCard from '../components/MetricCard';
import Button from '../components/Button';
import Table from '../components/Table';

export default function MotorCampanha({ username, token, apiUrl, onLogout }) {
  const [verTodos, setVerTodos] = useState(false);
  const [topN, setTopN] = useState(50);
  const [loadingBatch, setLoadingBatch] = useState(false);
  const [batchResult, setBatchResult] = useState(null);

  const runBatchCampaign = async () => {
    setLoadingBatch(true);
    setBatchResult(null);
    try {
      const payload = { top_n: verTodos ? 0 : Number(topN) };
      const res = await fetch(`${apiUrl}/recommendations`, {
        method: "POST",
        headers: { 
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        const data = await res.json();
        setBatchResult(data);
      } else {
        alert("Erro ao processar campanha.");
      }
    } catch (err) {
      alert("Erro de conexão ou timeout na API.");
    } finally {
      setLoadingBatch(false);
    }
  };

  const downloadCSV = () => {
    if (!batchResult || !batchResult.recomendacoes) return;
    const items = batchResult.recomendacoes;
    const replacer = (key, value) => value === null ? '' : value;
    const header = Object.keys(items[0]);
    let csv = items.map(row => header.map(fieldName => JSON.stringify(row[fieldName], replacer)).join(','));
    csv.unshift(header.join(','));
    csv = csv.join('\r\n');

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", "campanha_leads.csv");
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-[#0A0D14] flex font-sans text-slate-100 antialiased">
      {/* Sidebar */}
      <aside className="w-72 bg-[#121824] border-r border-slate-800/80 flex flex-col justify-between p-6 z-10">
        <div>
          <div className="flex items-center gap-3 px-1 mb-8">
            <div className="w-9 h-9 bg-slate-800 border border-slate-700/50 rounded-xl flex items-center justify-center text-indigo-400 shadow-sm">
              <Cog className="w-4 h-4" />
            </div>
            <div>
              <h2 className="font-semibold text-slate-100 text-xs tracking-wide">Datathon FIAP</h2>
              <p className="text-[10px] text-slate-400 tracking-wider uppercase">ML Engineering</p>
            </div>
          </div>

          <nav className="space-y-1">
            <div className="w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl font-medium text-xs bg-indigo-600/15 text-indigo-400 border border-indigo-500/30">
              <Users className="w-4 h-4" />
              <span>Motor de Campanha</span>
            </div>
          </nav>
        </div>

        <div className="border-t border-slate-800/80 pt-4 px-1">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-slate-800 border border-slate-700/50 flex items-center justify-center font-semibold text-slate-200 text-xs">
                {username ? username[0].toUpperCase() : 'U'}
              </div>
              <div className="overflow-hidden">
                <p className="text-xs font-medium text-slate-200 truncate">{username}</p>
                <p className="text-[10px] text-emerald-400 font-medium flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span> Online
                </p>
              </div>
            </div>
          </div>
          <Button onClick={onLogout} variant="danger" className="w-full">
            <LogOut className="w-3.5 h-3.5" />
            <span>Sair</span>
          </Button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 p-10 overflow-y-auto max-h-screen bg-[#07090F]">
        <div className="max-w-6xl mx-auto space-y-6">
          <div>
            <h1 className="text-xl font-semibold tracking-tight text-slate-100">Motor de Campanha</h1>
            <p className="text-slate-400 text-xs mt-1">Seleção de leads por potencial de retorno financeiro.</p>
          </div>

          <Card className="space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-slate-200 text-xs">Parâmetros</h3>
                <p className="text-[11px] text-slate-400">Selecione o corte de leads ou realize a extração completa.</p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" checked={verTodos} onChange={(e) => setVerTodos(e.target.checked)} className="sr-only peer" />
                <div className="w-9 h-5 bg-slate-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-indigo-600 border border-slate-700"></div>
                <span className="ml-3 text-xs font-medium text-slate-300">Todos</span>
              </label>
            </div>

            {!verTodos && (
              <div className="space-y-2 pt-1">
                <div className="flex justify-between text-xs font-medium text-slate-400">
                  <span>Quantidade</span>
                  <span className="text-indigo-400 font-semibold">{topN} leads</span>
                </div>
                <input 
                  type="range" 
                  min="10" 
                  max="500" 
                  step="10"
                  value={topN} 
                  onChange={(e) => setTopN(e.target.value)}
                  className="w-full accent-indigo-500 cursor-pointer bg-slate-800 rounded-lg h-1.5" 
                />
              </div>
            )}

            <Button 
              onClick={runBatchCampaign}
              disabled={loadingBatch}
              className="w-full"
            >
              {loadingBatch ? "Executando pipeline preditivo no MLflow..." : "Executar"}
            </Button>
          </Card>

          {batchResult && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <MetricCard 
                  title="Leads" 
                  value={batchResult.recomendacoes.length} 
                />
                <MetricCard 
                  title="Lucro Projetado" 
                  value={`R$ ${batchResult.recomendacoes.reduce((acc, curr) => acc + curr.valor_esperado, 0).toFixed(2)}`} 
                  valueColor="text-emerald-400"
                />
              </div>

              <Card className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-slate-200 text-xs">Oportunidades</h3>
                  <Button onClick={downloadCSV} variant="secondary">
                    <Download className="w-3.5 h-3.5" />
                    <span>Exportar CSV</span>
                  </Button>
                </div>

                <Table data={batchResult.recomendacoes} />
              </Card>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}