import React from 'react';

export default function Table({ data }) {
  if (!data || data.length === 0) return null;

  return (
    <div className="overflow-x-auto border border-slate-800 rounded-xl">
      <table className="w-full text-left border-collapse text-xs">
        <thead>
          <tr className="bg-[#182030] border-b border-slate-800 text-slate-400 font-semibold uppercase tracking-wider text-[10px]">
            <th className="p-3">Client ID</th>
            <th className="p-3">Probabilidade</th>
            <th className="p-3">Valor Esperado</th>
            <th className="p-3">Decisão</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/60 text-slate-300">
          {data.map((item, index) => (
            <tr key={index} className="hover:bg-slate-800/30 transition-colors">
              <td className="p-3 font-medium text-slate-100">{item.client_id}</td>
              <td className="p-3">{(item.probabilidade_aceitacao * 100).toFixed(1)}%</td>
              <td className="p-3 font-semibold text-emerald-400">R$ {item.valor_esperado.toFixed(2)}</td>
              <td className="p-3">
                <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold tracking-wide ${item.decisao === 'Ofertar' ? 'bg-emerald-950/40 text-emerald-400 border border-emerald-900/30' : 'bg-rose-950/40 text-rose-400 border border-rose-900/30'}`}>
                  {item.decisao}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}