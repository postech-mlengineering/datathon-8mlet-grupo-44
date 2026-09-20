import React from 'react';
import Card from './Card';

export default function MetricCard({ title, value, valueColor = 'text-slate-100' }) {
  return (
    <Card className="p-5">
      <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">{title}</p>
      <p className={`text-2xl font-semibold mt-1 tracking-tight ${valueColor}`}>{value}</p>
    </Card>
  );
}