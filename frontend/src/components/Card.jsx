import React from 'react';

export default function Card({ children, className = '' }) {
  return (
    <div className={`bg-[#121824] p-6 rounded-2xl border border-slate-800/80 shadow-sm ${className}`}>
      {children}
    </div>
  );
}