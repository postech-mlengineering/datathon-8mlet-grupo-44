import React from 'react';

export default function Button({ children, onClick, disabled, variant = 'primary', className = '', ...props }) {
  const baseStyle = "py-2.5 px-4 font-medium rounded-xl shadow-md transition-all text-xs flex items-center justify-center gap-2 disabled:opacity-50";
  const variants = {
    primary: "bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-600/20",
    secondary: "bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700/60",
    danger: "bg-red-950/40 hover:bg-red-900/50 text-red-400 border border-red-900/30"
  };

  return (
    <button 
      onClick={onClick} 
      disabled={disabled} 
      className={`${baseStyle} ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}