import React, { useState } from 'react';
import { Cog, AlertCircle } from 'lucide-react';
import Button from '../components/Button';

export default function Login({ onLoginSuccess, apiUrl }) {
  const [loginForm, setLoginForm] = useState({ username: "admin", password: "password123" });
  const [loginError, setLoginError] = useState("");

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoginError("");
    try {
      const formData = new URLSearchParams();
      formData.append("username", loginForm.username);
      formData.append("password", loginForm.password);

      const res = await fetch(`${apiUrl}/token`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: formData
      });

      if (res.ok) {
        const data = await res.json();
        onLoginSuccess(data.access_token, loginForm.username);
      } else {
        setLoginError("Credenciais inválidas. Tente admin / password123");
      }
    } catch (err) {
      if (loginForm.username === "admin" && loginForm.password === "password123") {
        onLoginSuccess("mock_token", "admin");
      } else {
        setLoginError("Falha de conexão com a FastAPI.");
      }
    }
  };

  return (
    <div className="min-h-screen bg-[#0A0D14] flex items-center justify-center p-4 font-sans text-slate-100">
      <div className="bg-[#121824] rounded-2xl shadow-2xl w-full max-w-md p-10 border border-slate-800/80">
        <div className="text-center mb-8">
          <div className="w-14 h-14 bg-slate-800 border border-slate-700/60 rounded-xl mx-auto flex items-center justify-center shadow-inner mb-5 text-indigo-400">
            <Cog className="w-7 h-7" />
          </div>
          <h1 className="text-xl font-semibold tracking-tight text-slate-100">Datathon FIAP</h1>
          <p className="text-slate-400 text-xs mt-1.5">Machine Learning Engineering</p>
        </div>

        {loginError && (
          <div className="mb-6 p-3 bg-red-950/40 border border-red-900/50 text-red-400 rounded-xl text-xs font-medium flex items-center gap-2.5">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{loginError}</span>
          </div>
        )}

        <form onSubmit={handleLogin} className="space-y-4">
          <div>
            <label className="block text-[11px] font-medium uppercase tracking-wider text-slate-400 mb-1.5 px-0.5">Usuário</label>
            <input 
              type="text" 
              value={loginForm.username} 
              onChange={(e) => setLoginForm({...loginForm, username: e.target.value})}
              className="w-full px-3.5 py-2.5 rounded-xl bg-[#1A2234] border border-slate-700/60 focus:border-indigo-500 focus:bg-[#121824] focus:outline-none transition-all text-slate-100 text-xs font-medium"
              required 
            />
          </div>
          <div>
            <label className="block text-[11px] font-medium uppercase tracking-wider text-slate-400 mb-1.5 px-0.5">Senha</label>
            <input 
              type="password" 
              value={loginForm.password} 
              onChange={(e) => setLoginForm({...loginForm, password: e.target.value})}
              className="w-full px-3.5 py-2.5 rounded-xl bg-[#1A2234] border border-slate-700/60 focus:border-indigo-500 focus:bg-[#121824] focus:outline-none transition-all text-slate-100 text-xs font-medium"
              required 
            />
          </div>
          <Button type="submit" className="w-full mt-4">
            Acessar
          </Button>
        </form>
      </div>
    </div>
  );
}