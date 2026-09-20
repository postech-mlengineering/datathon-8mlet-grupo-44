import React, { useState } from 'react';
import Login from './pages/Login';
import MotorCampanha from './pages/MotorCampanha';

const API_URL = "http://localhost:8000";

export default function App() {
  const [token, setToken] = useState(localStorage.getItem("token") || null);
  const [username, setUsername] = useState(localStorage.getItem("username") || "");

  const handleLoginSuccess = (newToken, newUsername) => {
    setToken(newToken);
    setUsername(newUsername);
    localStorage.setItem("token", newToken);
    localStorage.setItem("username", newUsername);
  };

  const handleLogout = () => {
    setToken(null);
    setUsername("");
    localStorage.clear();
  };

  if (!token) {
    return <Login onLoginSuccess={handleLoginSuccess} apiUrl={API_URL} />;
  }

  return (
    <MotorCampanha 
      username={username} 
      token={token} 
      apiUrl={API_URL} 
      onLogout={handleLogout} 
    />
  );
}