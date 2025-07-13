"use client";

import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import axios from 'utils/axios';

interface LoginData {
  email: string;
  password: string;
}

interface Account {
  _id: string;
  name: string;
  surname: string;
  email: string;
  password?: string;
  role: 'user' | 'admin';
  isVerified: boolean;
  verificationCode?: string;
  verificationCodeExpires?: Date;
  mfaSecret?: string;
  mfaEnabled: boolean;
  organizationName: string;
  invitedBy?: string;
  canInvite: boolean;
  mustCompleteProfile: boolean; // Ajouté pour correspondre au modèle
}

export interface FormData {
  name?: string;
  surname?: string;
  organizationName: string;
  email: string;
  password: string;
}

interface AuthContextType {
  account: Account | null;
  token: string | null;
  isLoggedIn: boolean;
  login: (data: LoginData) => Promise<void>;
  register: (data: FormData & { role: 'user' | 'admin'; canInvite: boolean; isVerified: boolean; mfaEnabled: boolean }) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [account, setAccount] = useState<Account | null>(null);
  const [token, setToken] = useState<string | null>(null);

  // Restaurer l'état d'authentification depuis localStorage au chargement
  useEffect(() => {
    const savedToken = localStorage.getItem('token');
    const savedAccount = localStorage.getItem('account');
    console.log('Restauration depuis localStorage - Token:', savedToken);
    console.log('Restauration depuis localStorage - Account:', savedAccount);
    if (savedToken && savedAccount) {
      setToken(savedToken);
      try {
        const parsedAccount = JSON.parse(savedAccount) as Account;
        setAccount(parsedAccount);
      } catch (err) {
        console.error('Erreur lors de la restauration de l\'account:', err);
        localStorage.removeItem('token');
        localStorage.removeItem('account');
      }
    }
  }, []);

  const isLoggedIn = !!account && !!token;

  const login = async (data: LoginData) => {
    try {
      const response = await axios.post('/auth/login', { email: data.email, password: data.password });
      const { token, data: accountData } = response.data;
      console.log('Nouveau token reçu:', token);
      console.log('Données de l\'account reçues:', accountData);
      const fullAccount: Account = {
        ...accountData,
        role: accountData.role || 'user',
        isVerified: accountData.isVerified || false,
        mfaEnabled: accountData.mfaEnabled || false,
        organizationName: accountData.organizationName || '',
        canInvite: accountData.canInvite || false,
        mustCompleteProfile: accountData.mustCompleteProfile || false, // Ajouté
      };
      setToken(token);
      setAccount(fullAccount);
      localStorage.setItem('token', token);
      localStorage.setItem('account', JSON.stringify(fullAccount));
      console.log('Token sauvegardé dans localStorage:', localStorage.getItem('token'));
      console.log('Account sauvegardé dans localStorage:', localStorage.getItem('account'));
    } catch (error: any) {
      console.error('Erreur lors de la connexion:', error);
      throw error;
    }
  };

  const register = async (data: FormData & { role: 'user' | 'admin'; canInvite: boolean; isVerified: boolean; mfaEnabled: boolean }) => {
    const response = await axios.post('/auth/register', data);
    const { token, account: accountData } = response.data;
    console.log('Nouveau token reçu (register):', token);
    console.log('Données de l\'account reçues (register):', accountData);
    setToken(token);
    setAccount(accountData);
    localStorage.setItem('token', token);
    localStorage.setItem('account', JSON.stringify(accountData));
    console.log('Token sauvegardé dans localStorage (register):', localStorage.getItem('token'));
    console.log('Account sauvegardé dans localStorage (register):', localStorage.getItem('account'));
  };

  const logout = () => {
    console.log('Déconnexion: suppression du token et de l\'account');
    setToken(null);
    setAccount(null);
    localStorage.removeItem('token');
    localStorage.removeItem('account');
  };

  return (
    <AuthContext.Provider value={{ account, token, isLoggedIn, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within an AuthProvider");
  return context;
};