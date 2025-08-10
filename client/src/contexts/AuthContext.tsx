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
  status: "pending" | "accepted" | "expired" | "AdministratorOrganization";
  invitedBy?: string;
  canInvite: boolean;
  mustCompleteProfile: boolean;
  feedbackText?: string;
  featureSuggestions?: string[];
  rating?: number;
  teamSize: string;
  region: string;
}

export interface FormData {
  name?: string;
  surname?: string;
  organizationName: string;
  email: string;
  password: string;
  teamSize: string;
  region: string;
}

interface AuthContextType {
  account: Account | null;
  token: string | null;
  isLoggedIn: boolean;
  login: (data: LoginData) => Promise<void>;
  register: (data: FormData & { role: 'user' | 'admin'; canInvite: boolean; isVerified: boolean; mfaEnabled: boolean; status: string }) => Promise<void>;
  logout: () => void;
  updateAccount: (updatedAccount: Account) => void;
  checkAuth: () => boolean;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const initialToken = localStorage.getItem('token');
const initialAccount = initialToken ? (JSON.parse(localStorage.getItem('account') || '{}') as Account) : null;

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [account, setAccount] = useState<Account | null>(initialAccount);
  const [token, setToken] = useState<string | null>(initialToken);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const savedToken = localStorage.getItem('token');
    const savedAccount = localStorage.getItem('account');
    if (savedToken && savedAccount) {
      try {
        const parsedAccount = JSON.parse(savedAccount) as Account;
        setToken(savedToken);
        setAccount(parsedAccount);
      } catch (err) {
        console.error('Erreur lors de la restauration de l\'account:', err);
        localStorage.removeItem('token');
        localStorage.removeItem('account');
        setToken(null);
        setAccount(null);
      }
    }
    setIsLoading(false);
  }, []);

  const isLoggedIn = !!account && !!token;

  const login = async (data: LoginData) => {
    try {
      const response = await axios.post('/auth/login', { email: data.email, password: data.password });
      const { token, data: accountData } = response.data;
      const fullAccount: Account = {
        ...accountData,
        role: accountData.role || 'user',
        isVerified: accountData.isVerified || false,
        mfaEnabled: accountData.mfaEnabled || false,
        organizationName: accountData.organizationName || '',
        canInvite: accountData.canInvite || false,
        mustCompleteProfile: accountData.mustCompleteProfile || false,
        teamSize: accountData.teamSize || '',
        region: accountData.region || '',
      };
      setToken(token);
      setAccount(fullAccount);
      localStorage.setItem('token', token);
      localStorage.setItem('account', JSON.stringify(fullAccount));
    } catch (error: any) {
      console.error('Erreur lors de la connexion:', error);
      throw error;
    }
  };

  const register = async (data: FormData & { role: 'user' | 'admin'; canInvite: boolean; isVerified: boolean; mfaEnabled: boolean; status: string }) => {
    try {
      const response = await axios.post('/auth/register', data);
      const { token, data: accountData } = response.data;
      const fullAccount: Account = {
        ...accountData,
        email: data.email,
        role: data.role || 'user',
        isVerified: data.isVerified || false,
        mfaEnabled: data.mfaEnabled || false,
        organizationName: data.organizationName || '',
        canInvite: data.canInvite || false,
        mustCompleteProfile: false,
        teamSize: data.teamSize || '',
        region: data.region || '',
      };
      setToken(token);
      setAccount(fullAccount);
      localStorage.setItem('token', token);
      localStorage.setItem('account', JSON.stringify(fullAccount));
    } catch (error: any) {
      console.error('Erreur lors de l\'inscription:', error);
      throw error;
    }
  };

  const logout = () => {
    setToken(null);
    setAccount(null);
    localStorage.removeItem('token');
    localStorage.removeItem('account');
  };

  const updateAccount = (updatedAccount: Account) => {
    setAccount(updatedAccount);
    localStorage.setItem('account', JSON.stringify(updatedAccount));
  };

  const checkAuth = (): boolean => {
    return isLoggedIn;
  };

  return (
    <AuthContext.Provider value={{ account, token, isLoggedIn, login, register, logout, updateAccount, checkAuth, isLoading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within an AuthProvider");
  return context;
};