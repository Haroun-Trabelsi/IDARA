import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from 'contexts/AuthContext';

interface AuthGuardProps {
  children: React.ReactNode;
}

const AuthGuard: React.FC<AuthGuardProps> = ({ children }) => {
  const { checkAuth } = useAuth();
  const location = useLocation();

  if (!checkAuth()) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <>{children}</>;
};

export default AuthGuard;