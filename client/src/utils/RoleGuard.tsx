import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from 'contexts/AuthContext';

interface RoleGuardProps {
  children: React.ReactNode;
  isAdminRoute?: boolean; // Paramètre optionnel pour indiquer si c'est une route admin
}

const RoleGuard: React.FC<RoleGuardProps> = ({ children, isAdminRoute = false }) => {
  const { account } = useAuth();
  const location = useLocation();

  // Si l'utilisateur n'est pas connecté, rediriger vers /login
  if (!account) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // Vérifier le rôle uniquement pour les routes admin
  if (isAdminRoute && account.role !== 'admin') {
    return <Navigate to="/" replace />; // Rediriger vers la page user par défaut si ce n'est pas un admin
  }

  // Si c'est une route user ou une route admin valide, autoriser l'accès
  return <>{children}</>;
};

export default RoleGuard;