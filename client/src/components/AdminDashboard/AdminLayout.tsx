"use client";

import React, { FC, useState } from 'react';
import { Search, Moon, Sun, User, BarChart3, Building2, Clapperboard, Mail, LogOut } from 'lucide-react';
import { CSSProperties } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';

interface AdminLayoutProps {
  darkMode: boolean;
  setDarkMode: (value: boolean) => void;
  selectedMenuItem: string;
  setSelectedMenuItem: (value: string) => void;
  children: React.ReactNode;
}

const AdminLayout: FC<AdminLayoutProps> = ({ darkMode, setDarkMode, selectedMenuItem, setSelectedMenuItem, children }) => {
  const [isMenuOpen, setIsMenuOpen] = useState<boolean>(false);
  const navigate = useNavigate();
  const { logout } = useAuth();

  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'organizations', label: 'Organizations', icon: Building2 },
    { id: 'projects', label: 'Projects', icon: Clapperboard },
    { id: 'emails', label: 'Email Center', icon: Mail },
  ];

  const handleLogout = () => {
    logout(); // Utiliser la fonction logout de AuthContext
    setIsMenuOpen(false);
    navigate('/login', { replace: true }); // Rediriger avec replace pour éviter l'historique
  };

  const styles: { [key: string]: CSSProperties } = {
    container: {
      display: 'flex',
      height: '100vh',
      backgroundColor: '#1a202c',
      color: '#e2e8f0',
      fontFamily: 'Arial, sans-serif',
    },
    mainLayout: {
      display: 'flex',
      width: '100%',
    },
    sidebar: {
      width: '250px',
      backgroundColor: '#2d3748',
      padding: '20px',
      display: 'flex',
      flexDirection: 'column' as const,
      height: '100vh',
    },
    sidebarHeader: {
      marginBottom: '40px',
    },
    logo: {
      fontSize: '24px',
      fontWeight: 'bold',
      color: '#4299e1',
      margin: 0,
    },
    logoSubtext: {
      fontSize: '12px',
      color: '#a0aec0',
      margin: 0,
    },
    sidebarMenu: {
      display: 'flex',
      flexDirection: 'column' as const,
      gap: '10px',
    },
    menuItem: {
      display: 'flex',
      alignItems: 'center',
      padding: '10px',
      color: '#a0aec0',
      borderRadius: '8px',
      cursor: 'pointer',
      backgroundColor: 'transparent',
      border: 'none',
      width: '100%',
      textAlign: 'left' as const,
    },
    menuItemSelected: {
      backgroundColor: '#4299e1',
      color: '#ffffff',
    },
    menuItemHover: {
      backgroundColor: '#4a5568',
    },
    menuIcon: {
      marginRight: '10px',
    },
    header: {
      padding: '20px',
      borderBottom: '1px solid #2d3748',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    headerTitle: {
      fontSize: '20px',
      margin: 0,
    },
    headerActions: {
      display: 'flex',
      alignItems: 'center',
      gap: '15px',
    },
    searchContainer: {
      display: 'flex',
      alignItems: 'center',
      backgroundColor: '#2d3748',
      borderRadius: '8px',
      padding: '5px',
    },
    searchIcon: {
      color: '#a0aec0',
      marginRight: '5px',
    },
    searchInput: {
      border: 'none',
      backgroundColor: 'transparent',
      color: '#e2e8f0',
      outline: 'none',
      width: '150px',
    },
    toggleContainer: {
      display: 'flex',
      alignItems: 'center',
      gap: '10px',
    },
    toggle: {
      width: '48px',
      height: '24px',
      borderRadius: '12px',
      backgroundColor: '#d1d5db',
      position: 'relative' as const,
      border: 'none',
      cursor: 'pointer',
      padding: 0,
    },
    toggleThumb: {
      width: '20px',
      height: '20px',
      backgroundColor: '#ffffff',
      borderRadius: '50%',
      position: 'absolute' as const,
      top: '2px',
      transition: 'left 0.2s',
      left: darkMode ? '24px' : '4px',
    },
    profileButton: {
      backgroundColor: 'transparent',
      border: 'none',
      cursor: 'pointer',
      position: 'relative' as const,
    },
    avatar: {
      width: '32px',
      height: '32px',
      backgroundColor: '#4299e1',
      borderRadius: '50%',
      display: 'flex' as const,
      alignItems: 'center' as const,
      justifyContent: 'center' as const,
    },
    dropdownMenu: {
      position: 'absolute' as const,
      top: '40px',
      right: '0',
      backgroundColor: '#2d3748',
      borderRadius: '8px',
      boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
      zIndex: 1000,
      width: '150px',
    },
    dropdownItem: {
      display: 'flex',
      alignItems: 'center',
      padding: '10px',
      color: '#e2e8f0',
      cursor: 'pointer',
      borderRadius: '4px',
      backgroundColor: 'transparent',
      border: 'none',
      width: '100%',
      textAlign: 'left' as const,
    },
    dropdownItemHover: {
      backgroundColor: '#4a5568',
    },
    mainContent: {
      flex: 1,
      display: 'flex',
      flexDirection: 'column' as const,
    },
    content: {
      padding: '20px',
    },
  };

  return (
    <div style={styles.container}>
      <div style={styles.mainLayout}>
        <div style={styles.sidebar}>
          <div style={styles.sidebarHeader}>
            <h1 style={styles.logo}>FTRAK Admin</h1>
            <p style={styles.logoSubtext}>VFX Management Platform</p>
          </div>
          <div style={styles.sidebarMenu}>
            {menuItems.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  onClick={() => setSelectedMenuItem(item.id)}
                  style={{
                    ...styles.menuItem,
                    ...(selectedMenuItem === item.id ? styles.menuItemSelected : {}),
                  }}
                  onMouseEnter={(e) => {
                    if (selectedMenuItem !== item.id) e.currentTarget.style.backgroundColor = styles.menuItemHover.backgroundColor || '#4a5568';
                  }}
                  onMouseLeave={(e) => {
                    if (selectedMenuItem !== item.id) e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                >
                  <Icon style={styles.menuIcon} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </div>
        </div>
        <div style={styles.mainContent}>
          <div style={styles.header}>
            <h2 style={styles.headerTitle}>Admin Dashboard</h2>
            <div style={styles.headerActions}>
              <div style={styles.searchContainer}>
                <Search style={styles.searchIcon} />
                <input type="text" placeholder="Search..." style={styles.searchInput} />
              </div>
              <div style={styles.toggleContainer}>
                <Moon size={16} />
                <button onClick={() => setDarkMode(!darkMode)} style={styles.toggle}>
                  <div style={styles.toggleThumb} />
                </button>
                <Sun size={16} />
              </div>
              <div style={styles.profileButton}>
                <button onClick={() => setIsMenuOpen(!isMenuOpen)} style={styles.profileButton}>
                  <div style={styles.avatar}>
                    <User size={16} color="#ffffff" />
                  </div>
                </button>
                {isMenuOpen && (
                  <div style={styles.dropdownMenu}>
                    <button
                      onClick={handleLogout}
                      style={styles.dropdownItem}
                      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = styles.dropdownItemHover.backgroundColor || '#4a5568')}
                      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                    >
                      <LogOut size={16} style={{ marginRight: '10px' }} />
                      <span>Logout</span>
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
          <div style={styles.content}>{children}</div>
        </div>
      </div>
    </div>
  );
};

export default AdminLayout;