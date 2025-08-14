
"use client";

import React, { FC, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import AdminLayout from './AdminLayout';
import AdminDashboardContent from './AdminDashboardContent';
import axios from 'axios';
import { useAuth } from '../../contexts/AuthContext';

// Interfaces
interface Organization {
  id: string;
  name: string;
  email: string;
  projects: number;
  users: number;
  status: 'active' | 'inactive';
  joinDate: string;
  lastActive: string;
  rating: number;
  feedbacks: { rating: number; feedbackText: string; featureSuggestions: string[]; createdAt: string }[];
}

interface OrganizationGrowthData {
  month: string;
  organizations: number;
  users: number;
}

interface RatingDistribution {
  rating: number;
  count: number;
  percentage: number;
}

interface ProjectMetric {
  organizationName: string;
  totalProjects: number;
  activeProjects: number;
  completedProjects: number;
  totalTasks: number;
}

const AdminDashboard: FC = () => {
  const [darkMode, setDarkMode] = useState<boolean>(true);
  const [selectedMenuItem, setSelectedMenuItem] = useState<string>('dashboard');
  const [emailSubject, setEmailSubject] = useState<string>('');
  const [emailMessage, setEmailMessage] = useState<string>('');
  const [dashboardData, setDashboardData] = useState<{
    totalOrganizations: number;
    totalProjects: number;
    totalUsers: number;
    organizationGrowthData: OrganizationGrowthData[];
    ratingDistribution: RatingDistribution[];
    organizationsList: Organization[];
    projectMetrics: ProjectMetric[];
  }>({
    totalOrganizations: 0,
    totalProjects: 0,
    totalUsers: 0,
    organizationGrowthData: [],
    ratingDistribution: [],
    organizationsList: [],
    projectMetrics: [],
  });
  const navigate = useNavigate();
  const { isLoggedIn } = useAuth();

  // Rediriger si non connecté
  useEffect(() => {
    if (!isLoggedIn) {
      navigate('/login', { replace: true });
    }
  }, [isLoggedIn, navigate]);

  // Charger les données du dashboard
  useEffect(() => {
    let isMounted = true;

    const fetchData = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token || !isMounted) {
          return;
        }
        const response = await axios.get('http://localhost:8080/api/admin/dashboard', {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (isMounted) {
          setDashboardData(response.data);
        }
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        if (isMounted) {
          navigate('/login', { replace: true });
        }
      }
    };

    fetchData();

    return () => {
      isMounted = false;
    };
  }, [navigate]);

  return (
    <AdminLayout
      darkMode={darkMode}
      setDarkMode={setDarkMode}
      selectedMenuItem={selectedMenuItem}
      setSelectedMenuItem={setSelectedMenuItem}
    >
      <AdminDashboardContent
        darkMode={darkMode}
        selectedMenuItem={selectedMenuItem}
        emailSubject={emailSubject}
        setEmailSubject={setEmailSubject}
        emailMessage={emailMessage}
        setEmailMessage={setEmailMessage}
        dashboardData={dashboardData}
      />
    </AdminLayout>
  );
};

export default AdminDashboard;
