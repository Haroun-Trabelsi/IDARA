import React, { useState, useEffect, FC } from 'react';
import AdminLayout from './AdminLayout';
import AdminDashboardContent from './AdminDashboardContent';
import axios from 'axios';

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

const AdminDashboard: FC = () => {
  const [darkMode, setDarkMode] = useState<boolean>(true);
  const [selectedMenuItem, setSelectedMenuItem] = useState<string>('dashboard');
  const [emailSubject, setEmailSubject] = useState<string>('');
  const [emailMessage, setEmailMessage] = useState<string>('');
  const [dashboardData, setDashboardData] = useState<{
    totalOrganizations: number;
    activeProjects: number;
    totalUsers: number;
    organizationGrowthData: OrganizationGrowthData[];
    ratingDistribution: RatingDistribution[];
    organizationsList: Organization[];
  }>({
    totalOrganizations: 0,
    activeProjects: 0,
    totalUsers: 0,
    organizationGrowthData: [],
    ratingDistribution: [],
    organizationsList: []
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:8080/api/admin/dashboard', {
          headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
        });
        setDashboardData(response.data);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
      }
    };
    fetchData();
  }, []);

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