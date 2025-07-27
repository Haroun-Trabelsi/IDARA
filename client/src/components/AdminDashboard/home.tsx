import React, { useState, FC } from 'react';
import { 
  Search, 
  Bell, 
  User, 
  Sun, 
  Moon, 
  BarChart3, 
  Building2, 
  Users, 
  Mail, 
  Settings, 
  Clapperboard, 
  TrendingUp, 
  Plus, 
  Edit, 
  Trash2, 
  Send, 
  Eye,
  LucideIcon
} from 'lucide-react';

// Interfaces for data structures
interface Stat {
  title: string;
  value: number | string;
  icon: LucideIcon;
  color: string;
  change: string;
}

interface Organization {
  id: number;
  name: string;
  email: string;
  projects: number;
  users: number;
  status: 'active' | 'inactive';
  joinDate: string;
  lastActive: string;
}

interface MenuItem {
  id: string;
  label: string;
  icon: LucideIcon;
}

// Mock data
const mockStats: {
  totalOrganizations: number;
  activeProjects: number;
  totalUsers: number;
  monthlyGrowth: number;
} = {
  totalOrganizations: 147,
  activeProjects: 89,
  totalUsers: 1256,
  monthlyGrowth: 12.5
};

const mockOrganizations: Organization[] = [
  {
    id: 1,
    name: 'Pixar Animation Studios',
    email: 'contact@pixar.com',
    projects: 12,
    users: 245,
    status: 'active',
    joinDate: '2024-01-15',
    lastActive: '2024-07-20'
  },
  {
    id: 2,
    name: 'Industrial Light & Magic',
    email: 'info@ilm.com',
    projects: 8,
    users: 189,
    status: 'active',
    joinDate: '2024-02-03',
    lastActive: '2024-07-22'
  },
  {
    id: 3,
    name: 'Weta Digital',
    email: 'hello@wetadigital.com',
    projects: 5,
    users: 156,
    status: 'inactive',
    joinDate: '2024-03-12',
    lastActive: '2024-07-18'
  },
  {
    id: 4,
    name: 'MPC Studios',
    email: 'contact@mpc.com',
    projects: 15,
    users: 298,
    status: 'active',
    joinDate: '2024-01-28',
    lastActive: '2024-07-23'
  }
];

// TypeScript interfaces for styles
interface Styles {
  [key: string]: React.CSSProperties;
}

const styles: Styles = {
  container: {
    minHeight: '100vh',
    backgroundColor: '#0f172a',
    color: '#e2e8f0',
    fontFamily: 'Arial, sans-serif'
  },
  mainLayout: {
    display: 'flex',
    height: '100vh'
  },
  sidebar: {
    width: '280px',
    height: '100%',
    backgroundColor: '#1a202c',
    borderRight: '1px solid #2d3748',
    display: 'flex',
    flexDirection: 'column'
  },
  sidebarHeader: {
    padding: '24px',
    borderBottom: '1px solid #2d3748'
  },
  logo: {
    fontSize: '24px',
    fontWeight: 'bold',
    color: '#4299e1',
    margin: '0 0 4px 0'
  },
  logoSubtext: {
    fontSize: '14px',
    color: '#a0aec0',
    margin: 0
  },
  sidebarMenu: {
    flex: 1,
    paddingTop: '16px'
  },
  menuItem: {
    width: '100%',
    display: 'flex',
    alignItems: 'center',
    padding: '12px 24px',
    border: 'none',
    backgroundColor: 'transparent',
    color: 'inherit',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 500,
    transition: 'all 0.2s ease'
  },
  menuItemSelected: {
    backgroundColor: '#4299e1',
    color: '#ffffff'
  },
  menuItemHover: {
    backgroundColor: '#2d3748'
  },
  menuIcon: {
    marginRight: '12px',
    width: '20px',
    height: '20px'
  },
  mainContent: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden'
  },
  header: {
    height: '64px',
    backgroundColor: '#1a202c',
    borderBottom: '1px solid #2d3748',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0 24px'
  },
  headerTitle: {
    fontSize: '20px',
    fontWeight: 'bold',
    margin: 0
  },
  headerActions: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px'
  },
  searchContainer: {
    position: 'relative'
  },
  searchInput: {
    paddingLeft: '40px',
    paddingRight: '16px',
    paddingTop: '8px',
    paddingBottom: '8px',
    width: '250px',
    borderRadius: '8px',
    border: '1px solid #4a5568',
    backgroundColor: '#2d3748',
    color: '#e2e8f0',
    fontSize: '14px',
    outline: 'none'
  },
  searchIcon: {
    position: 'absolute',
    left: '12px',
    top: '50%',
    transform: 'translateY(-50%)',
    width: '16px',
    height: '16px',
    color: '#a0aec0'
  },
  toggleContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
  },
  toggle: {
    position: 'relative',
    width: '44px',
    height: '24px',
    backgroundColor: '#4299e1',
    borderRadius: '12px',
    border: 'none',
    cursor: 'pointer',
    transition: 'background-color 0.2s ease'
  },
  toggleThumb: {
    width: '16px',
    height: '16px',
    backgroundColor: '#ffffff',
    borderRadius: '50%',
    position: 'absolute',
    top: '4px',
    left: '24px',
    transition: 'left 0.2s ease'
  },
  notificationButton: {
    position: 'relative',
    padding: '8px',
    border: 'none',
    backgroundColor: 'transparent',
    borderRadius: '8px',
    cursor: 'pointer',
    color: 'inherit'
  },
  notificationBadge: {
    position: 'absolute',
    top: '0',
    right: '0',
    backgroundColor: '#ef4444',
    color: '#ffffff',
    borderRadius: '50%',
    width: '20px',
    height: '20px',
    fontSize: '12px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  },
  profileButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px',
    border: 'none',
    backgroundColor: 'transparent',
    borderRadius: '8px',
    cursor: 'pointer',
    color: 'inherit'
  },
  avatar: {
    width: '32px',
    height: '32px',
    backgroundColor: '#4299e1',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  },
  content: {
    flex: 1,
    overflow: 'auto',
    padding: '24px'
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '24px',
    marginBottom: '32px'
  },
  statCard: {
    backgroundColor: '#1a202c',
    border: '1px solid #2d3748',
    borderRadius: '12px',
    padding: '24px',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.3)'
  },
  statCardContent: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start'
  },
  statInfo: {
    flex: 1
  },
  statTitle: {
    fontSize: '14px',
    color: '#a0aec0',
    marginBottom: '4px'
  },
  statValue: {
    fontSize: '36px',
    fontWeight: 'bold',
    marginBottom: '8px'
  },
  statChange: {
    fontSize: '14px',
    color: '#10b981'
  },
  statIcon: {
    width: '48px',
    height: '48px',
    borderRadius: '8px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#ffffff'
  },
  tableCard: {
    backgroundColor: '#1a202c',
    border: '1px solid #2d3748',
    borderRadius: '12px',
    marginBottom: '32px',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.3)'
  },
  tableHeader: {
    padding: '24px',
    borderBottom: '1px solid #2d3748',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  tableTitle: {
    fontSize: '18px',
    fontWeight: 'bold',
    margin: 0
  },
  tableActions: {
    display: 'flex',
    gap: '12px'
  },
  button: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 16px',
    borderRadius: '8px',
    border: 'none',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 500,
    transition: 'all 0.2s ease'
  },
  primaryButton: {
    backgroundColor: '#4299e1',
    color: '#ffffff'
  },
  secondaryButton: {
    backgroundColor: 'transparent',
    color: 'inherit',
    border: '1px solid #4a5568'
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse'
  },
  tableHeadCell: {
    textAlign: 'left',
    padding: '16px 24px',
    fontWeight: 500,
    color: '#a0aec0',
    borderBottom: '1px solid #2d3748'
  },
  tableBodyCell: {
    padding: '16px 24px',
    borderBottom: '1px solid #2d3748'
  },
  tableRow: {
    cursor: 'pointer',
    transition: 'background-color 0.2s ease'
  },
  orgName: {
    fontWeight: 500,
    marginBottom: '4px'
  },
  orgEmail: {
    fontSize: '14px',
    color: '#a0aec0'
  },
  statusBadge: {
    padding: '4px 8px',
    borderRadius: '16px',
    fontSize: '12px',
    fontWeight: 500
  },
  activeStatus: {
    backgroundColor: '#065f46',
    color: '#34d399'
  },
  inactiveStatus: {
    backgroundColor: '#374151',
    color: '#d1d5db'
  },
  actionButtons: {
    display: 'flex',
    gap: '8px'
  },
  iconButton: {
    padding: '4px',
    border: 'none',
    backgroundColor: 'transparent',
    borderRadius: '4px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'background-color 0.2s ease'
  },
  emailCard: {
    backgroundColor: '#1a202c',
    border: '1px solid #2d3748',
    borderRadius: '12px',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.3)'
  },
  emailContent: {
    padding: '24px'
  },
  emailGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '24px'
  },
  formGroup: {
    marginBottom: '16px'
  },
  label: {
    display: 'block',
    fontSize: '14px',
    fontWeight: 500,
    marginBottom: '8px'
  },
  input: {
    width: '100%',
    padding: '8px 12px',
    borderRadius: '8px',
    border: '1px solid #4a5568',
    backgroundColor: '#2d3748',
    color: '#e2e8f0',
    fontSize: '14px',
    outline: 'none',
    boxSizing: 'border-box'
  },
  textarea: {
    width: '100%',
    padding: '8px 12px',
    borderRadius: '8px',
    border: '1px solid #4a5568',
    backgroundColor: '#2d3748',
    color: '#e2e8f0',
    fontSize: '14px',
    outline: 'none',
    resize: 'none',
    boxSizing: 'border-box',
    minHeight: '120px'
  },
  quickActions: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px'
  },
  quickActionTitle: {
    fontSize: '14px',
    fontWeight: 500,
    color: '#a0aec0',
    marginBottom: '16px'
  },
  fullWidthButton: {
    width: '100%',
    justifyContent: 'center',
    padding: '12px 16px'
  },
  emailStats: {
    marginTop: '24px',
    padding: '16px',
    backgroundColor: '#2d3748',
    borderRadius: '8px'
  },
  emailStatsTitle: {
    fontSize: '14px',
    fontWeight: 500,
    marginBottom: '12px'
  },
  statRow: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '8px',
    fontSize: '14px'
  },
  statLabel: {
    color: '#a0aec0'
  },
  statNumber: {
    fontWeight: 500
  },
  activityCard: {
    backgroundColor: '#1a202c',
    border: '1px solid #2d3748',
    borderRadius: '12px',
    marginTop: '32px',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.3)'
  },
  activityItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
    padding: '12px 0',
    borderBottom: '1px solid #2d3748'
  },
  activityDot: {
    width: '12px',
    height: '12px',
    borderRadius: '50%'
  },
  activityContent: {
    flex: 1
  },
  activityAction: {
    fontSize: '14px',
    fontWeight: 500,
    marginBottom: '2px'
  },
  activityOrg: {
    fontSize: '12px',
    color: '#a0aec0'
  },
  activityTime: {
    fontSize: '12px',
    color: '#a0aec0'
  }
};

const AdminDashboard: FC = () => {
  const [darkMode, setDarkMode] = useState<boolean>(true);
  const [selectedMenuItem, setSelectedMenuItem] = useState<string>('dashboard');
  const [emailSubject, setEmailSubject] = useState<string>('');
  const [emailMessage, setEmailMessage] = useState<string>('');

  const menuItems: MenuItem[] = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'organizations', label: 'Organizations', icon: Building2 },
    { id: 'users', label: 'Users', icon: Users },
    { id: 'projects', label: 'Projects', icon: Clapperboard },
    { id: 'emails', label: 'Email Center', icon: Mail },
    { id: 'settings', label: 'Settings', icon: Settings }
  ];

  const stats: Stat[] = [
    {
      title: 'Total Organizations',
      value: mockStats.totalOrganizations,
      icon: Building2,
      color: '#4299e1',
      change: '+12%'
    },
    {
      title: 'Active Projects',
      value: mockStats.activeProjects,
      icon: Clapperboard,
      color: '#10b981',
      change: '+8%'
    },
    {
      title: 'Total Users',
      value: mockStats.totalUsers,
      icon: Users,
      color: '#f59e0b',
      change: '+15%'
    },
    {
      title: 'Monthly Growth',
      value: `${mockStats.monthlyGrowth}%`,
      icon: TrendingUp,
      color: '#8b5cf6',
      change: '+2.1%'
    }
  ];





  // Sidebar Component
  const Sidebar: FC = () => (
    <div style={styles.sidebar}>
      <div style={styles.sidebarHeader}>
        <h1 style={styles.logo}>FTRAK Admin</h1>
        <p style={styles.logoSubtext}>VFX Management Platform</p>
      </div>
  
      <div style={styles.sidebarMenu}>
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isSelected = selectedMenuItem === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setSelectedMenuItem(item.id)}
              style={{
                ...styles.menuItem,
                ...(isSelected ? styles.menuItemSelected : {}),
              }}
              onMouseEnter={(e: React.MouseEvent<HTMLButtonElement>) => {
                if (!isSelected) {
                  e.currentTarget.style.backgroundColor =
                    styles.menuItemHover.backgroundColor ?? 'transparent';
                }
              }}
              onMouseLeave={(e: React.MouseEvent<HTMLButtonElement>) => {
                if (!isSelected) {
                  e.currentTarget.style.backgroundColor = 'transparent';
                }
              }}
            >
              <Icon style={styles.menuIcon} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );



  // Header Component
  const Header: FC = () => (
    <div style={styles.header}>
      <div>
        <h2 style={styles.headerTitle}>Admin Dashboard</h2>
      </div>

      <div style={styles.headerActions}>
        <div style={styles.searchContainer}>
          <Search style={styles.searchIcon} />
          <input
            type="text"
            placeholder="Search..."
            style={styles.searchInput}
          />
        </div>

        <div style={styles.toggleContainer}>
          <Moon size={16} />
          <button
            onClick={() => setDarkMode(!darkMode)}
            style={{
              ...styles.toggle,
              backgroundColor: darkMode ? '#4299e1' : '#d1d5db'
            }}
          >
            <div style={{
              ...styles.toggleThumb,
              left: darkMode ? '24px' : '4px'
            }} />
          </button>
          <Sun size={16} />
        </div>

        <button style={styles.notificationButton}>
          <Bell size={20} />
          <span style={styles.notificationBadge}>3</span>
        </button>

        <button style={styles.profileButton}>
          <div style={styles.avatar}>
            <User size={16} color="#ffffff" />
          </div>
        </button>
      </div>
    </div>
  );

  // Stats Cards Component
  const StatsCards: FC = () => (
    <div style={styles.statsGrid}>
      {stats.map((stat, index) => {
        const Icon = stat.icon;
        return (
          <div key={index} style={styles.statCard}>
            <div style={styles.statCardContent}>
              <div style={styles.statInfo}>
                <p style={styles.statTitle}>{stat.title}</p>
                <p style={styles.statValue}>{stat.value}</p>
                <p style={styles.statChange}>{stat.change} from last month</p>
              </div>
              <div style={{...styles.statIcon, backgroundColor: stat.color}}>
                <Icon size={24} />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );

  // Organizations Table Component
  const OrganizationsTable: FC = () => (
    <div style={styles.tableCard}>
      <div style={styles.tableHeader}>
        <h3 style={styles.tableTitle}>Organizations Management</h3>
        <div style={styles.tableActions}>
          <button style={{...styles.button, ...styles.primaryButton}}>
            <Mail size={16} />
            Send Email to All
          </button>
          <button style={{...styles.button, ...styles.secondaryButton}}>
            <Plus size={16} />
            Add Organization
          </button>
        </div>
      </div>

      <div style={{overflowX: 'auto'}}>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.tableHeadCell}>Organization</th>
              <th style={styles.tableHeadCell}>Projects</th>
              <th style={styles.tableHeadCell}>Users</th>
              <th style={styles.tableHeadCell}>Status</th>
              <th style={styles.tableHeadCell}>Last Active</th>
              <th style={styles.tableHeadCell}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {mockOrganizations.map((org) => (
              <tr 
                key={org.id} 
                style={styles.tableRow}
                onMouseEnter={(e: React.MouseEvent<HTMLTableRowElement>) => {
                  (e.currentTarget as HTMLTableRowElement).style.backgroundColor = darkMode ? '#2d3748' : '#f8fafc';
                }}
                onMouseLeave={(e: React.MouseEvent<HTMLTableRowElement>) => {
                  (e.currentTarget as HTMLTableRowElement).style.backgroundColor = 'transparent';
                }}
              >
                <td style={styles.tableBodyCell}>
                  <div>
                    <p style={styles.orgName}>{org.name}</p>
                    <p style={styles.orgEmail}>{org.email}</p>
                  </div>
                </td>
                <td style={styles.tableBodyCell}>{org.projects}</td>
                <td style={styles.tableBodyCell}>{org.users}</td>
                <td style={styles.tableBodyCell}>
                  <span
                    style={{
                      ...styles.statusBadge,
                      ...(org.status === 'active' ? styles.activeStatus : styles.inactiveStatus)
                    }}
                  >
                    {org.status}
                  </span>
                </td>
                <td style={styles.tableBodyCell}>{org.lastActive}</td>
                <td style={styles.tableBodyCell}>
                  <div style={styles.actionButtons}>
                    <button style={{...styles.iconButton, color: '#4299e1'}}>
                      <Eye size={16} />
                    </button>
                    <button style={{...styles.iconButton, color: '#4299e1'}}>
                      <Edit size={16} />
                    </button>
                    <button style={{...styles.iconButton, color: '#ef4444'}}>
                      <Trash2 size={16} />
                    </button>
                    <button style={{...styles.iconButton, color: '#10b981'}}>
                      <Send size={16} />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  // Email Center Component
  const EmailCenter: FC = () => (
    <div style={styles.emailCard}>
      <div style={styles.tableHeader}>
        <h3 style={styles.tableTitle}>Email Center</h3>
      </div>
      
      <div style={styles.emailContent}>
        <div style={styles.emailGrid}>
          <div>
            <div style={styles.formGroup}>
              <label style={styles.label}>Subject</label>
              <input
                type="text"
                value={emailSubject}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEmailSubject(e.target.value)}
                placeholder="Enter email subject..."
                style={styles.input}
              />
            </div>
            
            <div style={styles.formGroup}>
              <label style={styles.label}>Message</label>
              <textarea
                value={emailMessage}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setEmailMessage(e.target.value)}
                placeholder="Enter your message..."
                style={styles.textarea}
              />
            </div>
          </div>
          
          <div>
            <h4 style={styles.quickActionTitle}>Quick Actions</h4>
            <div style={styles.quickActions}>
              <button style={{...styles.button, ...styles.secondaryButton, ...styles.fullWidthButton}}>
                <Mail size={16} />
                Send to All Organizations
              </button>
              <button style={{...styles.button, ...styles.secondaryButton, ...styles.fullWidthButton}}>
                <Mail size={16} />
                Send to Active Only
              </button>
              <button style={{...styles.button, ...styles.secondaryButton, ...styles.fullWidthButton}}>
                <Mail size={16} />
                Send to Inactive Only
              </button>
              <button style={{...styles.button, ...styles.primaryButton, ...styles.fullWidthButton}}>
                <Send size={16} />
                Send Email
              </button>
            </div>
            
            <div style={styles.emailStats}>
              <h5 style={styles.emailStatsTitle}>Email Statistics</h5>
              <div style={styles.statRow}>
                <span style={styles.statLabel}>Emails sent today:</span>
                <span style={styles.statNumber}>24</span>
              </div>
              <div style={styles.statRow}>
                <span style={styles.statLabel}>Open rate:</span>
                <span style={{...styles.statNumber, color: '#10b981'}}>78.5%</span>
              </div>
              <div style={styles.statRow}>
                <span style={styles.statLabel}>Click rate:</span>
                <span style={{...styles.statNumber, color: '#4299e1'}}>12.3%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Recent Activity Component
  const RecentActivity: FC = () => (
    <div style={styles.activityCard}>
      <div style={styles.tableHeader}>
        <h3 style={styles.tableTitle}>Recent Activity</h3>
      </div>
      
      <div style={styles.emailContent}>
        <div>
          {[
            {
              action: 'New organization registered',
              org: 'Framestore Studios',
              time: '2 hours ago',
              color: '#10b981'
            },
            {
              action: 'Project completed',
              org: 'Pixar Animation Studios',
              time: '4 hours ago',
              color: '#4299e1'
            },
            {
              action: 'User limit exceeded',
              org: 'MPC Studios',
              time: '6 hours ago',
              color: '#f59e0b'
            },
            {
              action: 'Payment failed',
              org: 'Weta Digital',
              time: '1 day ago',
              color: '#ef4444'
            }
          ].map((activity, index) => (
            <div key={index} style={{...styles.activityItem, borderBottom: index === 3 ? 'none' : styles.activityItem.borderBottom}}>
              <div style={{...styles.activityDot, backgroundColor: activity.color}} />
              <div style={styles.activityContent}>
                <p style={styles.activityAction}>{activity.action}</p>
                <p style={styles.activityOrg}>{activity.org}</p>
              </div>
              <span style={styles.activityTime}>{activity.time}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  return (
    <div style={styles.container}>
      <div style={styles.mainLayout}>
        <Sidebar />
        <div style={styles.mainContent}>
          <Header />
          <div style={styles.content}>
            <StatsCards />
            <OrganizationsTable />
            <EmailCenter />
            <RecentActivity />
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;