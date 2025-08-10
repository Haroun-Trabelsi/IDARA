
import React, { FC, useRef, Dispatch, SetStateAction } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell
} from 'recharts';
import { Eye, Edit, Trash2, Send, Star, Building2, Clapperboard, Users, Mail } from 'lucide-react';
import axios from 'axios';

// Interfaces
interface Stat {
  title: string;
  value: number | string;
  icon: React.ElementType;
  color: string;
  change: string;
}

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

interface AdminDashboardContentProps {
  darkMode: boolean;
  selectedMenuItem: string;
  emailSubject: string;
  setEmailSubject: Dispatch<SetStateAction<string>>;
  emailMessage: string;
  setEmailMessage: Dispatch<SetStateAction<string>>;
  dashboardData: {
    totalOrganizations: number;
    totalProjects: number;
    totalUsers: number;
    organizationGrowthData: OrganizationGrowthData[];
    ratingDistribution: RatingDistribution[];
    organizationsList: Organization[];
    projectMetrics: ProjectMetric[];
  };
}

const AdminDashboardContent: FC<AdminDashboardContentProps> = ({
  darkMode,
  
  emailSubject,
  setEmailSubject,
  emailMessage,
  setEmailMessage,
  dashboardData,
}) => {
  // Références pour le défilement fluide
  const dashboardRef = useRef<HTMLDivElement>(null);
  const organizationsRef = useRef<HTMLDivElement>(null);
  const projectsRef = useRef<HTMLDivElement>(null);
  const emailCenterRef = useRef<HTMLDivElement>(null);

  // Gestion du clic sur la barre latérale
 /* const handleMenuClick = (menuItem: string, ref: React.RefObject<HTMLDivElement | null>) => {
    if (ref.current) {
      ref.current.scrollIntoView({ behavior: 'smooth' });
    }
  };*/

  // Gestion de l'envoi d'email
  const handleSendEmail = async (subject: string, message: string, target: string) => {
    try {
      const response = await axios.post('http://localhost:8080/api/admin/send-email', {
        subject,
        message,
        target,
      }, {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` },
      });
      alert(response.data.message);
    } catch (error) {
      console.error('Error sending email:', error);
      alert('Failed to send email');
    }
  };

  // Calculer averageRating et totalReviews
  const averageRating = dashboardData.organizationsList.reduce((acc, org) => acc + org.rating, 0) / (dashboardData.organizationsList.length || 1);
  const totalReviews = dashboardData.organizationsList.reduce((acc, org) => acc + org.feedbacks.length, 0);

  const stats: Stat[] = [
    {
      title: 'Total Organizations',
      value: dashboardData.totalOrganizations,
      icon: Building2,
      color: '#4299e1',
      change: '+12%',
    },
    {
      title: 'Total Projects',
      value: dashboardData.totalProjects,
      icon: Clapperboard,
      color: '#10b981',
      change: '+8%',
    },
    {
      title: 'Total Users',
      value: dashboardData.totalUsers,
      icon: Users,
      color: '#f59e0b',
      change: '+15%',
    },
  ];

  const StarRating: FC<{ rating: number; size?: number }> = ({ rating, size = 16 }) => {
    const stars = [];
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 !== 0;
    for (let i = 0; i < 5; i++) {
      if (i < fullStars) stars.push(<Star key={i} size={size} fill="#f59e0b" color="#f59e0b" />);
      else if (i === fullStars && hasHalfStar) {
        stars.push(
          <div key={i} style={{ position: 'relative', display: 'inline-block' }}>
            <Star size={size} color="#374151" fill="#374151" />
            <div style={{ position: 'absolute', top: 0, left: 0, width: '50%', overflow: 'hidden' }}>
              <Star size={size} fill="#f59e0b" color="#f59e0b" />
            </div>
          </div>
        );
      } else stars.push(<Star key={i} size={size} color="#374151" fill="#374151" />);
    }
    return <div style={{ display: 'flex', gap: '2px' }}>{stars}</div>;
  };

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
              <div style={{ ...styles.statIcon, backgroundColor: stat.color }}>
                <Icon size={24} />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );

  const ChartsSection: FC = () => (
    <div style={styles.chartsGrid}>
      <div style={styles.chartCard}>
        <h3 style={styles.chartTitle}>Organization Growth</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={dashboardData.organizationGrowthData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="month" stroke="#a0aec0" />
            <YAxis stroke="#a0aec0" />
            <Tooltip contentStyle={{ backgroundColor: '#1a202c', border: '1px solid #2d3748', borderRadius: '8px', color: '#e2e8f0' }} />
            <Line type="monotone" dataKey="organizations" stroke="#4299e1" strokeWidth={3} dot={{ fill: '#4299e1', strokeWidth: 2, r: 6 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div style={styles.chartCard}>
        <h3 style={styles.chartTitle}>User Growth</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={dashboardData.organizationGrowthData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="month" stroke="#a0aec0" />
            <YAxis stroke="#a0aec0" />
            <Tooltip contentStyle={{ backgroundColor: '#1a202c', border: '1px solid #2d3748', borderRadius: '8px', color: '#e2e8f0' }} />
            <Bar dataKey="users" fill="#10b981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div style={styles.chartCard}>
        <h3 style={styles.chartTitle}>Rating Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={dashboardData.ratingDistribution}
              cx="50%"
              cy="50%"
              outerRadius={100}
              fill="#8884d8"
              dataKey="count"
            >
              {dashboardData.ratingDistribution.map((_, index) => (
                <Cell key={`cell-${index}`} fill={['#10b981', '#4299e1', '#f59e0b', '#ef4444', '#8b5cf6'][index % 5]} />
              ))}
            </Pie>
            <Tooltip contentStyle={{ backgroundColor: '#1a202c', border: '1px solid #2d3748', borderRadius: '8px', color: '#e2e8f0' }} />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const RatingDashboard: FC = () => (
    <div style={styles.ratingCard}>
      <h3 style={styles.chartTitle}>Rating Dashboard</h3>
      <div style={styles.ratingOverview}>
        <div style={styles.overallRating}>
          <div style={styles.ratingValue}>{averageRating.toFixed(1)}</div>
          <div style={styles.ratingText}>Average Rating</div>
          <div style={styles.starsContainer}>
            <StarRating rating={averageRating} size={24} />
          </div>
          <div style={styles.totalReviews}>Based on {totalReviews} reviews</div>
        </div>
        <div style={styles.ratingDistribution}>
          {dashboardData.ratingDistribution.map((item) => (
            <div key={item.rating} style={styles.ratingRow}>
              <div style={styles.ratingLabel}>
                <span>{item.rating}</span>
                <Star size={14} fill="#f59e0b" color="#f59e0b" />
              </div>
              <div style={styles.ratingBar}>
                <div style={{ ...styles.ratingBarFill, width: `${item.percentage}%` }} />
              </div>
              <div style={styles.ratingCount}>{item.count}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const OrganizationsTable: FC = () => (
    <div style={styles.tableCard}>
      <div style={styles.tableHeader}>
        <h3 style={styles.tableTitle}>Organization Management</h3>
        <div style={styles.tableActions}>
          <button style={{ ...styles.button, ...styles.primaryButton }} onClick={() => handleSendEmail(emailSubject, emailMessage, 'all')}>
            <Mail size={16} /> Send Email to All
          </button>
          <select
            style={{ ...styles.button, ...styles.secondaryButton, padding: '8px' }}
            onChange={(e) => handleSendEmail(emailSubject, emailMessage, e.target.value)}
          >
            <option value="">Select an organization</option>
            {dashboardData.organizationsList.map(org => (
              <option key={org.id} value={org.name}>{org.name}</option>
            ))}
          </select>
        </div>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.tableHeadCell}>Organization</th>
              <th style={styles.tableHeadCell}>Projects</th>
              <th style={styles.tableHeadCell}>Users</th>
              <th style={styles.tableHeadCell}>Rating</th>
              <th style={styles.tableHeadCell}>Status</th>
              <th style={styles.tableHeadCell}>Last Active</th>
              <th style={styles.tableHeadCell}>Feedbacks</th>
              <th style={styles.tableHeadCell}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {dashboardData.organizationsList.map((org) => (
              <tr key={org.id} style={styles.tableRow}
                onMouseEnter={(e) => (e.currentTarget as HTMLTableRowElement).style.backgroundColor = darkMode ? '#2d3748' : '#f8fafc'}
                onMouseLeave={(e) => (e.currentTarget as HTMLTableRowElement).style.backgroundColor = 'transparent'}
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
                  <div style={styles.orgRating}>
                    <StarRating rating={org.rating} size={14} />
                    <span style={styles.ratingNumber}>{org.rating.toFixed(1)}</span>
                  </div>
                </td>
                <td style={styles.tableBodyCell}>
                  <span style={{ ...styles.statusBadge, ...(org.status === 'active' ? styles.activeStatus : styles.inactiveStatus) }}>
                    {org.status === 'active' ? 'Active' : 'Inactive'}
                  </span>
                </td>
                <td style={styles.tableBodyCell}>{org.lastActive}</td>
                <td style={styles.tableBodyCell}>
                  {org.feedbacks.length ? (
                    <ul style={{ padding: 0, margin: 0, listStyle: 'none' }}>
                      {org.feedbacks.map((f, index) => (
                        <li key={index} style={{ marginBottom: '5px' }}>
                          <div><strong>Rating:</strong> {f.rating}/5</div>
                          <div><strong>Text:</strong> {f.feedbackText || 'No text'}</div>
                          <div><strong>Suggestions:</strong> {f.featureSuggestions.join(', ') || 'None'}</div>
                          <div><strong>Date:</strong> {f.createdAt}</div>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    'No feedbacks'
                  )}
                </td>
                <td style={styles.tableBodyCell}>
                  <div style={styles.actionButtons}>
                    <button style={{ ...styles.iconButton, color: '#4299e1' }}><Eye size={16} /></button>
                    <button style={{ ...styles.iconButton, color: '#4299e1' }}><Edit size={16} /></button>
                    <button style={{ ...styles.iconButton, color: '#ef4444' }}><Trash2 size={16} /></button>
                    <button style={{ ...styles.iconButton, color: '#10b981' }} onClick={() => handleSendEmail(emailSubject, emailMessage, org.name)}><Send size={16} /></button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  const ProjectMetricsTable: FC = () => (
    <div style={styles.tableCard}>
      <div style={styles.tableHeader}>
        <h3 style={styles.tableTitle}>Project Metrics by Organization</h3>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.tableHeadCell}>Organization</th>
              <th style={styles.tableHeadCell}>Total Projects</th>
              <th style={styles.tableHeadCell}>Active Projects</th>
              <th style={styles.tableHeadCell}>Completed Projects</th>
              <th style={styles.tableHeadCell}>Total Tasks</th>
            </tr>
          </thead>
          <tbody>
            {dashboardData.projectMetrics.map((metric, index) => (
              <tr key={index} style={styles.tableRow}
                onMouseEnter={(e) => (e.currentTarget as HTMLTableRowElement).style.backgroundColor = darkMode ? '#2d3748' : '#f8fafc'}
                onMouseLeave={(e) => (e.currentTarget as HTMLTableRowElement).style.backgroundColor = 'transparent'}
              >
                <td style={styles.tableBodyCell}>{metric.organizationName}</td>
                <td style={styles.tableBodyCell}>{metric.totalProjects}</td>
                <td style={styles.tableBodyCell}>{metric.activeProjects}</td>
                <td style={styles.tableBodyCell}>{metric.completedProjects}</td>
                <td style={styles.tableBodyCell}>{metric.totalTasks}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

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
                onChange={(e) => setEmailSubject(e.target.value)}
                placeholder="Enter email subject..."
                style={styles.input}
              />
            </div>
            <div style={styles.formGroup}>
              <label style={styles.label}>Message</label>
              <textarea
                value={emailMessage}
                onChange={(e) => setEmailMessage(e.target.value)}
                placeholder="Enter your message..."
                style={styles.textarea}
              />
            </div>
          </div>
          <div>
            <h4 style={styles.quickActionTitle}>Quick Actions</h4>
            <div style={styles.quickActions}>
              <button style={{ ...styles.button, ...styles.secondaryButton, ...styles.fullWidthButton }}
                onClick={() => handleSendEmail(emailSubject, emailMessage, 'all')}>
                <Mail size={16} /> Send to All Organizations
              </button>
              <select
                style={{ ...styles.button, ...styles.secondaryButton, ...styles.fullWidthButton, padding: '12px' }}
                onChange={(e) => handleSendEmail(emailSubject, emailMessage, e.target.value)}
              >
                <option value="">Select an organization</option>
                {dashboardData.organizationsList.map(org => (
                  <option key={org.id} value={org.name}>{org.name}</option>
                ))}
              </select>
              <button style={{ ...styles.button, ...styles.primaryButton, ...styles.fullWidthButton }}
                onClick={() => handleSendEmail(emailSubject, emailMessage, 'all')}>
                <Send size={16} /> Send Email
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
                <span style={{ ...styles.statNumber, color: '#10b981' }}>78.5%</span>
              </div>
              <div style={styles.statRow}>
                <span style={styles.statLabel}>Click rate:</span>
                <span style={{ ...styles.statNumber, color: '#4299e1' }}>12.3%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const styles: { [key: string]: React.CSSProperties } = {
    statsGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '20px',
      marginBottom: '20px',
    },
    statCard: {
      backgroundColor: '#2d3748',
      borderRadius: '8px',
      padding: '15px',
    },
    statCardContent: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    statInfo: {
      textAlign: 'left' as const,
    },
    statTitle: {
      fontSize: '14px',
      color: '#a0aec0',
      margin: 0,
    },
    statValue: {
      fontSize: '20px',
      fontWeight: 'bold',
      margin: '5px 0',
      color: '#e2e8f0',
    },
    statChange: {
      fontSize: '12px',
      color: '#10b981',
      margin: 0,
    },
    statIcon: {
      width: '40px',
      height: '40px',
      borderRadius: '50%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#ffffff',
    },
    chartsGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
      gap: '20px',
      marginBottom: '20px',
    },
    chartCard: {
      backgroundColor: '#2d3748',
      borderRadius: '8px',
      padding: '15px',
    },
    chartTitle: {
      fontSize: '16px',
      margin: '0 0 10px 0',
      color: '#e2e8f0',
    },
    ratingCard: {
      backgroundColor: '#2d3748',
      borderRadius: '8px',
      padding: '15px',
      marginBottom: '20px',
    },
    ratingOverview: {
      display: 'flex',
      gap: '20px',
    },
    overallRating: {
      textAlign: 'center' as const,
    },
    ratingValue: {
      fontSize: '36px',
      fontWeight: 'bold',
      color: '#e2e8f0',
      marginBottom: '10px',
    },
    ratingText: {
      fontSize: '14px',
      color: '#a0aec0',
      marginBottom: '10px',
    },
    starsContainer: {
      marginBottom: '10px',
    },
    totalReviews: {
      fontSize: '12px',
      color: '#a0aec0',
    },
    ratingDistribution: {
      flex: 1,
    },
    ratingRow: {
      display: 'flex',
      alignItems: 'center',
      marginBottom: '10px',
    },
    ratingLabel: {
      width: '50px',
      display: 'flex',
      alignItems: 'center',
      gap: '5px',
    },
    ratingBar: {
      flex: 1,
      height: '10px',
      backgroundColor: '#4a5568',
      borderRadius: '5px',
      margin: '0 10px',
    },
    ratingBarFill: {
      height: '100%',
      backgroundColor: '#10b981',
      borderRadius: '5px',
    },
    ratingCount: {
      width: '50px',
      textAlign: 'right' as const,
      color: '#e2e8f0',
    },
    tableCard: {
      backgroundColor: '#2d3748',
      borderRadius: '8px',
      padding: '15px',
      marginBottom: '20px',
    },
    tableHeader: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: '15px',
    },
    tableTitle: {
      fontSize: '16px',
      margin: 0,
      color: '#e2e8f0',
    },
    tableActions: {
      display: 'flex',
      gap: '10px',
    },
    button: {
      padding: '8px 16px',
      borderRadius: '8px',
      border: 'none',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center',
      gap: '5px',
    },
    primaryButton: {
      backgroundColor: '#4299e1',
      color: '#ffffff',
    },
    secondaryButton: {
      backgroundColor: '#4a5568',
      color: '#e2e8f0',
    },
    fullWidthButton: {
      width: '100%',
    },
    table: {
      width: '100%',
      borderCollapse: 'collapse' as const,
    },
    tableHeadCell: {
      padding: '10px',
      backgroundColor: '#4a5568',
      color: '#e2e8f0',
      textAlign: 'left' as const,
    },
    tableRow: {
      transition: 'background-color 0.2s',
    },
    tableBodyCell: {
      padding: '10px',
      borderBottom: '1px solid #2d3748',
    },
    orgName: {
      fontWeight: 'bold',
      margin: 0,
      color: '#e2e8f0',
    },
    orgEmail: {
      fontSize: '12px',
      color: '#a0aec0',
      margin: 0,
    },
    orgRating: {
      display: 'flex',
      alignItems: 'center',
      gap: '5px',
    },
    ratingNumber: {
      color: '#e2e8f0',
    },
    statusBadge: {
      padding: '4px 8px',
      borderRadius: '12px',
      fontSize: '12px',
    },
    activeStatus: {
      backgroundColor: '#10b981',
      color: '#ffffff',
    },
    inactiveStatus: {
      backgroundColor: '#ef4444',
      color: '#ffffff',
    },
    actionButtons: {
      display: 'flex',
      gap: '5px',
    },
    iconButton: {
      backgroundColor: 'transparent',
      border: 'none',
      cursor: 'pointer',
      padding: '5px',
    },
    emailCard: {
      backgroundColor: '#2d3748',
      borderRadius: '8px',
      padding: '15px',
      marginBottom: '20px',
    },
    emailContent: {
      padding: '15px 0',
    },
    emailGrid: {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: '20px',
    },
    formGroup: {
      marginBottom: '15px',
    },
    label: {
      display: 'block' as const,
      fontSize: '14px',
      color: '#a0aec0',
      marginBottom: '5px',
    },
    input: {
      width: '100%',
      padding: '8px',
      borderRadius: '8px',
      border: '1px solid #4a5568',
      backgroundColor: '#2d3748',
      color: '#e2e8f0',
    },
    textarea: {
      width: '100%',
      padding: '8px',
      borderRadius: '8px',
      border: '1px solid #4a5568',
      backgroundColor: '#2d3748',
      color: '#e2e8f0',
      height: '150px',
      resize: 'vertical' as const,
    },
    quickActionTitle: {
      fontSize: '16px',
      marginBottom: '10px',
      color: '#e2e8f0',
    },
    quickActions: {
      display: 'flex',
      flexDirection: 'column' as const,
      gap: '10px',
    },
    emailStats: {
      marginTop: '20px',
    },
    emailStatsTitle: {
      fontSize: '14px',
      color: '#a0aec0',
      marginBottom: '10px',
    },
    statRow: {
      display: 'flex',
      justifyContent: 'space-between',
      marginBottom: '5px',
    },
    statLabel: {
      fontSize: '12px',
      color: '#a0aec0',
    },
    statNumber: {
      fontSize: '12px',
      color: '#e2e8f0',
    },
    section: {
      marginBottom: '40px',
    },
  };

  return (
    <div>
      <div ref={dashboardRef} style={styles.section}>
        <h2 style={{ color: '#e2e8f0', marginBottom: '20px' }}>Dashboard</h2>
        <StatsCards />
        <ChartsSection />
        <RatingDashboard />
      </div>
      <div ref={organizationsRef} style={styles.section}>
        <h2 style={{ color: '#e2e8f0', marginBottom: '20px' }}>Organizations</h2>
        <OrganizationsTable />
      </div>
      <div ref={projectsRef} style={styles.section}>
        <h2 style={{ color: '#e2e8f0', marginBottom: '20px' }}>Projects</h2>
        <ProjectMetricsTable />
      </div>
      <div ref={emailCenterRef} style={styles.section}>
        <h2 style={{ color: '#e2e8f0', marginBottom: '20px' }}>Mail Center</h2>
        <EmailCenter />
      </div>
    </div>
  );
};

export default AdminDashboardContent;
