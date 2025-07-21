"use client"

import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText,
  TextField,
  Button,
  Card,
  CardContent,
  Avatar,
  IconButton,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  CircularProgress
} from "@mui/material";
import { 
  Settings, 
  People,
  Business,
  Delete,
  PersonAdd,
  GetApp,
  InfoOutlined,
  Edit
} from "@mui/icons-material";
import axios from 'utils/axios';
import { useAuth } from 'contexts/AuthContext';

interface MenuItem {
  id: string;
  label: string;
  icon: React.ReactNode;
}

interface Member {
  _id: string;
  name: string;
  surname?: string;
  email: string;
  role?: 'Owner' | 'Admin' | 'Member' | 'Guest';
  status: "pending" | "accepted" | "expired | Administrator";
  invitedDate?: string;
  canInvite?: boolean;
  invitedBy?: string;
}

const menuItems: MenuItem[] = [
  { id: "settings", label: "Settings", icon: <Settings sx={{ color: "#4299e1" }} /> },
  { id: "members", label: "Members", icon: <People sx={{ color: "#4299e1" }} /> },
];

export default function OrganizationPage() {
  const { token, logout, account } = useAuth();
  const [activeMenu, setActiveMenu] = useState("settings");
  const [editMode, setEditMode] = useState({
    name: false,
    address: false
  });
  const [orgData, setOrgData] = useState({
    name: account?.organizationName || 'Organization Name',
    address: account?.organizationName || '',
    id: account?.organizationName || '1127060442655O'
  });
  
  const [members, setMembers] = useState<Member[]>([]);
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);
  const [selectedMember, setSelectedMember] = useState<Member | null>(null);
  const [inviteDialogOpen, setInviteDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [inviteEmail, setInviteEmail] = useState('');
  const [filterText, setFilterText] = useState('');
  const [membersTab, setMembersTab] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    console.log('useEffect triggered - Checking token and fetching members');
    console.log('Token from useAuth:', token);
    console.log('Account from useAuth:', account);
    if (!token && !localStorage.getItem('token')) {
      console.log('No token found, redirecting to /login');
      setError('No token available, please log in again');
      window.location.href = '/login';
      return;
    }
    if (account) {
      fetchMembers();
    }
  }, [token, account]);

  const fetchMembers = async () => {
    console.log('fetchMembers called');
    const authToken = token || localStorage.getItem('token');
    console.log('Using authToken:', authToken);
    if (!authToken) {
      console.log('No authToken available, aborting fetch');
      return;
    }
    try {
      console.log('Sending GET request to /col/accounts');
      const response = await axios.get('/col/accounts', { headers: { Authorization: `Bearer ${authToken}` } });
      console.log('API response data:', response.data);
      const enrichedMembers = response.data.map((m: any) => ({
        _id: m.id,
        name: m.name || '',
        surname: m.surname || (account && account._id === m.id ? account.surname : ''),
        email: m.email,
        status: m.status || 'pending',
        invitedDate: m.invitedDate || null,
        canInvite: m.canInvite || (account && account._id === m.id && account.canInvite),
        invitedBy: m.invitedBy || null,
      }));
      console.log('Enriched members:', enrichedMembers);
      setMembers(enrichedMembers);
      setLoading(false);
    } catch (err: any) {
      console.error('Error in fetchMembers:', err.response ? err.response.data : err.message);
      setError(err.response?.data?.message || 'Failed to fetch members');
      setLoading(false);
      if (err.response?.status === 401) {
        console.log('401 Unauthorized - Logging out');
        logout();
      }
    }
  };

  const handleOpenInviteDialog = () => {
    console.log('handleOpenInviteDialog called');
    if (!account?.canInvite || (account?.invitedBy && account?.invitedBy !== account?._id) || !account?._id) {
      console.log('No permission to invite members');
      setError('You do not have permission to invite collaborators.');
      return;
    }
    setInviteDialogOpen(true);
  };

  const handleInviteMember = async () => {
    console.log('handleInviteMember called with email:', inviteEmail);
    if (!inviteEmail.trim()) {
      console.log('Invalid email, setting error message');
      setMessage({ type: 'error', text: 'Please fill in the email' });
      return;
    }
    const authToken = token || localStorage.getItem('token');
    console.log('Using authToken for invite:', authToken);
    if (!authToken) {
      console.log('No authToken available, aborting invite');
      return;
    }
    try {
      console.log('Sending POST request to /col/invite-account');
      await axios.post('/col/invite-account', { email: inviteEmail, language: 'en' }, { headers: { Authorization: `Bearer ${authToken}` } });
      console.log('Invite successful');
      setMessage({ type: 'success', text: 'Invitation sent successfully' });
      setInviteDialogOpen(false);
      setInviteEmail('');
      fetchMembers();
    } catch (err: any) {
      console.error('Error in handleInviteMember:', err.response ? err.response.data : err.message);
      setMessage({ type: 'error', text: err.response?.data?.message || 'Failed to send invitation' });
      if (err.response?.status === 401) {
        console.log('401 Unauthorized - Logging out');
        logout();
      }
    }
    setTimeout(() => setMessage(null), 3000);
  };

  const handleRemoveMember = async () => {
    console.log('handleRemoveMember called for member:', selectedMember);
    if (!selectedMember) return;
    const authToken = token ;
    console.log('Using authToken for delete:', authToken);
    if (!authToken) {
      console.log('No authToken available, aborting delete');
      return;
    }
    try {
      console.log('Sending DELETE request to /col/accounts/', selectedMember._id);
      await axios.delete(`/col/accounts/${selectedMember._id}`, { headers: { Authorization: `Bearer ${authToken}` } });
      console.log('Member deleted successfully');
      setMessage({ type: 'success', text: 'Member deleted successfully' });
      setSelectedMember(null);
      setDeleteDialogOpen(false);
      fetchMembers();
    } catch (err: any) {
      console.error('Error in handleRemoveMember:', err.response ? err.response.data : err.message);
      setMessage({ type: 'error', text: err.response?.data?.message || 'Failed to delete member' });
      if (err.response?.status === 401) {
        console.log('401 Unauthorized - Logging out');
        logout();
      }
    }
    setTimeout(() => setMessage(null), 3000);
  };

  const handleExportCSV = () => {
    console.log('handleExportCSV called');
    const csvContent = [
      ['Name', 'Email', 'Status', 'Invited Date'],
      ...members.map(member => [
        member.name,
        member.email,
        member.status,
        member.invitedDate || '-'
      ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'organization_members.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const getStatusDisplay = (member: Member) => {
    if (account?._id && member._id === account._id && account.canInvite) return 'Owner';
    if (member.invitedBy && (!account?._id || member.invitedBy !== account._id)) return '-';
    switch (member.status) {
      case 'pending': return 'Invited';
      case 'accepted': return 'Invited';
      case 'expired | Administrator': return 'Expired/Admin';
      default: return member.status;
    }
  };

  const getStatusColor = (member: Member) => {
    if (account?._id && member._id === account._id && account.canInvite) return '#4299e1';
    if (member.invitedBy && (!account?._id || member.invitedBy !== account._id)) return '#6b7280';
    return {
      'pending': '#f59e0b',
      'accepted': '#10b981',
      'expired | Administrator': '#ef4444',
    }[member.status] || '#6b7280';
  };

  const getStatusIcon = (member: Member) => {
    if (account?._id && member._id === account._id && account.canInvite) return '👑';
    if (member.invitedBy && (!account?._id || member.invitedBy !== account._id)) return '-';
    return {
      'pending': '⏳',
      'accepted': '✓',
      'expired | Administrator': '⚠️',
    }[member.status] || '?';
  };

  const getAvatarInitials = (name: string, surname?: string) => {
    return `${name.charAt(0)}${surname?.charAt(0) || ''}`.toUpperCase();
  };

  const handleEditToggle = (field: 'name' | 'address') => {
    setEditMode(prev => ({ ...prev, [field]: !prev[field] }));
  };

  const handleSave = async (field: 'name' | 'address') => {
    const authToken = token || localStorage.getItem('token');
    if (!authToken) return;
    try {
      await axios.put('/col/organization', 
        { [field]: orgData[field] }, 
        { headers: { Authorization: `Bearer ${authToken}` } }
      );
      setMessage({ type: 'success', text: `${field.charAt(0).toUpperCase() + field.slice(1)} updated successfully` });
      setEditMode(prev => ({ ...prev, [field]: false }));
    } catch (err: any) {
      setMessage({ type: 'error', text: err.response?.data?.message || `Failed to update ${field}` });
    }
    setTimeout(() => setMessage(null), 3000);
  };

  const renderSettingsField = (label: string, value: string, field: 'name' | 'address' | 'id', editable: boolean = true) => (
    <Box sx={{ 
      display: 'flex', 
      alignItems: 'center', 
      py: 2, 
      px: 3,
      gap: 2,
      borderBottom: '1px solid #2d3748',
      flexWrap: 'wrap'
    }}>
      <Typography 
        variant="body2" 
        sx={{ 
          color: '#a0aec0', 
          minWidth: '180px', 
          fontSize: '0.9rem',
          fontWeight: 500
        }}
      >
        {label}
      </Typography>
      <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', gap: 2, maxWidth: 400 }}>
        {field === 'id' ? (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <Typography variant="body2" sx={{ color: '#e2e8f0', fontSize: '0.9rem' }}>
              {value}
            </Typography>
            <IconButton size="small" sx={{ color: '#4299e1' }}>
              <GetApp sx={{ fontSize: '1.1rem' }} />
            </IconButton>
          </Box>
        ) : (
          <>
            <TextField
              value={value}
              onChange={(e) => setOrgData(prev => ({ ...prev, [field]: e.target.value }))}
              disabled={!editMode[field]}
              variant="outlined"
              size="small"
              sx={{ 
                flexGrow: 1,
                '& .MuiOutlinedInput-root': { 
                  bgcolor: editMode[field] ? '#2d3748' : 'transparent',
                  color: '#e2e8f0',
                  borderRadius: '8px',
                  '& fieldset': { borderColor: '#4a5568' },
                  '&:hover fieldset': { borderColor: '#718096' }
                },
                '& .MuiInputLabel-root': { color: '#a0aec0', fontSize: '0.9rem' }
              }}
            />
            {editable && (
              <Box sx={{ display: 'flex', gap: 1 }}>
                <IconButton 
                  size="small" 
                  sx={{ color: '#4299e1' }} 
                  onClick={() => handleEditToggle(field)}
                >
                  <Edit sx={{ fontSize: '1.1rem' }} />
                </IconButton>
                {editMode[field] && (
                  <Button
                    variant="contained"
                    size="small"
                    onClick={() => handleSave(field)}
                    sx={{
                      bgcolor: '#4299e1',
                      '&:hover': { bgcolor: '#3182ce' },
                      fontSize: '0.85rem',
                      px: 2
                    }}
                  >
                    Save
                  </Button>
                )}
              </Box>
            )}
          </>
        )}
      </Box>
    </Box>
  );

  const renderContent = () => {
    switch (activeMenu) {
      case "settings":
        return (
          <Box sx={{ maxWidth: 800, mx: 'auto' }}>
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',
              mb: 4 
            }}>
              <Typography variant="h4" sx={{ color: "#e2e8f0", fontWeight: 600 }}>
                Organization Settings
              </Typography>
              <Avatar 
                sx={{ 
                  width: 48, 
                  height: 48, 
                  bgcolor: "transparent", 
                  border: "2px solid #2d3748",
                  display: { xs: 'none', sm: 'flex' }
                }}
              >
                <Business sx={{ fontSize: "24px", color: "#4299e1" }} />
              </Avatar>
            </Box>
            
            {message && (
              <Alert severity={message.type} sx={{ mb: 4, borderRadius: '8px' }}>
                {message.text}
              </Alert>
            )}

            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <CircularProgress sx={{ color: '#4299e1' }} />
              </Box>
            ) : (
              <>
                <Card sx={{ 
                  bgcolor: "#1a202c", 
                  border: "1px solid #2d3748", 
                  mb: 4, 
                  borderRadius: '12px',
                  boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
                }}>
                  <CardContent sx={{ p: 4 }}>
                    <Box sx={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      mb: 4, 
                      flexDirection: { xs: 'column', sm: 'row' },
                      gap: 2 
                    }}>
                      <Avatar 
                        sx={{ 
                          width: 56, 
                          height: 56, 
                          bgcolor: "transparent", 
                          border: "2px solid #2d3748",
                          display: { xs: 'flex', sm: 'none' }
                        }}
                      >
                        <Business sx={{ fontSize: "28px", color: "#4299e1" }} />
                      </Avatar>
                      <Typography variant="h6" sx={{ 
                        color: "#e2e8f0", 
                        fontWeight: 600,
                        textAlign: { xs: 'center', sm: 'left' }
                      }}>
                        {orgData.name}
                      </Typography>
                    </Box>

                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      {renderSettingsField('Organization Name', orgData.name, 'name')}
                      {renderSettingsField('Organization ID', orgData.id, 'id', false)}
                      {renderSettingsField('Organization Address', orgData.address, 'address')}
                    </Box>
                  </CardContent>
                </Card>

                <Card sx={{ 
                  bgcolor: "#1a202c", 
                  border: "1px solid #2d3748", 
                  mb: 4, 
                  borderRadius: '12px',
                  boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
                }}>
                  <CardContent sx={{ p: 4 }}>
                    <Typography variant="h6" sx={{ color: "#e2e8f0", mb: 3, fontWeight: 500 }}>
                      Project Settings
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      <Box sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        py: 2, 
                        px: 3,
                        borderBottom: '1px solid #2d3748',
                        flexWrap: 'wrap',
                        gap: 2
                      }}>
                        <Typography variant="body2" sx={{ 
                          color: '#a0aec0', 
                          minWidth: '180px', 
                          fontSize: '0.9rem',
                          fontWeight: 500 
                        }}>
                          Default Project Visibility (pro)
                        </Typography>
                        <Typography variant="body2" sx={{ 
                          color: '#e2e8f0', 
                          fontSize: '0.9rem',
                          flexGrow: 1,
                          maxWidth: 400
                        }}>
                          All organization members

                        </Typography>
                      </Box>
                      <Box sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        py: 2, 
                        px: 3,
                        flexWrap: 'wrap',
                        gap: 2
                      }}>
                        
                        <Typography variant="body2" sx={{ 
                          color: '#e2e8f0', 
                          fontSize: '0.9rem',
                          flexGrow: 1,
                          maxWidth: 400
                        }}>
                          All organization members
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>

                <Box sx={{ mt: 4, pt: 3, borderTop: '1px solid #2d3748', textAlign: 'center' }}>
                  <Button 
                    variant="outlined"
                    sx={{ 
                      color: '#e53e3e',
                      borderColor: '#e53e3e',
                      borderRadius: '8px',
                      fontSize: '0.9rem',
                      px: 3,
                      py: 1,
                      '&:hover': { 
                        borderColor: '#c53030',
                        bgcolor: 'rgba(229, 62, 62, 0.1)'
                      }
                    }}
                  >
                    Delete Organization
                  </Button>
                </Box>
              </>
            )}
          </Box>
        );

      case "members":
        console.log('Current membersTab:', membersTab);
        return (
          <Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h4" sx={{ color: "#e2e8f0", fontWeight: 600 }}>
                Organization Members
              </Typography>
              <Box sx={{ display: 'flex', gap: 2 }}>
                <Button 
                  variant="outlined"
                  startIcon={<GetApp />}
                  onClick={handleExportCSV}
                  sx={{ 
                    borderColor: '#4a5568',
                    color: '#e2e8f0',
                    '&:hover': { borderColor: '#718096' }
                  }}
                >
                  Export CSV
                </Button>
                <Button 
                  variant="contained"
                  startIcon={<PersonAdd />}
                  onClick={handleOpenInviteDialog}
                  disabled={!account?.canInvite || (account?.invitedBy && account?.invitedBy !== account?._id) || !account?._id}
                  sx={{ 
                    bgcolor: '#4299e1',
                    '&:hover': { bgcolor: '#3182ce' },
                    '&:disabled': { bgcolor: '#4a5568' }
                  }}
                >
                  Invite organization members
                </Button>
              </Box>
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Avatar 
                sx={{ 
                  width: 32, 
                  height: 32, 
                  bgcolor: "transparent", 
                  border: "1px solid #2d3748",
                  mr: 2
                }}
              >
                <Business sx={{ fontSize: "16px", color: "#4299e1" }} />
              </Avatar>
              <Typography variant="body1" sx={{ color: "#e2e8f0" }}>
                {orgData.name}
              </Typography>
            </Box>
            
            {message && (
              <Alert severity={message.type} sx={{ mb: 3 }}>
                {message.text}
              </Alert>
            )}

            {error && (
              <Alert severity="error" sx={{ mb: 3 }}>
                {error}
              </Alert>
            )}

            <Box sx={{ borderBottom: '1px solid #2d3748', mb: 3 }}>
              <Tabs 
                value={membersTab} 
                onChange={(_, newValue) => {
                  console.log('Tab changed to:', newValue);
                  setMembersTab(newValue);
                }}
                sx={{
                  '& .MuiTab-root': { 
                    color: '#a0aec0',
                    textTransform: 'none',
                    fontSize: '14px',
                    fontWeight: 500
                  },
                  '& .Mui-selected': { 
                    color: '#4299e1' 
                  },
                  '& .MuiTabs-indicator': { 
                    backgroundColor: '#4299e1' 
                  }
                }}
              >
                <Tab label="Members" />
                <Tab label="Pending invites" />
              </Tabs>
            </Box>

            <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
              <Box sx={{ flexGrow: 1 }} />
              <TextField
                placeholder="Search organization members"
                value={filterText}
                onChange={(e) => setFilterText(e.target.value)}
                size="small"
                sx={{ 
                  width: 300,
                  '& .MuiOutlinedInput-root': { 
                    bgcolor: '#2d3748',
                    color: '#e2e8f0',
                    '& fieldset': { borderColor: '#4a5568' },
                    '&:hover fieldset': { borderColor: '#718096' }
                  },
                  '& .MuiInputBase-input::placeholder': { color: '#a0aec0' }
                }}
              />
            </Box>

            <Card sx={{ bgcolor: "#1a202c", border: "1px solid #2d3748" }}>
              <CardContent sx={{ p: 0 }}>
                <TableContainer component={Paper} sx={{ bgcolor: 'transparent' }}>
                  <Table>
                    <TableHead>
                      <TableRow sx={{ borderBottom: '1px solid #2d3748' }}>
                        <TableCell sx={{ color: '#a0aec0', fontWeight: 600, borderBottom: '1px solid #2d3748', fontSize: '14px' }}>
                          Organization members ↑
                        </TableCell>
                        <TableCell sx={{ color: '#a0aec0', fontWeight: 600, borderBottom: '1px solid #2d3748', fontSize: '14px' }}>
                          Roles
                        </TableCell>
                        <TableCell sx={{ color: '#a0aec0', fontWeight: 600, borderBottom: '1px solid #2d3748', fontSize: '14px' }}>
                          Invited Date
                        </TableCell>
                        <TableCell sx={{ width: 50, borderBottom: '1px solid #2d3748' }}>
                        </TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {loading ? (
                        <TableRow>
                          <TableCell colSpan={4} sx={{ textAlign: 'center', py: 2 }}>
                            <Typography color="#a0aec0">Loading...</Typography>
                          </TableCell>
                        </TableRow>
                      ) : membersTab === 1 && !members.some(m => m.status === 'pending' && m.email.toLowerCase().includes(filterText.toLowerCase()) && m._id !== account?._id) ? (
                        <TableRow>
                          <TableCell colSpan={4} sx={{ textAlign: 'center', py: 4, color: '#a0aec0' }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                              <InfoOutlined sx={{ color: '#a0aec0' }} />
                              <Typography>No pending invitations</Typography>
                            </Box>
                          </TableCell>
                        </TableRow>
                      ) : (
                        members
                          .filter(member => member.email.toLowerCase().includes(filterText.toLowerCase()))
                          .filter(member => {
                            const isMemberTab = membersTab === 0;
                            const isPendingTab = membersTab === 1;
                            const isOwner = account?._id && member._id === account._id && account.canInvite;
                            const showInMembers = isMemberTab && (member.status === 'accepted' || isOwner);
                            const showInPending = isPendingTab && member.status === 'pending' && !isOwner;
                            console.log('Filtering member:', member, 'Show in Members:', showInMembers, 'Show in Pending:', showInPending);
                            return showInMembers || showInPending;
                          })
                          .map((member) => (
                            <TableRow key={member._id} sx={{ borderBottom: '1px solid #2d3748' }}>
                              <TableCell sx={{ borderBottom: '1px solid #2d3748', py: 2 }}>
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                  <Avatar 
                                    sx={{ 
                                      width: 32, 
                                      height: 32, 
                                      bgcolor: account?._id && member._id === account._id && account.canInvite ? "#4299e1" : "#38b2ac", 
                                      fontSize: "14px",
                                      mr: 2
                                    }}
                                  >
                                    {getAvatarInitials(member.name, member.surname)}
                                  </Avatar>
                                  <Box>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                      <Typography variant="body2" sx={{ color: '#e2e8f0', fontWeight: 500, fontSize: '14px' }}>
                                        {member._id === account?._id ? ` ${member.name} ${member.surname || ''} (You) ` : `${member.name} ${member.surname || ''}`}
                                      </Typography>
                                      <Chip 
                                        label={
                                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                            <span>{getStatusIcon(member)}</span>
                                            <span>{getStatusDisplay(member)}</span>
                                          </Box>
                                        }
                                        size="small"
                                        sx={{ 
                                          bgcolor: `${getStatusColor(member)}20`,
                                          color: getStatusColor(member),
                                          border: `1px solid ${getStatusColor(member)}40`,
                                          height: '20px',
                                          fontSize: '11px'
                                        }}
                                      />
                                    </Box>
                                    <Typography variant="caption" sx={{ color: '#a0aec0', fontSize: '12px' }}>
                                      {member.email}
                                    </Typography>
                                  </Box>
                                </Box>
                              </TableCell>
                              <TableCell sx={{ borderBottom: '1px solid #2d3748', color: '#a0aec0', fontSize: '14px' }}>
                                {account?._id && member._id === account._id && account.canInvite ? '-' : '-'}
                              </TableCell>
                              <TableCell sx={{ borderBottom: '1px solid #2d3748', color: '#a0aec0', fontSize: '14px' }}>
                                {member.invitedDate || '-'}
                              </TableCell>
                              <TableCell sx={{ borderBottom: '1px solid #2d3748' }}>
                                {account?._id && account._id !== member._id && account?.canInvite ? (
                                  <IconButton
                                    onClick={() => { setSelectedMember(member); setDeleteDialogOpen(true); }}
                                    sx={{ color: '#ef4444' }}
                                    size="small"
                                  >
                                    <Delete sx={{ fontSize: '18px' }} />
                                  </IconButton>
                                ) : null}
                              </TableCell>
                            </TableRow>
                          ))
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
                
                <Box sx={{ p: 2, borderTop: '1px solid #2d3748', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="caption" sx={{ color: '#a0aec0' }}>
                    Rows per page: 50 ▼
                  </Typography>
                  <Typography variant="caption" sx={{ color: '#a0aec0' }}>
                    {members.filter(m => m.email.toLowerCase().includes(filterText.toLowerCase())).filter(m => {
                      const isMemberTab = membersTab === 0;
                      const isPendingTab = membersTab === 1;
                      const isOwner = account?._id && m._id === account._id && account.canInvite;
                      const showInMembers = isMemberTab && (m.status === 'accepted' || isOwner);
                      const showInPending = isPendingTab && m.status === 'pending' && !isOwner;
                      return showInMembers || showInPending;
                    }).length} of {members.length}
                  </Typography>
                </Box>
              </CardContent>
            </Card>

            <Dialog 
              open={inviteDialogOpen} 
              onClose={() => setInviteDialogOpen(false)}
              maxWidth="sm"
              fullWidth
              PaperProps={{ sx: { bgcolor: '#1a202c', border: '1px solid #2d3748' } }}
            >
              <DialogTitle sx={{ color: '#e2e8f0', borderBottom: '1px solid #2d3748' }}>
                Invite New Member
              </DialogTitle>
              <DialogContent sx={{ pt: 3 }}>
                <TextField
                  autoFocus
                  label="Email Address"
                  type="email"
                  fullWidth
                  variant="outlined"
                  value={inviteEmail}
                  onChange={(e) => setInviteEmail(e.target.value)}
                  sx={{ 
                    mb: 2,
                    '& .MuiOutlinedInput-root': { 
                      bgcolor: '#2d3748',
                      color: '#e2e8f0'
                    },
                    '& .MuiInputLabel-root': { color: '#a0aec0' }
                  }}
                />
              </DialogContent>
              <DialogActions sx={{ p: 3, borderTop: '1px solid #2d3748' }}>
                <Button 
                  onClick={() => setInviteDialogOpen(false)}
                  sx={{ color: '#a0aec0' }}
                >
                  Cancel
                </Button>
                <Button 
                  onClick={handleInviteMember}
                  variant="contained"
                  disabled={!account?.canInvite || (account?.invitedBy && account?.invitedBy !== account?._id) || !account?._id}
                  sx={{ 
                    bgcolor: '#4299e1',
                    '&:hover': { bgcolor: '#3182ce' },
                    '&:disabled': { bgcolor: '#4a5568' }
                  }}
                >
                  Send Invitation
                </Button>
              </DialogActions>
            </Dialog>

            <Dialog
              open={deleteDialogOpen}
              onClose={() => setDeleteDialogOpen(false)}
              maxWidth="sm"
              fullWidth
              PaperProps={{ sx: { bgcolor: '#1a202c', border: '1px solid #2d3748' } }}
            >
              <DialogTitle sx={{ color: '#e2e8f0', borderBottom: '1px solid #2d3748', pb: 2 }}>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Confirm deletion
                </Typography>
                <Typography variant="body2" color="#a0aec0" sx={{ mt: 1 }}>
                  Are you sure you want to delete the member {selectedMember?.email}?
                </Typography>
              </DialogTitle>
              <DialogActions sx={{ borderTop: '1px solid #2d3748', pt: 2, px: 3, pb: 3 }}>
                <Button onClick={() => setDeleteDialogOpen(false)} sx={{ color: '#a0aec0', textTransform: 'none' }}>
                  Cancel
                </Button>
                <Button
                  onClick={handleRemoveMember}
                  variant="contained"
                  sx={{
                    bgcolor: '#ef4444',
                    '&:hover': { bgcolor: '#dc2626' },
                    textTransform: 'none',
                    px: 3,
                  }}
                >
                  Delete
                </Button>
              </DialogActions>
            </Dialog>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Box sx={{ 
      display: "flex", 
      minHeight: "100vh", 
      bgcolor: "#0f172a",
      flexDirection: { xs: 'column', sm: 'row' }
    }}>
      <Box
        sx={{
          width: { xs: '100%', sm: 280 },
          bgcolor: "#1a202c",
          borderRight: { sm: "1px solid #2d3748" },
          borderBottom: { xs: "1px solid #2d3748", sm: "none" },
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Box sx={{ p: 3, borderBottom: "1px solid #2d3748" }}>
          <Typography variant="h6" sx={{ color: "#e2e8f0", fontWeight: 600 }}>
            Organization Management
          </Typography>
        </Box>

        <List sx={{ flexGrow: 1, px: 1, py: 2 }}>
          {menuItems.map((item) => (
            <ListItem
              component="button"
              key={item.id}
              onClick={() => setActiveMenu(item.id)}
              sx={{
                borderRadius: 2,
                mb: 1,
                bgcolor: activeMenu === item.id ? "rgba(66, 153, 225, 0.1)" : "transparent",
                "&:hover": {
                  bgcolor: activeMenu === item.id ? "rgba(66, 153, 225, 0.2)" : "rgba(255, 255, 255, 0.05)",
                },
              }}
            >
              <ListItemIcon sx={{ minWidth: 40 }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.label}
                primaryTypographyProps={{
                  color: activeMenu === item.id ? "#4299e1" : "#e2e8f0",
                  fontWeight: activeMenu === item.id ? 600 : 400,
                  fontSize: "14px",
                }}
              />
            </ListItem>
          ))}
        </List>
      </Box>

      <Box sx={{ 
        flexGrow: 1, 
        p: { xs: 2, sm: 4 }, 
        overflow: "auto",
        bgcolor: "#0f172a"
      }}>
        {renderContent()}
      </Box>
    </Box>
  );
}