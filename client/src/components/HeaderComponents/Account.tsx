"use client";

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
  Alert
} from "@mui/material";
import { 
  AccountCircle, 
  Notifications, 
  PlayArrow, 
  Security,
  Edit,
  Cancel
} from "@mui/icons-material";
import { useAuth } from 'contexts/AuthContext';
import axios from 'utils/axios';
import { validateProfile, validatePassword } from 'utils/validation';

interface MenuItem {
  id: string;
  label: string;
  icon: React.ReactNode;
}

const menuItems: MenuItem[] = [
  { id: "account", label: "Account", icon: <AccountCircle sx={{ color: "#4299e1" }} /> },
  { id: "notifications", label: "Notifications", icon: <Notifications sx={{ color: "#4299e1" }} /> },
  { id: "player", label: "Player", icon: <PlayArrow sx={{ color: "#4299e1" }} /> },
  { id: "security", label: "Security Settings", icon: <Security sx={{ color: "#4299e1" }} /> },
];

export default function AccountPage() {
  const { account, token, updateAccount } = useAuth();
  const [activeMenu, setActiveMenu] = useState("account");
  const [editMode, setEditMode] = useState(false);
  const [profileData, setProfileData] = useState({
    name: account?.name || '',
    surname: account?.surname || '',
    email: account?.email || ''
  });
  const [profileErrors, setProfileErrors] = useState({ name: '', surname: '', email: '' });
  const [passwordData, setPasswordData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: ''
  });
  const [passwordErrors, setPasswordErrors] = useState({ currentPassword: '', newPassword: '', confirmPassword: '' });
  // notificationSettings et playerSettings conservés pour une future implémentation
  // À implémenter dans l'onglet "Notifications"
  // À implémenter dans l'onglet "Player"
  const [securitySettings, setSecuritySettings] = useState({
    mfaEnabled: !!account?.mfaEnabled,
    sessionTimeout: 30,
    passwordExpiry: 90
  });
  const [mfaQrCode, setMfaQrCode] = useState<string | null>(null);
  const [mfaSecret, setMfaSecret] = useState<string | null>(null);
  const [mfaCode, setMfaCode] = useState('');
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);

  useEffect(() => {
    setProfileData({
      name: account?.name || '',
      surname: account?.surname || '',
      email: account?.email || ''
    });
    setSecuritySettings(prev => ({
      ...prev,
      mfaEnabled: !!account?.mfaEnabled
    }));
  }, [account]);

  const handleProfileChange = (field: string, value: string) => {
    setProfileData(prev => ({ ...prev, [field]: value }));
    // Supprimé la validation en temps réel
    setProfileErrors({ name: '', surname: '', email: '' }); // Réinitialise les erreurs
  };

  const handlePasswordChange = (field: string, value: string) => {
    setPasswordData(prev => ({ ...prev, [field]: value }));
    // Supprimé la validation en temps réel
    setPasswordErrors({ currentPassword: '', newPassword: '', confirmPassword: '' }); // Réinitialise les erreurs
  };

  const handleSaveProfile = async () => {
    const { isValid, errors } = validateProfile(profileData);
    setProfileErrors(errors);
    if (!isValid || !token) {
      setMessage({ type: 'error', text: !token ? 'You must be logged in to save changes' : 'Please fix the errors' });
      return;
    }
    try {
      const response = await axios.put('/auth/edit-profile', profileData, {
        headers: { Authorization: `Bearer ${token}` }
      });
      updateAccount(response.data.account);
      setEditMode(false);
      setProfileErrors({ name: '', surname: '', email: '' }); // Réinitialise les erreurs après succès
      setMessage({ type: 'success', text: 'Profile updated successfully!' });
      setTimeout(() => setMessage(null), 3000);
    } catch (err: any) {
      const message = err.response?.data?.message || 'Failed to update profile';
      setMessage({ type: 'error', text: message });
      console.error('Profile update error:', err);
    }
  };

  const handleChangePassword = async () => {
    const { isValid, errors } = validatePassword(passwordData);
    setPasswordErrors(errors);
    if (!isValid || !token) {
      setMessage({ type: 'error', text: !token ? 'You must be logged in to change password' : 'Please fix the errors' });
      return;
    }
    try {
      await axios.put('/auth/edit-profile/password', {
        currentPassword: passwordData.currentPassword,
        newPassword: passwordData.newPassword,
        confirmPassword: passwordData.confirmPassword // Ajout de confirmPassword
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setPasswordData({ currentPassword: '', newPassword: '', confirmPassword: '' });
      setPasswordErrors({ currentPassword: '', newPassword: '', confirmPassword: '' }); // Réinitialise les erreurs
      setMessage({ type: 'success', text: 'Password changed successfully!' });
      setTimeout(() => setMessage(null), 3000);
    } catch (err: any) {
      const errorMessage = err.response?.data?.message || 'Failed to change password';
      setMessage({ type: 'error', text: errorMessage });
      console.error('Password change error:', err.response?.data || err);
    }
  };

  const handleEnableMFA = async () => {
    if (!token) {
      setMessage({ type: 'error', text: 'You must be logged in to enable MFA' });
      return;
    }
    try {
      const response = await fetch('http://localhost:8080/auth/mfa/setup', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ email: profileData.email })
      });
      if (!response.ok) {
        throw new Error(`Failed to setup MFA: ${response.status}`);
      }
      const data = await response.json();
      setMfaQrCode(data.qrCode);
      setMfaSecret(data.secret);
      setMessage(null);
    } catch (err) {
      console.error('❌ MFA setup error:', err);
      setMessage({ type: 'error', text: err instanceof Error ? err.message : 'Failed to setup MFA' });
    }
  };

  const handleVerifyMFA = async () => {
    if (!token) {
      setMessage({ type: 'error', text: 'You must be logged in to verify MFA' });
      return;
    }
    try {
      const response = await fetch('http://localhost:8080/auth/mfa/verify', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ email: profileData.email, code: mfaCode, secret: mfaSecret })
      });
      if (!response.ok) {
        throw new Error(`Failed to verify MFA: ${response.status}`);
      }
      const data = await response.json();
      updateAccount(data.account);
      setSecuritySettings(prev => ({ ...prev, mfaEnabled: true }));
      setMfaQrCode(null);
      setMfaSecret(null);
      setMfaCode('');
      setMessage({ type: 'success', text: 'MFA enabled successfully!' });
      setTimeout(() => setMessage(null), 3000);
    } catch (err) {
      console.error('❌ MFA verify error:', err);
      setMessage({ type: 'error', text: err instanceof Error ? err.message : 'Invalid MFA code' });
    }
  };

  const renderContent = () => {
    switch (activeMenu) {
      case "account":
        return (
          <Box>
            <Typography variant="h5" sx={{ mb: 3, color: "#e2e8f0", fontWeight: 600 }}>
              Account Settings
            </Typography>
            
            {message && (
              <Alert severity={message.type} sx={{ mb: 3 }}>
                {message.text}
              </Alert>
            )}

            <Card sx={{ bgcolor: "#1a202c", border: "1px solid #2d3748", mb: 3 }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Avatar 
                    sx={{ 
                      width: 80, 
                      height: 80, 
                      bgcolor: "#4299e1", 
                      fontSize: "32px",
                      mr: 3
                    }}
                  >
                    {(profileData.name || '')[0]?.toUpperCase() || ''} {/* Vérification pour éviter undefined */}
                  </Avatar>
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" sx={{ color: "#e2e8f0", mb: 1 }}>
                      {`${profileData.name || ''} ${profileData.surname || ''}`}
                    </Typography>
                    <Typography variant="body2" sx={{ color: "#a0aec0" }}>
                      {profileData.email || ''}
                    </Typography>
                  </Box>
                  <IconButton 
                    onClick={() => setEditMode(!editMode)}
                    sx={{ color: "#4299e1" }}
                  >
                    {editMode ? <Cancel /> : <Edit />}
                  </IconButton>
                </Box>

                <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2 }}>
                  <TextField
                    label="First Name"
                    value={profileData.name}
                    onChange={(e) => handleProfileChange('name', e.target.value)}
                    disabled={!editMode}
                    error={!!profileErrors.name}
                    helperText={profileErrors.name}
                    sx={{ 
                      '& .MuiOutlinedInput-root': { 
                        bgcolor: editMode ? '#2d3748' : 'transparent',
                        color: '#e2e8f0'
                      },
                      '& .MuiInputLabel-root': { color: '#a0aec0' }
                    }}
                  />
                  <TextField
                    label="Last Name"
                    value={profileData.surname}
                    onChange={(e) => handleProfileChange('surname', e.target.value)}
                    disabled={!editMode}
                    error={!!profileErrors.surname}
                    helperText={profileErrors.surname}
                    sx={{ 
                      '& .MuiOutlinedInput-root': { 
                        bgcolor: editMode ? '#2d3748' : 'transparent',
                        color: '#e2e8f0'
                      },
                      '& .MuiInputLabel-root': { color: '#a0aec0' }
                    }}
                  />
                  <TextField
                    label="Email"
                    value={profileData.email}
                    onChange={(e) => handleProfileChange('email', e.target.value)}
                    disabled={!editMode}
                    error={!!profileErrors.email}
                    helperText={profileErrors.email}
                    sx={{ 
                      gridColumn: 'span 2',
                      '& .MuiOutlinedInput-root': { 
                        bgcolor: editMode ? '#2d3748' : 'transparent',
                        color: '#e2e8f0'
                      },
                      '& .MuiInputLabel-root': { color: '#a0aec0' }
                    }}
                  />
                </Box>

                {editMode && (
                  <Box sx={{ display: 'flex', gap: 2, mt: 3, justifyContent: 'flex-end' }}>
                    <Button 
                      variant="outlined" 
                      onClick={() => setEditMode(false)}
                      sx={{ 
                        borderColor: '#4a5568',
                        color: '#a0aec0',
                        '&:hover': { borderColor: '#718096' }
                      }}
                    >
                      Cancel
                    </Button>
                    <Button 
                      variant="contained" 
                      onClick={handleSaveProfile}
                      sx={{ 
                        bgcolor: '#4299e1',
                        '&:hover': { bgcolor: '#3182ce' }
                      }}
                    >
                      Save Changes
                    </Button>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Box>
        );

      case "security":
        return (
          <Box>
            <Typography variant="h5" sx={{ mb: 3, color: "#e2e8f0", fontWeight: 600 }}>
              Security Settings
            </Typography>
            
            {message && (
              <Alert severity={message.type} sx={{ mb: 3 }}>
                {message.text}
              </Alert>
            )}

            <Card sx={{ bgcolor: "#1a202c", border: "1px solid #2d3748", mb: 3 }}>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2, color: "#e2e8f0" }}>
                  Change Password
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <TextField
                    label="Current Password"
                    type="password"
                    value={passwordData.currentPassword}
                    onChange={(e) => handlePasswordChange('currentPassword', e.target.value)}
                    error={!!passwordErrors.currentPassword}
                    helperText={passwordErrors.currentPassword}
                    sx={{ 
                      '& .MuiOutlinedInput-root': { 
                        bgcolor: '#2d3748',
                        color: '#e2e8f0'
                      },
                      '& .MuiInputLabel-root': { color: '#a0aec0' }
                    }}
                  />
                  <TextField
                    label="New Password"
                    type="password"
                    value={passwordData.newPassword}
                    onChange={(e) => handlePasswordChange('newPassword', e.target.value)}
                    error={!!passwordErrors.newPassword}
                    helperText={passwordErrors.newPassword}
                    sx={{ 
                      '& .MuiOutlinedInput-root': { 
                        bgcolor: '#2d3748',
                        color: '#e2e8f0'
                      },
                      '& .MuiInputLabel-root': { color: '#a0aec0' }
                    }}
                  />
                  <TextField
                    label="Confirm New Password"
                    type="password"
                    value={passwordData.confirmPassword}
                    onChange={(e) => handlePasswordChange('confirmPassword', e.target.value)}
                    error={!!passwordErrors.confirmPassword}
                    helperText={passwordErrors.confirmPassword}
                    sx={{ 
                      '& .MuiOutlinedInput-root': { 
                        bgcolor: '#2d3748',
                        color: '#e2e8f0'
                      },
                      '& .MuiInputLabel-root': { color: '#a0aec0' }
                    }}
                  />
                  <Button 
                    variant="contained"
                    onClick={handleChangePassword}
                    sx={{ 
                      bgcolor: '#4299e1',
                      '&:hover': { bgcolor: '#3182ce' },
                      alignSelf: 'flex-start'
                    }}
                  >
                    Change Password
                  </Button>
                </Box>
              </CardContent>
            </Card>

            <Card sx={{ bgcolor: "#1a202c", border: "1px solid #2d3748" }}>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2, color: "#e2e8f0" }}>
                  Multi-Factor Authentication
                </Typography>
                
                {securitySettings.mfaEnabled ? (
                  <Typography variant="body2" sx={{ color: '#48bb78', mb: 2 }}>
                    MFA is enabled for your account!
                  </Typography>
                ) : mfaQrCode ? (
                  <Box>
                    <Typography variant="body2" sx={{ mb: 2, color: '#a0aec0' }}>
                      Scan this QR code with the Microsoft Authenticator app:
                    </Typography>
                    <Box sx={{ textAlign: 'center', mb: 2 }}>
                      <img src={mfaQrCode} alt="MFA QR Code" style={{ maxWidth: '200px', height: 'auto' }} />
                    </Box>
                    <TextField
                      label="Enter code from Authenticator"
                      value={mfaCode}
                      onChange={(e) => setMfaCode(e.target.value)}
                      sx={{ 
                        mb: 2,
                        '& .MuiOutlinedInput-root': { 
                          bgcolor: '#2d3748',
                          color: '#e2e8f0'
                        },
                        '& .MuiInputLabel-root': { color: '#a0aec0' }
                      }}
                    />
                    <Button 
                      variant="contained"
                      onClick={handleVerifyMFA}
                      sx={{ 
                        bgcolor: '#4299e1',
                        '&:hover': { bgcolor: '#3182ce' }
                      }}
                    >
                      Verify and Enable
                    </Button>
                  </Box>
                ) : (
                  <Button 
                    variant="contained"
                    onClick={handleEnableMFA}
                    sx={{ 
                      bgcolor: '#4299e1',
                      '&:hover': { bgcolor: '#3182ce' }
                    }}
                  >
                    Enable MFA
                  </Button>
                )}
              </CardContent>
            </Card>
          </Box>
        );

      case "notifications":
        return (
          <Box>
            <Typography variant="h5" sx={{ mb: 3, color: "#e2e8f0", fontWeight: 600 }}>
              Notification Settings
            </Typography>
            <Card sx={{ bgcolor: "#1a202c", border: "1px solid #2d3748" }}>
              <CardContent>
                {/* À implémenter */}
              </CardContent>
            </Card>
          </Box>
        );

      case "player":
        return (
          <Box>
            <Typography variant="h5" sx={{ mb: 3, color: "#e2e8f0", fontWeight: 600 }}>
              Player Settings
            </Typography>
            <Card sx={{ bgcolor: "#1a202c", border: "1px solid #2d3748" }}>
              <CardContent>
                {/* À implémenter */}
              </CardContent>
            </Card>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Box sx={{ display: "flex", height: "100vh", bgcolor: "#0f172a" }}>
      <Box
        sx={{
          width: 280,
          bgcolor: "#1a202c",
          borderRight: "1px solid #2d3748",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Box sx={{ p: 3, borderBottom: "1px solid #2d3748" }}>
          <Typography variant="h6" sx={{ color: "#e2e8f0", fontWeight: 600 }}>
            Account Settings
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

      <Box sx={{ flexGrow: 1, p: 4, overflow: "auto" }}>
        {renderContent()}
      </Box>
    </Box>
  );
}