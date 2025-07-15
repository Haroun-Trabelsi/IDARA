"use client"

import React, { useState } from 'react';
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
  Switch,
  FormControlLabel,
  Avatar,
  IconButton,
  Alert,
  Grid
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
  const { account, token } = useAuth();
  const [activeMenu, setActiveMenu] = useState("account");
  const [editMode, setEditMode] = useState(false);
  const [profileData, setProfileData] = useState({
    name: account?.name || 'John',
    surname: account?.surname || 'Doe',
    email: account?.email || 'ines.dahmani@esprit.tn'
  });
  
  const [notificationSettings, setNotificationSettings] = useState({
    generalEmail: true,
    pushNotifications: false,
    assetManagerCollectionCreated: true,
    assetManagerAssetCreated: true,
    assetManagerAssetLinkedToCollection: false,
    assetManagerAssetLinkedToProject: false,
    assetManagerAssetRemoved: true,
    assetManagerCollectionRemoved: true,
    assetManagerTransformationFailed: true
  });

  const [playerSettings, setPlayerSettings] = useState({
    autoplay: true,
    quality: 'HD',
    volume: 75,
    subtitles: false
  });

  const [securitySettings, setSecuritySettings] = useState({
    mfaEnabled: !!account?.mfaEnabled,
    sessionTimeout: 30,
    passwordExpiry: 90
  });

  const [passwordData, setPasswordData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: ''
  });

  const [mfaQrCode, setMfaQrCode] = useState<string | null>(null);
  const [mfaSecret, setMfaSecret] = useState<string | null>(null);
  const [mfaCode, setMfaCode] = useState('');
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);

  const handleSaveProfile = () => {
    console.log('Profile saved:', profileData);
    setEditMode(false);
    setMessage({ type: 'success', text: 'Profile updated successfully!' });
    setTimeout(() => setMessage(null), 3000);
  };

  const handleChangePassword = () => {
    if (passwordData.newPassword !== passwordData.confirmPassword) {
      setMessage({ type: 'error', text: 'Passwords do not match' });
      return;
    }
    console.log('Password changed');
    setPasswordData({ currentPassword: '', newPassword: '', confirmPassword: '' });
    setMessage({ type: 'success', text: 'Password changed successfully!' });
    setTimeout(() => setMessage(null), 3000);
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
                    {profileData.name[0].toUpperCase()}
                  </Avatar>
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" sx={{ color: "#e2e8f0", mb: 1 }}>
                      {`${profileData.name} ${profileData.surname}`}
                    </Typography>
                    <Typography variant="body2" sx={{ color: "#a0aec0" }}>
                      {profileData.email}
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
                    onChange={(e) => setProfileData(prev => ({ ...prev, name: e.target.value }))}
                    disabled={!editMode}
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
                    onChange={(e) => setProfileData(prev => ({ ...prev, surname: e.target.value }))}
                    disabled={!editMode}
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
                    onChange={(e) => setProfileData(prev => ({ ...prev, email: e.target.value }))}
                    disabled={!editMode}
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

      case "notifications":
        return (
          <Box>
            <Typography variant="h5" sx={{ mb: 3, color: "#e2e8f0", fontWeight: 600 }}>
              Notification Settings
            </Typography>
            
            <Card sx={{ bgcolor: "#1a202c", border: "1px solid #2d3748" }}>
              <CardContent>
                <Typography variant="body2" sx={{ mb: 3, color: "#a0aec0" }}>
                  These general settings apply to your project in all Unity products.
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid>
                    <Box>
                      <Typography variant="h6" sx={{ mb: 2, color: "#e2e8f0" }}>
                        Email
                      </Typography>
                      <FormControlLabel
                        control={
                          <Switch 
                            checked={notificationSettings.generalEmail}
                            onChange={(e) => setNotificationSettings(prev => ({ ...prev, generalEmail: e.target.checked }))}
                            sx={{ '& .MuiSwitch-switchBase.Mui-checked': { color: '#4299e1' } }}
                          />
                        }
                        label="General Email"
                        sx={{ color: '#e2e8f0' }}
                      />
                    </Box>
                    <Box sx={{ mt: 3 }}>
                      <Typography variant="h6" sx={{ mb: 2, color: "#e2e8f0" }}>
                        Chat
                      </Typography>
                      <FormControlLabel
                        control={
                          <Switch 
                            checked={notificationSettings.pushNotifications}
                            onChange={(e) => setNotificationSettings(prev => ({ ...prev, pushNotifications: e.target.checked }))}
                            sx={{ '& .MuiSwitch-switchBase.Mui-checked': { color: '#4299e1' } }}
                          />
                        }
                        label="Push Notifications"
                        sx={{ color: '#e2e8f0' }}
                      />
                    </Box>
                  </Grid>
                  <Grid>
                    <Box>
                      <Typography variant="h6" sx={{ mb: 2, color: "#e2e8f0" }}>
                        Asset Manager
                      </Typography>
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        <FormControlLabel
                          control={
                            <Switch 
                              checked={notificationSettings.assetManagerCollectionCreated}
                              onChange={(e) => setNotificationSettings(prev => ({ ...prev, assetManagerCollectionCreated: e.target.checked }))}
                              sx={{ '& .MuiSwitch-switchBase.Mui-checked': { color: '#4299e1' } }}
                            />
                          }
                          label="Collection Created"
                          sx={{ color: '#e2e8f0' }}
                        />
                        <FormControlLabel
                          control={
                            <Switch 
                              checked={notificationSettings.assetManagerAssetCreated}
                              onChange={(e) => setNotificationSettings(prev => ({ ...prev, assetManagerAssetCreated: e.target.checked }))}
                              sx={{ '& .MuiSwitch-switchBase.Mui-checked': { color: '#4299e1' } }}
                            />
                          }
                          label="Asset Created"
                          sx={{ color: '#e2e8f0' }}
                        />
                        <FormControlLabel
                          control={
                            <Switch 
                              checked={notificationSettings.assetManagerAssetLinkedToCollection}
                              onChange={(e) => setNotificationSettings(prev => ({ ...prev, assetManagerAssetLinkedToCollection: e.target.checked }))}
                              sx={{ '& .MuiSwitch-switchBase.Mui-checked': { color: '#4299e1' } }}
                            />
                          }
                          label="Asset Linked To Collection"
                          sx={{ color: '#e2e8f0' }}
                        />
                        <FormControlLabel
                          control={
                            <Switch 
                              checked={notificationSettings.assetManagerAssetLinkedToProject}
                              onChange={(e) => setNotificationSettings(prev => ({ ...prev, assetManagerAssetLinkedToProject: e.target.checked }))}
                              sx={{ '& .MuiSwitch-switchBase.Mui-checked': { color: '#4299e1' } }}
                            />
                          }
                          label="Asset Linked To Project"
                          sx={{ color: '#e2e8f0' }}
                        />
                        <FormControlLabel
                          control={
                            <Switch 
                              checked={notificationSettings.assetManagerAssetRemoved}
                              onChange={(e) => setNotificationSettings(prev => ({ ...prev, assetManagerAssetRemoved: e.target.checked }))}
                              sx={{ '& .MuiSwitch-switchBase.Mui-checked': { color: '#4299e1' } }}
                            />
                          }
                          label="Asset Removed"
                          sx={{ color: '#e2e8f0' }}
                        />
                        <FormControlLabel
                          control={
                            <Switch 
                              checked={notificationSettings.assetManagerCollectionRemoved}
                              onChange={(e) => setNotificationSettings(prev => ({ ...prev, assetManagerCollectionRemoved: e.target.checked }))}
                              sx={{ '& .MuiSwitch-switchBase.Mui-checked': { color: '#4299e1' } }}
                            />
                          }
                          label="Collection Removed"
                          sx={{ color: '#e2e8f0' }}
                        />
                        <FormControlLabel
                          control={
                            <Switch 
                              checked={notificationSettings.assetManagerTransformationFailed}
                              onChange={(e) => setNotificationSettings(prev => ({ ...prev, assetManagerTransformationFailed: e.target.checked }))}
                              sx={{ '& .MuiSwitch-switchBase.Mui-checked': { color: '#4299e1' } }}
                            />
                          }
                          label="Transformation Failed"
                          sx={{ color: '#e2e8f0' }}
                        />
                      </Box>
                    </Box>
                  </Grid>
                </Grid>
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
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  <FormControlLabel
                    control={
                      <Switch 
                        checked={playerSettings.autoplay}
                        onChange={(e) => setPlayerSettings(prev => ({ ...prev, autoplay: e.target.checked }))}
                        sx={{ '& .MuiSwitch-switchBase.Mui-checked': { color: '#4299e1' } }}
                      />
                    }
                    label="Autoplay"
                    sx={{ color: '#e2e8f0' }}
                  />
                  
                  <Box>
                    <Typography variant="body1" sx={{ color: '#e2e8f0', mb: 1 }}>
                      Video Quality
                    </Typography>
                    <TextField
                      select
                      value={playerSettings.quality}
                      onChange={(e) => setPlayerSettings(prev => ({ ...prev, quality: e.target.value }))}
                      SelectProps={{ native: true }}
                      sx={{ 
                        '& .MuiOutlinedInput-root': { 
                          bgcolor: '#2d3748',
                          color: '#e2e8f0'
                        }
                      }}
                    >
                      <option value="HD">HD</option>
                      <option value="Full HD">Full HD</option>
                      <option value="4K">4K</option>
                    </TextField>
                  </Box>

                  <Box>
                    <Typography variant="body1" sx={{ color: '#e2e8f0', mb: 1 }}>
                      Volume: {playerSettings.volume}%
                    </Typography>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={playerSettings.volume}
                      onChange={(e) => setPlayerSettings(prev => ({ ...prev, volume: parseInt(e.target.value) }))}
                      style={{ width: '100%', accentColor: '#4299e1' }}
                    />
                  </Box>

                  <FormControlLabel
                    control={
                      <Switch 
                        checked={playerSettings.subtitles}
                        onChange={(e) => setPlayerSettings(prev => ({ ...prev, subtitles: e.target.checked }))}
                        sx={{ '& .MuiSwitch-switchBase.Mui-checked': { color: '#4299e1' } }}
                      />
                    }
                    label="Subtitles"
                    sx={{ color: '#e2e8f0' }}
                  />
                </Box>
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
                    onChange={(e) => setPasswordData(prev => ({ ...prev, currentPassword: e.target.value }))}
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
                    onChange={(e) => setPasswordData(prev => ({ ...prev, newPassword: e.target.value }))}
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
                    onChange={(e) => setPasswordData(prev => ({ ...prev, confirmPassword: e.target.value }))}
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