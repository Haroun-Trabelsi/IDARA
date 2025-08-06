'use client';
import React, { useEffect, useState } from 'react';
import { Box, Typography, ThemeProvider, createTheme, CssBaseline, Table, TableBody, TableCell, TableHead, TableRow, Paper } from '@mui/material';
import axios from 'axios';
import Toolbar from 'components/Project-components/Toolbar';
import SecondaryNavigation from 'components/Project-components/SecondaryNavigation';
import Sidebar from 'components/Project-components/Sidebar';

// thème dark identique
const darkTheme = createTheme({ palette: { mode: 'dark', primary: { main: '#4299e1' }, secondary: { main: '#f59e0b' },
  background: { default: '#0f172a', paper: '#1a202c' },
  text: { primary: '#e2e8f0', secondary: '#a0aec0' } },
  components: { MuiCssBaseline: { styleOverrides: { body: { backgroundColor: '#0f172a' } } }}});

interface Message { _id: string; name: string; email: string; subject: string; message: string; createdAt: string; }

export default function AdminMessages() {
  const [messages, setMessages] = useState<Message[]>([]);
  
  useEffect(() => {
    axios.get("http://localhost:8080/contact/")
      .then(res => {
        if (res.data.success) setMessages(res.data.data);
      })
      .catch(console.error);
  }, []);

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', height: '100vh', flexDirection: 'column' }}>
        <Box sx={{ display: 'flex', flexGrow: 1, overflow: 'hidden' }}>
          <Sidebar />
          <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <SecondaryNavigation />
            <Toolbar />
            <Box sx={{ p: 2, color: '#e2e8f0' }}>
              <Typography variant="h5" gutterBottom>Messages de contact ({messages.length})</Typography>
              <Paper sx={{ width: '100%', overflow: 'auto', mt: 2 }}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Nom</TableCell><TableCell>Email</TableCell><TableCell>Sujet</TableCell><TableCell>Message</TableCell><TableCell>Date</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {messages.map(msg => (
                      <TableRow key={msg._id}>
                        <TableCell>{msg.name}</TableCell>
                        <TableCell>{msg.email}</TableCell>
                        <TableCell>{msg.subject}</TableCell>
                        <TableCell>{msg.message}</TableCell>
                        <TableCell>{new Date(msg.createdAt).toLocaleString()}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Paper>
            </Box>
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
}
