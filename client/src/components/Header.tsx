"use client"

import React, { useState, Fragment, MouseEventHandler, useEffect } from "react"
import { AppBar, Toolbar, Button, IconButton, Box, Menu, MenuItem, Avatar, Popover, List, ListSubheader, ListItemButton } from "@mui/material"
import { KeyboardArrowDown, Bookmark, Notifications, Settings, Search, ViewList } from "@mui/icons-material"
import { useAuth } from 'contexts/AuthContext'
import { useModalStore } from 'store/useModalStore'
import OnlineIndicator from 'components/OnlineIndicator'

export default function Header() {
  const { isLoggedIn, account, logout } = useAuth()
  const { setCurrentModal } = useModalStore()

  const [projectsAnchorEl, setProjectsAnchorEl] = useState<null | HTMLElement>(null)
  const [bookmarksAnchorEl, setBookmarksAnchorEl] = useState<null | HTMLElement>(null)
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null)
  const [popover, setPopover] = useState(false)
  const [projects, setProjects] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const response = await fetch("http://localhost:8080/api/projects")
        if (!response.ok) {
          throw new Error(`Failed to fetch projects: ${response.status}`)
        }
        const data = await response.json()
        setProjects(data)
      } catch (err) {
        console.error("❌ Fetch error:", err)
        setError(err instanceof Error ? err.message : 'Failed to fetch projects')
      } finally {
        setLoading(false)
      }
    }

    fetchProjects()
  }, [])

  const openPopover: MouseEventHandler<HTMLButtonElement> = (e) => {
    setPopover(true)
    setAnchorEl(e.currentTarget)
  }

  const closePopover = () => {
    setPopover(false)
    setAnchorEl(null)
  }

  const clickLogin = () => {
    setCurrentModal('LOGIN')
    closePopover()
  }

  const clickRegister = () => {
    setCurrentModal('REGISTER')
    closePopover()
  }

  return (
    <AppBar position="static" elevation={0} sx={{ bgcolor: "#2d3748", borderBottom: "1px solid #4a5568" }}>
      <Toolbar sx={{ justifyContent: "space-between", minHeight: "48px !important", px: 2 }}>

        {/* Left Navigation */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 3 }}>
          <Button 
            endIcon={<KeyboardArrowDown sx={{ fontSize: "16px" }} />} 
            onClick={(e) => setProjectsAnchorEl(e.currentTarget)} 
            sx={{ 
              color: "#e2e8f0", 
              textTransform: "none", 
              fontSize: "14px", 
              fontWeight: 400, 
              "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" } 
            }}
          >
            Projects
          </Button>
          
          <Menu 
            anchorEl={projectsAnchorEl} 
            open={Boolean(projectsAnchorEl)} 
            onClose={() => setProjectsAnchorEl(null)} 
            PaperProps={{ 
              sx: { 
                bgcolor: "#2d3748", 
                border: "1px solid #4a5568",
                maxHeight: 300,
                width: 200
              } 
            }}
          >
            {loading ? (
              <MenuItem sx={{ color: "#e2e8f0" }}>Loading projects...</MenuItem>
            ) : error ? (
              <MenuItem sx={{ color: "#e2e8f0" }}>Error loading projects</MenuItem>
            ) : projects.length === 0 ? (
              <MenuItem sx={{ color: "#e2e8f0" }}>No projects found</MenuItem>
            ) : (
              projects.map((project, index) => (
                <MenuItem 
                  key={index} 
                  sx={{ color: "#e2e8f0" }}
                  onClick={() => {
                    // Handle project selection here
                    setProjectsAnchorEl(null)
                  }}
                >
                  {project}
                </MenuItem>
              ))
            )}
          </Menu>

          <Button sx={{ color: "#e2e8f0", textTransform: "none", fontSize: "14px", fontWeight: 400, "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" } }}>
            My Tasks
          </Button>

          <Button startIcon={<Bookmark sx={{ fontSize: "16px" }} />} endIcon={<KeyboardArrowDown sx={{ fontSize: "16px" }} />} onClick={(e) => setBookmarksAnchorEl(e.currentTarget)} sx={{ color: "#e2e8f0", textTransform: "none", fontSize: "14px", fontWeight: 400, "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" } }}>
            Bookmarks
          </Button>
          <Menu anchorEl={bookmarksAnchorEl} open={Boolean(bookmarksAnchorEl)} onClose={() => setBookmarksAnchorEl(null)} PaperProps={{ sx: { bgcolor: "#2d3748", border: "1px solid #4a5568" } }}>
            <MenuItem sx={{ color: "#e2e8f0" }}>Bookmark 1</MenuItem>
            <MenuItem sx={{ color: "#e2e8f0" }}>Bookmark 2</MenuItem>
          </Menu>
        </Box>

        {/* Right Icons */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <IconButton size="small" sx={{ color: "#a0aec0", "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)", color: "#e2e8f0" } }}>
            <Notifications sx={{ fontSize: "18px" }} />
          </IconButton>

          <IconButton size="small" sx={{ color: "#a0aec0", "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)", color: "#e2e8f0" } }}>
            <ViewList sx={{ fontSize: "18px" }} />
          </IconButton>

          <IconButton size="small" sx={{ color: "#a0aec0", "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)", color: "#e2e8f0" } }}>
            <Search sx={{ fontSize: "18px" }} />
          </IconButton>

          <IconButton size="small" sx={{ color: "#a0aec0", "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)", color: "#e2e8f0" } }}>
            <Settings sx={{ fontSize: "18px" }} />
          </IconButton>

          <IconButton onClick={openPopover}>
            <OnlineIndicator online={isLoggedIn}>
              <Avatar sx={{ width: 28, height: 28, bgcolor: "#4299e1", fontSize: "12px", ml: 1 }} src={account?.username || ''} alt={account?.username || 'Guest'}>
                {account?.username ? account.username[0].toUpperCase() : 'G'}
              </Avatar>
            </OnlineIndicator>
          </IconButton>

          <Popover
            anchorEl={anchorEl}
            open={popover}
            onClose={closePopover}
            anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
            transformOrigin={{ vertical: "top", horizontal: "right" }}
          >
            <List style={{ minWidth: "100px" }}>
              <ListSubheader style={{ textAlign: "center" }}>
                Hello, {account?.username || "Guest"}
              </ListSubheader>

              {isLoggedIn ? (
                <ListItemButton onClick={logout}>Logout</ListItemButton>
              ) : (
                <Fragment>
                  <ListItemButton onClick={clickLogin}>Login</ListItemButton>
                  <ListItemButton onClick={clickRegister}>Register</ListItemButton>
                </Fragment>
              )}
            </List>
          </Popover>

        </Box>
      </Toolbar>
    </AppBar>
  )
}
